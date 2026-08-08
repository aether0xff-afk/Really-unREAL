from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta

from backend.fusion import EvidenceConversation, EvidenceMessage, PersonEvidence
from backend.generation import BurstLanguageModel, GeneratedBurst, evaluate_generated_burst
from backend.generation_context import build_generation_context
from backend.models import ChatMessage, MemorySource
from backend.replay import ReplayCase
from backend.replay_baseline import EmpiricalTimingBaseline
from backend.retrieval import CutoffExampleIndex
from backend.simulation.action_policy import Action
from backend.simulation.runtime import _delay_for, _generation_case


@dataclass(frozen=True, slots=True)
class ShadowPredictedEvent:
    action: Action
    at: datetime
    burst: GeneratedBurst


@dataclass(frozen=True, slots=True)
class ShadowReport:
    conversation_id: str
    start_at: str
    end_at: str
    real_event_count: int
    predicted_event_count: int
    matched_event_count: int
    event_precision: float
    event_recall: float
    median_absolute_timing_error_seconds: float | None
    mean_char_bigram_f1: float | None
    mean_token_f1: float | None
    mean_ending_f1: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _shadow_evidence(evidence: PersonEvidence, start_at: datetime) -> PersonEvidence:
    """Hide all target-authored future messages while retaining exogenous input."""

    conversations: list[EvidenceConversation] = []
    for conversation in evidence.conversations:
        visible = tuple(
            item
            for item in conversation.messages
            if item.message.timestamp < start_at
            or item.sender_person_id != evidence.person_id
        )
        conversations.append(
            EvidenceConversation(
                platform=conversation.platform,
                conversation_id=conversation.conversation_id,
                context=conversation.context,
                messages=visible,
            )
        )
    return PersonEvidence(evidence.person_id, tuple(conversations))


def _simulated_evidence_message(
    template: EvidenceMessage,
    *,
    person_id: str,
    at: datetime,
    text: str,
) -> EvidenceMessage:
    return EvidenceMessage(
        message=ChatMessage(
            timestamp=at,
            sender=person_id,
            text=text,
            source=MemorySource.SIMULATION,
            metadata={"platform": template.platform, "shadow": True},
        ),
        platform=template.platform,
        conversation_id=template.conversation_id,
        context=template.context,
        sender_person_id=person_id,
        evidence_weight=template.evidence_weight,
    )


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 6) if values else None


def run_shadow_simulation(
    *,
    evidence: PersonEvidence,
    replay_cases: list[ReplayCase],
    timing: EmpiricalTimingBaseline,
    language_model: BurstLanguageModel,
    conversation_id: str,
    start_at: datetime | None = None,
    end_at: datetime | None = None,
    context_size: int = 30,
) -> tuple[ShadowReport, tuple[ShadowPredictedEvent, ...]]:
    """Replay a past interval without feeding hidden target messages back in.

    Counterpart messages remain exogenous observations, like incoming messages in
    live mode. Target-authored reality after ``start_at`` is withheld from the
    simulator and used only after the run for event/content scoring. Simulated
    target messages become the subsequent visible context, producing a true
    closed-loop drift test instead of independent teacher-forced cases.
    """

    real_cases = sorted(
        [case for case in replay_cases if case.conversation_id == conversation_id],
        key=lambda case: (case.action_at, case.case_id),
    )
    if not real_cases:
        raise ValueError("conversation has no replay cases")
    start_at = start_at or real_cases[0].observation_end
    end_at = end_at or real_cases[-1].action_at
    if end_at <= start_at:
        raise ValueError("shadow end_at must be after start_at")
    real_cases = [case for case in real_cases if start_at < case.action_at <= end_at]
    if not real_cases:
        raise ValueError("shadow interval has no real target events")

    conversation = next(
        (
            item
            for item in evidence.conversations
            if item.conversation_id == conversation_id
        ),
        None,
    )
    if conversation is None:
        raise ValueError("conversation_id not found in evidence")

    safe_evidence = _shadow_evidence(evidence, start_at)
    pre_shadow_cases = [case for case in replay_cases if case.action_at < start_at]
    index = CutoffExampleIndex.from_replay_cases(pre_shadow_cases)

    visible = [
        item for item in conversation.messages
        if item.message.timestamp < start_at
    ][-context_size:]
    exogenous = [
        item for item in conversation.messages
        if start_at <= item.message.timestamp <= end_at
        and item.sender_person_id != evidence.person_id
    ]
    exogenous_index = 0

    pending_action: Action | None = None
    pending_at: datetime | None = None

    def schedule(action: Action, after: datetime) -> None:
        nonlocal pending_action, pending_at
        delay = _delay_for(
            timing,
            platform=conversation.platform,
            conversation_id=conversation_id,
            action=action,
        )
        if delay is None:
            pending_action = None
            pending_at = None
            return
        pending_action = action
        pending_at = after + timedelta(seconds=delay)

    if visible:
        initial_action = (
            Action.INITIATE
            if visible[-1].sender_person_id == evidence.person_id
            else Action.REPLY
        )
        schedule(initial_action, start_at)

    predicted: list[ShadowPredictedEvent] = []
    template = conversation.messages[0]

    while True:
        next_external = (
            exogenous[exogenous_index].message.timestamp
            if exogenous_index < len(exogenous)
            else None
        )
        candidates = [value for value in (next_external, pending_at) if value is not None]
        if not candidates:
            break
        next_at = min(candidates)
        if next_at > end_at:
            break

        if next_external is not None and next_external <= (pending_at or datetime.max):
            incoming = exogenous[exogenous_index]
            exogenous_index += 1
            visible.append(incoming)
            visible = visible[-context_size:]
            schedule(Action.REPLY, incoming.message.timestamp)
            continue

        assert pending_action is not None and pending_at is not None
        action = pending_action
        due_at = pending_at
        case = _generation_case(
            twin_person_id=evidence.person_id,
            platform=conversation.platform,
            conversation_id=conversation_id,
            action=action,
            due_at=due_at,
            context=tuple(visible),
        )
        packet = build_generation_context(
            case,
            safe_evidence,
            index,
            chosen_action=action,
            action_specific_retrieval=True,
        )
        burst = language_model.generate_burst(packet)
        predicted.append(ShadowPredictedEvent(action=action, at=due_at, burst=burst))
        for offset, text in enumerate(burst.messages):
            visible.append(
                _simulated_evidence_message(
                    template,
                    person_id=evidence.person_id,
                    at=due_at + timedelta(seconds=offset),
                    text=text,
                )
            )
        visible = visible[-context_size:]
        if action == Action.REPLY:
            schedule(Action.INITIATE, due_at)
        else:
            pending_action = None
            pending_at = None

    unused = set(range(len(predicted)))
    timing_errors: list[float] = []
    bigram_scores: list[float] = []
    token_scores: list[float] = []
    ending_scores: list[float] = []
    matched = 0

    for real in real_cases:
        candidates = [
            index
            for index in unused
            if real.action_is_ambiguous or predicted[index].action == real.action
        ]
        if not candidates:
            continue
        best = min(
            candidates,
            key=lambda index: abs((predicted[index].at - real.action_at).total_seconds()),
        )
        unused.remove(best)
        item = predicted[best]
        matched += 1
        timing_errors.append(abs((item.at - real.action_at).total_seconds()))
        metrics = evaluate_generated_burst(item.burst, real)
        bigram_scores.append(metrics.char_bigram_f1)
        token_scores.append(metrics.token_f1)
        ending_scores.append(metrics.ending_f1)

    report = ShadowReport(
        conversation_id=conversation_id,
        start_at=start_at.isoformat(),
        end_at=end_at.isoformat(),
        real_event_count=len(real_cases),
        predicted_event_count=len(predicted),
        matched_event_count=matched,
        event_precision=round(matched / len(predicted), 6) if predicted else 0.0,
        event_recall=round(matched / len(real_cases), 6),
        median_absolute_timing_error_seconds=(
            round(float(statistics.median(timing_errors)), 3) if timing_errors else None
        ),
        mean_char_bigram_f1=_mean(bigram_scores),
        mean_token_f1=_mean(token_scores),
        mean_ending_f1=_mean(ending_scores),
    )
    return report, tuple(predicted)
