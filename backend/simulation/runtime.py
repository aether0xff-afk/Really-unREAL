from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from backend.fusion import EvidenceMessage, PersonEvidence
from backend.generation import BurstLanguageModel, GeneratedBurst
from backend.generation_context import build_generation_context
from backend.replay import ReplayCase
from backend.replay_baseline import EmpiricalTimingBaseline
from backend.retrieval import CutoffExampleIndex
from backend.simulation.action_policy import Action
from backend.simulation.store import SQLiteSimulationStore, ScheduledEvent


@dataclass(frozen=True, slots=True)
class SimulationEmission:
    event_id: str
    action: Action
    due_at: datetime
    generated_at: datetime
    burst: GeneratedBurst


def _delay_for(
    baseline: EmpiricalTimingBaseline,
    *,
    platform: str,
    conversation_id: str,
    action: Action,
) -> float | None:
    if action == Action.WAIT:
        return None
    if (conversation_id, action) in baseline.conversation_thresholds:
        return baseline.conversation_thresholds[(conversation_id, action)]
    if (platform, action) in baseline.platform_thresholds:
        return baseline.platform_thresholds[(platform, action)]
    if action in baseline.action_thresholds:
        return baseline.action_thresholds[action]
    return baseline.global_threshold if action == Action.REPLY else None


def _generation_case(
    *,
    twin_person_id: str,
    platform: str,
    conversation_id: str,
    action: Action,
    due_at: datetime,
    context: tuple[EvidenceMessage, ...],
) -> ReplayCase:
    previous_at = context[-1].message.timestamp if context else due_at
    return ReplayCase(
        case_id=f"live:{conversation_id}:{due_at.isoformat()}:{action.value}",
        person_id=twin_person_id,
        platform=platform,
        conversation_id=conversation_id,
        evidence_context=(context[-1].context if context else next(iter(_direct_contexts(platform)))),
        evidence_weight=1.0,
        action=action,
        observation_end=due_at,
        action_at=due_at,
        observed_delay_seconds=max(0.0, (due_at - previous_at).total_seconds()),
        delay_lower_seconds=0.0,
        delay_upper_seconds=max(0.0, (due_at - previous_at).total_seconds()),
        context=context,
        target_burst=(),
        burst_size=0,
        session_restart=False,
        action_is_ambiguous=False,
    )


def _direct_contexts(platform: str):
    from backend.fusion import EvidenceContext

    if platform == "kakao":
        return (EvidenceContext.KAKAO_DIRECT,)
    return (EvidenceContext.INSTAGRAM_DIRECT,)


class LiveSimulationEngine:
    """One-conversation discrete-event runtime.

    Incoming real messages are external observations. The engine persists only
    future action times. Text generation occurs at dispatch time, so WAIT remains
    a real outcome and closing/reopening the app does not collapse elapsed time.

    The runtime never sends to a real platform. ``process_due`` returns simulated
    emissions to the caller/UI and stores them as SIMULATION memory only.
    """

    def __init__(
        self,
        *,
        twin_person_id: str,
        platform: str,
        conversation_id: str,
        evidence: PersonEvidence,
        retrieval_index: CutoffExampleIndex,
        timing: EmpiricalTimingBaseline,
        language_model: BurstLanguageModel,
        store: SQLiteSimulationStore,
    ) -> None:
        if evidence.person_id != twin_person_id:
            raise ValueError("evidence and twin_person_id must match")
        self.twin_person_id = twin_person_id
        self.platform = platform
        self.conversation_id = conversation_id
        self.evidence = evidence
        self.retrieval_index = retrieval_index
        self.timing = timing
        self.language_model = language_model
        self.store = store

    def observe_counterpart_message(self, *, observed_at: datetime) -> ScheduledEvent:
        self.store.cancel_pending(
            twin_person_id=self.twin_person_id,
            platform=self.platform,
            conversation_id=self.conversation_id,
            action=Action.INITIATE,
        )
        delay = _delay_for(
            self.timing,
            platform=self.platform,
            conversation_id=self.conversation_id,
            action=Action.REPLY,
        )
        if delay is None:
            raise RuntimeError("reply timing is unavailable")
        return self.store.schedule(
            twin_person_id=self.twin_person_id,
            platform=self.platform,
            conversation_id=self.conversation_id,
            action=Action.REPLY,
            due_at=observed_at + timedelta(seconds=delay),
            created_at=observed_at,
        )

    def schedule_idle_initiation(self, *, after: datetime) -> ScheduledEvent | None:
        delay = _delay_for(
            self.timing,
            platform=self.platform,
            conversation_id=self.conversation_id,
            action=Action.INITIATE,
        )
        if delay is None:
            return None
        return self.store.schedule(
            twin_person_id=self.twin_person_id,
            platform=self.platform,
            conversation_id=self.conversation_id,
            action=Action.INITIATE,
            due_at=after + timedelta(seconds=delay),
            created_at=after,
        )

    def process_due(
        self,
        *,
        now: datetime,
        visible_context: tuple[EvidenceMessage, ...],
    ) -> list[SimulationEmission]:
        emissions: list[SimulationEmission] = []
        for event in self.store.due_events(now):
            if (
                event.twin_person_id != self.twin_person_id
                or event.platform != self.platform
                or event.conversation_id != self.conversation_id
            ):
                continue

            # Recovery may happen long after the event became due. Never let an
            # overdue event see real messages that arrived after its own due time.
            event_context = tuple(
                item
                for item in visible_context
                if item.message.timestamp <= event.due_at
            )
            case = _generation_case(
                twin_person_id=self.twin_person_id,
                platform=self.platform,
                conversation_id=self.conversation_id,
                action=event.action,
                due_at=event.due_at,
                context=event_context,
            )
            packet = build_generation_context(
                case,
                self.evidence,
                self.retrieval_index,
                chosen_action=event.action,
                action_specific_retrieval=True,
            )
            burst = self.language_model.generate_burst(packet)
            timestamped = tuple(
                (event.due_at + timedelta(seconds=index), text)
                for index, text in enumerate(burst.messages)
            )
            self.store.append_simulation_messages(
                twin_person_id=self.twin_person_id,
                platform=self.platform,
                conversation_id=self.conversation_id,
                sender_person_id=self.twin_person_id,
                messages=timestamped,
                metadata={"action": event.action.value, "event_id": event.event_id},
            )
            self.store.mark_processed(event.event_id)
            emissions.append(
                SimulationEmission(
                    event_id=event.event_id,
                    action=event.action,
                    due_at=event.due_at,
                    generated_at=now,
                    burst=burst,
                )
            )
            if event.action == Action.REPLY:
                self.schedule_idle_initiation(after=event.due_at)

        self.store.set_last_processed(now)
        return emissions

    def recover(
        self,
        *,
        now: datetime,
        visible_context: tuple[EvidenceMessage, ...],
    ) -> list[SimulationEmission]:
        return self.process_due(now=now, visible_context=visible_context)
