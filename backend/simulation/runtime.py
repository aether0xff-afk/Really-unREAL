from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol

from backend.fusion import EvidenceMessage, PersonEvidence
from backend.generation import BurstLanguageModel, GeneratedBurst
from backend.generation_context import build_generation_context
from backend.live_behavior import LatentReadTimingModel, LiveResponsePolicy, TargetContinuationPolicy
from backend.live_timing import BurstGapSampler
from backend.models import ChatMessage, MemorySource
from backend.providers.errors import PermanentGenerationError, TransientGenerationError
from backend.replay import ReplayCase
from backend.replay_baseline import EmpiricalTimingBaseline
from backend.retrieval import CutoffExampleIndex
from backend.simulation.action_policy import Action, MESSAGE_ACTIONS
from backend.simulation.store import SQLiteSimulationStore, ScheduledEvent


_RETRY_DELAYS_SECONDS = (5, 15, 30, 60, 120, 300)


class TimingSampler(Protocol):
    def sample_delay_seconds(
        self,
        *,
        platform: str,
        conversation_id: str,
        action: Action,
        observed_at: datetime | None = None,
        visible_context: Sequence[EvidenceMessage] = (),
    ) -> float | None: ...


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
    return baseline.delay_for_action(
        conversation_id=conversation_id,
        platform=platform,
        action=action,
    )


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
        evidence_context=(
            context[-1].context if context else next(iter(_direct_contexts(platform)))
        ),
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
        session_restart=(action == Action.INITIATE),
        action_is_ambiguous=False,
    )


def _direct_contexts(platform: str):
    from backend.fusion import EvidenceContext

    if platform == "kakao":
        return (EvidenceContext.KAKAO_DIRECT,)
    return (EvidenceContext.INSTAGRAM_DIRECT,)


class LiveSimulationEngine:
    """Persistent discrete-event runtime with behavior/provider separation."""

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
        timing_sampler: TimingSampler | None = None,
        response_policy: LiveResponsePolicy | None = None,
        continuation_policy: TargetContinuationPolicy | None = None,
        read_timing_model: LatentReadTimingModel | None = None,
        burst_gap_sampler: BurstGapSampler | None = None,
        raw_response_examples: int = 0,
    ) -> None:
        if evidence.person_id != twin_person_id:
            raise ValueError("evidence and twin_person_id must match")
        if raw_response_examples < 0:
            raise ValueError("raw_response_examples must be >= 0")
        self.twin_person_id = twin_person_id
        self.platform = platform
        self.conversation_id = conversation_id
        self.evidence = evidence
        self.retrieval_index = retrieval_index
        self.timing = timing
        self.timing_sampler = timing_sampler
        self.response_policy = response_policy
        self.continuation_policy = continuation_policy
        self.read_timing_model = read_timing_model
        self.burst_gap_sampler = burst_gap_sampler
        self.language_model = language_model
        self.store = store
        self.raw_response_examples = int(raw_response_examples)

    def _next_delay(
        self,
        action: Action,
        *,
        observed_at: datetime,
        visible_context: Sequence[EvidenceMessage] = (),
    ) -> float | None:
        if self.timing_sampler is not None:
            sampled = self.timing_sampler.sample_delay_seconds(
                platform=self.platform,
                conversation_id=self.conversation_id,
                action=action,
                observed_at=observed_at,
                visible_context=visible_context,
            )
            if sampled is not None:
                return max(0.0, float(sampled))
        return _delay_for(
            self.timing,
            platform=self.platform,
            conversation_id=self.conversation_id,
            action=action,
        )

    def observe_counterpart_message(
        self,
        *,
        observed_at: datetime,
        visible_context: Sequence[EvidenceMessage] = (),
        replace_existing_reply: bool = True,
    ) -> ScheduledEvent | None:
        # A real/simulation user message supersedes only future idle behaviors.
        # An already CLAIMED generation is never cancelled by a later message.
        for idle_action in (Action.FOLLOW_UP, Action.INITIATE):
            self.store.cancel_pending(
                twin_person_id=self.twin_person_id,
                platform=self.platform,
                conversation_id=self.conversation_id,
                action=idle_action,
            )

        chosen = (
            self.response_policy.choose_after_counterpart_message(
                observed_at=observed_at,
                visible_context=visible_context,
            )
            if self.response_policy is not None
            else Action.REPLY
        )
        if chosen == Action.WAIT:
            return None

        delay = self._next_delay(
            Action.REPLY,
            observed_at=observed_at,
            visible_context=visible_context,
        )
        if delay is None:
            return None

        reply = self.store.schedule(
            twin_person_id=self.twin_person_id,
            platform=self.platform,
            conversation_id=self.conversation_id,
            action=Action.REPLY,
            due_at=observed_at + timedelta(seconds=delay),
            created_at=observed_at,
            replace_same_action=replace_existing_reply,
        )

        if self.read_timing_model is not None:
            read_delay = self.read_timing_model.sample_delay_seconds(delay)
            self.store.schedule(
                twin_person_id=self.twin_person_id,
                platform=self.platform,
                conversation_id=self.conversation_id,
                action=Action.READ,
                due_at=observed_at + timedelta(seconds=read_delay),
                created_at=observed_at,
                replace_same_action=True,
            )
        return reply

    def _schedule_idle_action(
        self,
        action: Action,
        *,
        after: datetime,
        visible_context: Sequence[EvidenceMessage],
    ) -> ScheduledEvent | None:
        if action not in {Action.FOLLOW_UP, Action.INITIATE}:
            return None
        if self.continuation_policy is not None:
            chosen = self.continuation_policy.choose(
                action,
                observed_at=after,
                visible_context=visible_context,
            )
            if chosen == Action.WAIT:
                return None
        delay = self._next_delay(
            action,
            observed_at=after,
            visible_context=visible_context,
        )
        if delay is None:
            return None
        return self.store.schedule(
            twin_person_id=self.twin_person_id,
            platform=self.platform,
            conversation_id=self.conversation_id,
            action=action,
            due_at=after + timedelta(seconds=delay),
            created_at=after,
            replace_same_action=True,
        )

    def schedule_idle_initiation(
        self,
        *,
        after: datetime,
        visible_context: Sequence[EvidenceMessage] = (),
    ) -> ScheduledEvent | None:
        return self._schedule_idle_action(
            Action.INITIATE,
            after=after,
            visible_context=visible_context,
        )

    def schedule_idle_follow_up(
        self,
        *,
        after: datetime,
        visible_context: Sequence[EvidenceMessage] = (),
    ) -> ScheduledEvent | None:
        return self._schedule_idle_action(
            Action.FOLLOW_UP,
            after=after,
            visible_context=visible_context,
        )

    def _post_generation_context(
        self,
        event_context: tuple[EvidenceMessage, ...],
        timestamped: tuple[tuple[datetime, str], ...],
    ) -> tuple[EvidenceMessage, ...]:
        evidence_context = (
            event_context[-1].context
            if event_context
            else next(iter(_direct_contexts(self.platform)))
        )
        generated = tuple(
            EvidenceMessage(
                message=ChatMessage(
                    timestamp=timestamp,
                    sender=self.twin_person_id,
                    text=text,
                    source=MemorySource.SIMULATION,
                ),
                platform=self.platform,
                conversation_id=self.conversation_id,
                context=evidence_context,
                sender_person_id=self.twin_person_id,
                evidence_weight=1.0,
            )
            for timestamp, text in timestamped
        )
        return event_context + generated

    def _timestamp_burst(
        self,
        event: ScheduledEvent,
        burst: GeneratedBurst,
    ) -> tuple[tuple[datetime, str], ...]:
        gaps = (
            self.burst_gap_sampler.sample_gaps(
                conversation_id=self.conversation_id,
                count=len(burst.messages),
            )
            if self.burst_gap_sampler is not None
            else tuple(0.0 for _ in range(max(0, len(burst.messages) - 1)))
        )
        output: list[tuple[datetime, str]] = []
        timestamp = event.due_at
        for index, text in enumerate(burst.messages):
            if index > 0:
                timestamp += timedelta(seconds=max(0.0, gaps[index - 1]))
            output.append((timestamp, text))
        return tuple(output)

    def _defer_transient(
        self,
        event: ScheduledEvent,
        *,
        now: datetime,
        error: Exception,
    ) -> None:
        seconds = _RETRY_DELAYS_SECONDS[
            min(event.generation_attempts, len(_RETRY_DELAYS_SECONDS) - 1)
        ]
        self.store.defer_event(
            event.event_id,
            retry_at=now + timedelta(seconds=seconds),
            error=str(error),
        )

    def process_due(
        self,
        *,
        now: datetime,
        visible_context: tuple[EvidenceMessage, ...],
    ) -> list[SimulationEmission]:
        emissions: list[SimulationEmission] = []
        claimed = self.store.claim_due_events(
            now=now,
            twin_person_id=self.twin_person_id,
            platform=self.platform,
            conversation_id=self.conversation_id,
        )

        for event in claimed:
            if event.action == Action.READ:
                self.store.mark_messages_read(
                    twin_person_id=self.twin_person_id,
                    platform=self.platform,
                    conversation_id=self.conversation_id,
                    sender_person_id="self",
                    read_at=event.due_at,
                    sent_before_or_at=event.due_at,
                    source="SIMULATION_LATENT_READ",
                )
                try:
                    self.store.complete_claimed_event(event.event_id)
                except KeyError:
                    pass
                continue

            if event.action not in MESSAGE_ACTIONS:
                try:
                    self.store.complete_claimed_event(event.event_id)
                except KeyError:
                    pass
                continue

            # Causal cutoff is the modeled behavior time, never provider retry time.
            event_context = tuple(
                item for item in visible_context if item.message.timestamp <= event.due_at
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
                raw_response_examples=self.raw_response_examples,
            )

            try:
                burst = self.language_model.generate_burst(packet)
            except TransientGenerationError as exc:
                self._defer_transient(event, now=now, error=exc)
                continue
            except PermanentGenerationError as exc:
                self.store.block_event(event.event_id, error=str(exc))
                continue
            except Exception as exc:
                self.store.block_event(event.event_id, error=str(exc))
                continue

            timestamped = self._timestamp_burst(event, burst)
            try:
                self.store.complete_claimed_event_with_messages(
                    event_id=event.event_id,
                    twin_person_id=self.twin_person_id,
                    platform=self.platform,
                    conversation_id=self.conversation_id,
                    sender_person_id=self.twin_person_id,
                    messages=timestamped,
                    metadata={"action": event.action.value, "event_id": event.event_id},
                )
            except KeyError:
                # The user may have reset the conversation while generation was
                # in flight. Atomic completion prevents the stale output from
                # leaking back into the new session.
                continue

            emissions.append(
                SimulationEmission(
                    event_id=event.event_id,
                    action=event.action,
                    due_at=event.due_at,
                    generated_at=now,
                    burst=burst,
                )
            )

            if timestamped:
                after = timestamped[-1][0]
                post_context = self._post_generation_context(event_context, timestamped)
                follow = self.schedule_idle_follow_up(
                    after=after,
                    visible_context=post_context,
                )
                if follow is None:
                    self.schedule_idle_initiation(
                        after=after,
                        visible_context=post_context,
                    )

        self.store.set_last_processed(now)
        return emissions

    def recover(
        self,
        *,
        now: datetime,
        visible_context: tuple[EvidenceMessage, ...],
    ) -> list[SimulationEmission]:
        self.store.recover_stale_claims(now=now)
        return self.process_due(now=now, visible_context=visible_context)
