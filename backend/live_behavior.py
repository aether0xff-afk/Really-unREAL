from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Sequence

from backend.fusion import EvidenceContext, EvidenceMessage, PersonEvidence
from backend.live_timing import classify_message_kind, visible_timing_features
from backend.simulation.action_policy import Action


_DIRECT_CONTEXTS = {EvidenceContext.KAKAO_DIRECT, EvidenceContext.INSTAGRAM_DIRECT}


@dataclass(frozen=True, slots=True)
class BinaryBehaviorObservation:
    conversation_id: str
    observed_at: datetime
    positive: bool
    hour_band: int
    weekend: int
    recent_activity: str
    previous_gap: str
    message_kind: str
    evidence_weight: float


class _ShrunkBinaryPolicy:
    def __init__(
        self,
        observations: Iterable[BinaryBehaviorObservation],
        *,
        focus_conversation_id: str,
        seed: int | None,
        minimum_cell_events: int = 3,
        prior_strength: float = 5.0,
        empty_probability: float = 0.0,
    ) -> None:
        self.observations = tuple(observations)
        self.focus_conversation_id = focus_conversation_id
        self.minimum_cell_events = max(2, int(minimum_cell_events))
        self.prior_strength = max(0.0, float(prior_strength))
        self.empty_probability = min(1.0, max(0.0, float(empty_probability)))
        self._rng = random.Random(seed)

    def _rate(self, rows: Sequence[BinaryBehaviorObservation]) -> tuple[float, float]:
        total = sum(max(0.0, row.evidence_weight) for row in rows)
        success = sum(
            max(0.0, row.evidence_weight) for row in rows if row.positive
        )
        return success, total

    @property
    def global_probability(self) -> float:
        success, total = self._rate(self.observations)
        if total <= 0:
            return self.empty_probability
        return (success + 1.0) / (total + 2.0)

    def probability(
        self,
        *,
        observed_at: datetime,
        visible_context: Sequence[EvidenceMessage],
    ) -> float:
        if not self.observations:
            return self.empty_probability
        live = visible_timing_features(observed_at, visible_context)
        global_p = self.global_probability

        levels = [
            [
                row for row in self.observations
                if row.conversation_id == self.focus_conversation_id
                and row.hour_band == live.hour_band
                and row.recent_activity == live.recent_activity
                and row.previous_gap == live.previous_gap
                and row.message_kind == live.last_message_kind
            ],
            [
                row for row in self.observations
                if row.conversation_id == self.focus_conversation_id
                and row.message_kind == live.last_message_kind
            ],
            [
                row for row in self.observations
                if row.conversation_id == self.focus_conversation_id
            ],
            [row for row in self.observations if row.message_kind == live.last_message_kind],
            list(self.observations),
        ]

        for index, rows in enumerate(levels):
            if not rows:
                continue
            if index < len(levels) - 1 and len(rows) < self.minimum_cell_events:
                continue
            success, total = self._rate(rows)
            if total <= 0:
                continue
            probability = (
                success + self.prior_strength * global_p
            ) / (total + self.prior_strength)
            return min(1.0, max(0.0, probability))
        return global_p

    def sample_positive(
        self,
        *,
        observed_at: datetime,
        visible_context: Sequence[EvidenceMessage],
    ) -> bool:
        return self._rng.random() < self.probability(
            observed_at=observed_at,
            visible_context=visible_context,
        )


def _observation(
    *,
    conversation_id: str,
    observed_at: datetime,
    positive: bool,
    visible_context: Sequence[EvidenceMessage],
    evidence_weight: float,
) -> BinaryBehaviorObservation:
    features = visible_timing_features(observed_at, visible_context)
    return BinaryBehaviorObservation(
        conversation_id=conversation_id,
        observed_at=observed_at,
        positive=positive,
        hour_band=features.hour_band,
        weekend=features.weekend,
        recent_activity=features.recent_activity,
        previous_gap=features.previous_gap,
        message_kind=features.last_message_kind,
        evidence_weight=max(0.0, float(evidence_weight)),
    )


class LiveResponsePolicy:
    """Decide REPLY vs WAIT before reply timing or text generation."""

    def __init__(self, base: _ShrunkBinaryPolicy) -> None:
        self._base = base

    @classmethod
    def from_evidence(
        cls,
        evidence: PersonEvidence,
        *,
        self_person_id: str,
        focus_conversation_id: str,
        burst_gap_seconds: float = 120.0,
        session_gap_hours: float = 6.0,
        seed: int | None = None,
    ) -> "LiveResponsePolicy":
        rows: list[BinaryBehaviorObservation] = []
        session_gap_seconds = session_gap_hours * 3600.0

        for conversation in evidence.conversations:
            if conversation.context not in _DIRECT_CONTEXTS:
                continue
            messages = list(conversation.messages)
            index = 0
            while index < len(messages):
                if messages[index].sender_person_id != self_person_id:
                    index += 1
                    continue
                burst_start = index
                burst_end = index + 1
                while burst_end < len(messages):
                    previous = messages[burst_end - 1]
                    current = messages[burst_end]
                    if current.sender_person_id != self_person_id:
                        break
                    gap = (current.message.timestamp - previous.message.timestamp).total_seconds()
                    if gap > burst_gap_seconds:
                        break
                    burst_end += 1
                if burst_end >= len(messages):
                    break

                last_user = messages[burst_end - 1]
                next_visible = messages[burst_end]
                replied = False
                if next_visible.sender_person_id == evidence.person_id:
                    gap = max(
                        0.0,
                        (next_visible.message.timestamp - last_user.message.timestamp).total_seconds(),
                    )
                    replied = gap <= session_gap_seconds

                context_start = max(0, burst_start - 30)
                visible_context = tuple(messages[context_start:burst_end])
                rows.append(
                    _observation(
                        conversation_id=conversation.conversation_id,
                        observed_at=last_user.message.timestamp,
                        positive=replied,
                        visible_context=visible_context,
                        evidence_weight=last_user.evidence_weight,
                    )
                )
                index = burst_end

        return cls(
            _ShrunkBinaryPolicy(
                rows,
                focus_conversation_id=focus_conversation_id,
                seed=seed,
                empty_probability=1.0,
            )
        )

    @property
    def global_reply_probability(self) -> float:
        return self._base.global_probability

    def reply_probability(
        self,
        *,
        observed_at: datetime,
        visible_context: Sequence[EvidenceMessage],
    ) -> float:
        return self._base.probability(
            observed_at=observed_at,
            visible_context=visible_context,
        )

    def choose_after_counterpart_message(
        self,
        *,
        observed_at: datetime,
        visible_context: Sequence[EvidenceMessage],
    ) -> Action:
        return (
            Action.REPLY
            if self._base.sample_positive(
                observed_at=observed_at,
                visible_context=visible_context,
            )
            else Action.WAIT
        )


class TargetContinuationPolicy:
    """Decide whether the target sends FOLLOW_UP / later INITIATE or stays silent."""

    def __init__(
        self,
        *,
        follow_up: _ShrunkBinaryPolicy,
        initiate: _ShrunkBinaryPolicy,
    ) -> None:
        self.follow_up = follow_up
        self.initiate = initiate

    @classmethod
    def from_evidence(
        cls,
        evidence: PersonEvidence,
        *,
        focus_conversation_id: str,
        burst_gap_seconds: float = 120.0,
        session_gap_hours: float = 6.0,
        seed: int | None = None,
    ) -> "TargetContinuationPolicy":
        follow_rows: list[BinaryBehaviorObservation] = []
        initiate_rows: list[BinaryBehaviorObservation] = []
        session_gap_seconds = session_gap_hours * 3600.0

        for conversation in evidence.conversations:
            if conversation.context not in _DIRECT_CONTEXTS:
                continue
            messages = list(conversation.messages)
            index = 0
            while index < len(messages):
                if messages[index].sender_person_id != evidence.person_id:
                    index += 1
                    continue
                burst_start = index
                burst_end = index + 1
                while burst_end < len(messages):
                    previous = messages[burst_end - 1]
                    current = messages[burst_end]
                    if current.sender_person_id != evidence.person_id:
                        break
                    gap = (current.message.timestamp - previous.message.timestamp).total_seconds()
                    if gap > burst_gap_seconds:
                        break
                    burst_end += 1
                if burst_end >= len(messages):
                    break

                last_target = messages[burst_end - 1]
                next_visible = messages[burst_end]
                follow_positive = False
                initiate_positive = False
                if next_visible.sender_person_id == evidence.person_id:
                    gap = max(
                        0.0,
                        (next_visible.message.timestamp - last_target.message.timestamp).total_seconds(),
                    )
                    follow_positive = gap <= session_gap_seconds
                    initiate_positive = gap > session_gap_seconds

                context_start = max(0, burst_start - 30)
                visible_context = tuple(messages[context_start:burst_end])
                common = dict(
                    conversation_id=conversation.conversation_id,
                    observed_at=last_target.message.timestamp,
                    visible_context=visible_context,
                    evidence_weight=last_target.evidence_weight,
                )
                follow_rows.append(_observation(positive=follow_positive, **common))
                initiate_rows.append(_observation(positive=initiate_positive, **common))
                index = burst_end

        return cls(
            follow_up=_ShrunkBinaryPolicy(
                follow_rows,
                focus_conversation_id=focus_conversation_id,
                seed=seed,
                empty_probability=0.0,
            ),
            initiate=_ShrunkBinaryPolicy(
                initiate_rows,
                focus_conversation_id=focus_conversation_id,
                seed=None if seed is None else seed + 1,
                empty_probability=0.0,
            ),
        )

    def choose(
        self,
        action: Action,
        *,
        observed_at: datetime,
        visible_context: Sequence[EvidenceMessage],
    ) -> Action:
        if action == Action.FOLLOW_UP:
            positive = self.follow_up.sample_positive(
                observed_at=observed_at,
                visible_context=visible_context,
            )
        elif action == Action.INITIATE:
            positive = self.initiate.sample_positive(
                observed_at=observed_at,
                visible_context=visible_context,
            )
        else:
            return Action.WAIT
        return action if positive else Action.WAIT


class LatentReadTimingModel:
    """Heuristic READ timing because exports contain no ground-truth read receipt."""

    def __init__(self, *, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def sample_delay_seconds(self, reply_delay_seconds: float) -> float:
        reply_delay = max(0.0, float(reply_delay_seconds))
        if reply_delay <= 0:
            return 0.0
        fraction = self._rng.betavariate(1.4, 2.4)
        return min(reply_delay, max(0.0, reply_delay * fraction))
