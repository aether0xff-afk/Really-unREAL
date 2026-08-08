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
class ResponseObservation:
    conversation_id: str
    observed_at: datetime
    replied: bool
    hour_band: int
    weekend: int
    recent_activity: str
    previous_gap: str
    message_kind: str
    evidence_weight: float


class LiveResponsePolicy:
    """Decide REPLY vs WAIT before reply timing or text generation.

    Negatives come from counterpart bursts that were followed by another
    counterpart burst (or a long session restart) before the target answered.
    The final export edge is treated as censored and is not used as a negative.
    Small cells are shrunk toward the person's global response rate.
    """

    def __init__(
        self,
        observations: Iterable[ResponseObservation],
        *,
        focus_conversation_id: str,
        seed: int | None = None,
        minimum_cell_events: int = 3,
        prior_strength: float = 5.0,
    ) -> None:
        self.observations = tuple(observations)
        self.focus_conversation_id = focus_conversation_id
        self.minimum_cell_events = max(2, int(minimum_cell_events))
        self.prior_strength = max(0.0, float(prior_strength))
        self._rng = random.Random(seed)

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
        observations: list[ResponseObservation] = []
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

                # The final user burst is right-censored: the export may simply end
                # before a response happened, so it must not become a fake WAIT.
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
                features = visible_timing_features(last_user.message.timestamp, visible_context)
                observations.append(
                    ResponseObservation(
                        conversation_id=conversation.conversation_id,
                        observed_at=last_user.message.timestamp,
                        replied=replied,
                        hour_band=features.hour_band,
                        weekend=features.weekend,
                        recent_activity=features.recent_activity,
                        previous_gap=features.previous_gap,
                        message_kind=classify_message_kind(last_user.message.text),
                        evidence_weight=max(0.0, float(last_user.evidence_weight)),
                    )
                )
                index = burst_end

        return cls(
            observations,
            focus_conversation_id=focus_conversation_id,
            seed=seed,
        )

    def _rate(self, rows: Sequence[ResponseObservation]) -> tuple[float, float]:
        total = sum(max(0.0, row.evidence_weight) for row in rows)
        success = sum(
            max(0.0, row.evidence_weight) for row in rows if row.replied
        )
        return success, total

    @property
    def global_reply_probability(self) -> float:
        success, total = self._rate(self.observations)
        if total <= 0:
            return 1.0
        return (success + 1.0) / (total + 2.0)

    def reply_probability(
        self,
        *,
        observed_at: datetime,
        visible_context: Sequence[EvidenceMessage],
    ) -> float:
        if not self.observations:
            return 1.0

        live = visible_timing_features(observed_at, visible_context)
        global_p = self.global_reply_probability

        def enough(rows: list[ResponseObservation]) -> bool:
            return len(rows) >= self.minimum_cell_events

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
            if not rows or (index < len(levels) - 1 and not enough(rows)):
                continue
            success, total = self._rate(rows)
            if total <= 0:
                continue
            probability = (
                success + self.prior_strength * global_p
            ) / (total + self.prior_strength)
            return min(1.0, max(0.0, probability))
        return global_p

    def choose_after_counterpart_message(
        self,
        *,
        observed_at: datetime,
        visible_context: Sequence[EvidenceMessage],
    ) -> Action:
        probability = self.reply_probability(
            observed_at=observed_at,
            visible_context=visible_context,
        )
        return Action.REPLY if self._rng.random() < probability else Action.WAIT


class LatentReadTimingModel:
    """Explicitly heuristic READ timing because exports contain no read receipts.

    It only separates READ from REPLY when a reply behavior already exists. The
    result is labeled SIMULATION inference and must never be presented as a real
    Kakao receipt or a learned ground-truth read timestamp.
    """

    def __init__(self, *, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def sample_delay_seconds(self, reply_delay_seconds: float) -> float:
        reply_delay = max(0.0, float(reply_delay_seconds))
        if reply_delay <= 0:
            return 0.0
        fraction = self._rng.betavariate(1.4, 2.4)
        return min(reply_delay, max(0.0, reply_delay * fraction))
