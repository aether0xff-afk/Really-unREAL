from __future__ import annotations

import random
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Sequence

from backend.fusion import EvidenceMessage
from backend.replay import ReplayCase, chronological_split
from backend.replay_hazard import DiscreteHazardModel, select_temporal_model
from backend.simulation.action_policy import Action, MESSAGE_ACTIONS


_INTERROGATIVE_RE = re.compile(
    r"(^|\s)(뭐|왜|언제|어디|누구|누가|몇|어떻게|어케|얼마|어느)(\s|$)|"
    r"(뭐함|뭐해|뭐하|몇시|어디감|어디가|언제감|언제와|가능함|가능해|맞음|맞아)$"
)


@dataclass(frozen=True, slots=True)
class LiveTimingFeatures:
    hour_band: int
    weekend: int
    recent_activity: str
    previous_gap: str
    since_last: str
    last_message_kind: str


def classify_message_kind(text: str) -> str:
    text = text.strip()
    compact = "".join(text.split())
    if "?" in text or _INTERROGATIVE_RE.search(text):
        return "question"
    if len(compact) <= 4:
        return "very_short"
    return "statement"


def _activity_bucket(count: int) -> str:
    if count <= 0:
        return "0"
    if count == 1:
        return "1"
    if count <= 4:
        return "2-4"
    return "5+"


def _gap_bucket(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    if seconds <= 60:
        return "<=1m"
    if seconds <= 300:
        return "<=5m"
    if seconds <= 1800:
        return "<=30m"
    if seconds <= 7200:
        return "<=2h"
    return ">2h"


def _since_last_bucket(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    if seconds <= 60:
        return "<=1m"
    if seconds <= 300:
        return "<=5m"
    if seconds <= 1800:
        return "<=30m"
    if seconds <= 7200:
        return "<=2h"
    if seconds <= 21600:
        return "<=6h"
    if seconds <= 86400:
        return "<=1d"
    if seconds <= 604800:
        return "<=7d"
    return ">7d"


def visible_timing_features(
    observed_at: datetime,
    context: Sequence[EvidenceMessage],
) -> LiveTimingFeatures:
    visible = sorted(
        (item for item in context if item.message.timestamp <= observed_at),
        key=lambda item: item.message.timestamp,
    )
    cutoff = observed_at - timedelta(minutes=15)
    recent = sum(cutoff <= item.message.timestamp <= observed_at for item in visible)
    previous_gap: float | None = None
    since_last: float | None = None
    if visible:
        since_last = max(0.0, (observed_at - visible[-1].message.timestamp).total_seconds())
    if len(visible) >= 2:
        previous_gap = max(
            0.0,
            (visible[-1].message.timestamp - visible[-2].message.timestamp).total_seconds(),
        )
    return LiveTimingFeatures(
        hour_band=observed_at.hour // 4,
        weekend=int(observed_at.weekday() >= 5),
        recent_activity=_activity_bucket(recent),
        previous_gap=_gap_bucket(previous_gap),
        since_last=_since_last_bucket(since_last),
        last_message_kind=(classify_message_kind(visible[-1].message.text) if visible else "none"),
    )


def _historical_features(case: ReplayCase) -> LiveTimingFeatures:
    return visible_timing_features(case.observation_end, case.context)


class BurstGapSampler:
    """Sample within-burst bubble spacing from REAL target bursts."""

    def __init__(self, cases: Iterable[ReplayCase], *, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._by_conversation: dict[str, list[float]] = {}
        self._all: list[float] = []
        for case in cases:
            previous = None
            for item in case.target_burst:
                if previous is not None:
                    gap = max(0.0, (item.message.timestamp - previous).total_seconds())
                    if gap <= 120.0:
                        self._by_conversation.setdefault(case.conversation_id, []).append(gap)
                        self._all.append(gap)
                previous = item.message.timestamp

    def sample_gaps(self, *, conversation_id: str, count: int) -> tuple[float, ...]:
        if count <= 1:
            return ()
        pool = self._by_conversation.get(conversation_id) or self._all
        if not pool:
            return tuple(0.0 for _ in range(count - 1))
        return tuple(float(self._rng.choice(pool)) for _ in range(count - 1))


class ContextualLiveTimingSampler:
    """Context-aware live timing with validation-gated hazard deployment.

    A validated hazard model gets first priority. Empirical context cells are used
    only when the richer model did not earn deployment, and fine cells require a
    minimum support floor instead of letting n=3 override validation evidence.
    """

    def __init__(
        self,
        cases: Iterable[ReplayCase],
        *,
        person_id: str,
        seed: int | None = None,
        minimum_context_events: int = 5,
    ) -> None:
        self.cases = tuple(cases)
        if not self.cases:
            raise ValueError("cannot build contextual live timing without replay cases")
        self.person_id = person_id
        self.minimum_context_events = max(3, int(minimum_context_events))
        self._rng = random.Random(seed)
        self._hazard: DiscreteHazardModel | None = None
        self.selection_reason = "contextual empirical backoff"

        split = chronological_split(self.cases)
        if len(split.train) >= 30 and len(split.validation) >= 5:
            try:
                selection, _, _, _, _ = select_temporal_model(
                    split.train,
                    split.validation,
                    minimum_train_events=30,
                    minimum_validation_events=5,
                    improvement_margin=0.01,
                )
                if selection.selected_model == "hazard":
                    self._hazard = DiscreteHazardModel.fit(self.cases)
                    self._hazard.decision_threshold = selection.hazard_decision_threshold
                    self.selection_reason = selection.reason
            except ValueError:
                self._hazard = None

    @property
    def model_name(self) -> str:
        return "hazard" if self._hazard is not None else "contextual_empirical"

    def has_action_evidence(self, action: Action) -> bool:
        return bool(self._confident_action_cases(action))

    def _sample_case(self, cases: Sequence[ReplayCase]) -> float:
        weights = [max(0.0, float(case.evidence_weight)) for case in cases]
        if not any(weights):
            weights = [1.0] * len(cases)
        chosen = self._rng.choices(list(cases), weights=weights, k=1)[0]
        lower = max(0.0, float(chosen.delay_lower_seconds))
        upper = max(lower, float(chosen.delay_upper_seconds))
        return lower if lower == upper else self._rng.uniform(lower, upper)

    def _confident_action_cases(self, action: Action) -> list[ReplayCase]:
        return [
            case
            for case in self.cases
            if not case.action_is_ambiguous and case.action == action
        ]

    def _action_is_valid(
        self,
        action: Action,
        visible_context: Sequence[EvidenceMessage],
    ) -> bool:
        if action not in MESSAGE_ACTIONS:
            return False
        if not visible_context:
            return action == Action.INITIATE
        last_sender = visible_context[-1].sender_person_id
        if action == Action.REPLY:
            return last_sender != self.person_id
        if action in {Action.FOLLOW_UP, Action.INITIATE}:
            return last_sender == self.person_id
        return False

    def _contextual_candidates(
        self,
        *,
        platform: str,
        conversation_id: str,
        action: Action,
        observed_at: datetime,
        visible_context: Sequence[EvidenceMessage],
    ) -> list[ReplayCase]:
        live = visible_timing_features(observed_at, visible_context)
        confident = self._confident_action_cases(action)
        if not confident:
            return []

        def feat(case: ReplayCase) -> LiveTimingFeatures:
            return _historical_features(case)

        levels = [
            [
                case for case in confident
                if case.conversation_id == conversation_id and feat(case) == live
            ],
            [
                case for case in confident
                if case.conversation_id == conversation_id
                and feat(case).hour_band == live.hour_band
                and feat(case).recent_activity == live.recent_activity
                and feat(case).since_last == live.since_last
                and feat(case).last_message_kind == live.last_message_kind
            ],
            [
                case for case in confident
                if case.conversation_id == conversation_id
                and feat(case).last_message_kind == live.last_message_kind
            ],
            [case for case in confident if case.conversation_id == conversation_id],
            [
                case for case in confident
                if case.platform == platform
                and feat(case).hour_band == live.hour_band
                and feat(case).last_message_kind == live.last_message_kind
            ],
            [case for case in confident if case.platform == platform],
            confident,
        ]
        for index, pool in enumerate(levels):
            if not pool:
                continue
            if index >= 3 or len(pool) >= self.minimum_context_events:
                return pool
        return []

    def _live_case(
        self,
        *,
        platform: str,
        conversation_id: str,
        action: Action,
        observed_at: datetime,
        visible_context: Sequence[EvidenceMessage],
    ) -> ReplayCase | None:
        context = tuple(visible_context)
        if not context or not self._action_is_valid(action, context):
            return None
        return ReplayCase(
            case_id=f"live-timing:{conversation_id}:{observed_at.isoformat()}:{action.value}",
            person_id=self.person_id,
            platform=platform,
            conversation_id=conversation_id,
            evidence_context=context[-1].context,
            evidence_weight=1.0,
            action=action,
            observation_end=observed_at,
            action_at=observed_at,
            observed_delay_seconds=0.0,
            delay_lower_seconds=0.0,
            delay_upper_seconds=0.0,
            context=context,
            target_burst=(),
            burst_size=0,
            session_restart=(action == Action.INITIATE),
            action_is_ambiguous=False,
        )

    def sample_delay_seconds(
        self,
        *,
        platform: str,
        conversation_id: str,
        action: Action,
        observed_at: datetime | None = None,
        visible_context: Sequence[EvidenceMessage] = (),
    ) -> float | None:
        if action in {Action.WAIT, Action.READ}:
            return None
        if not self._action_is_valid(action, visible_context):
            return None
        if not self._confident_action_cases(action):
            return None

        observed_at = observed_at or datetime.now()

        # Validation-gated model first; tiny empirical cells can no longer shadow it.
        if self._hazard is not None:
            live_case = self._live_case(
                platform=platform,
                conversation_id=conversation_id,
                action=action,
                observed_at=observed_at,
                visible_context=visible_context,
            )
            if live_case is not None:
                sampled = self._hazard.sample_delay_seconds(
                    live_case,
                    seed=self._rng.randrange(0, 2**32),
                )
                return None if sampled is None else max(0.0, float(sampled))

        candidates = self._contextual_candidates(
            platform=platform,
            conversation_id=conversation_id,
            action=action,
            observed_at=observed_at,
            visible_context=visible_context,
        )
        return self._sample_case(candidates) if candidates else None
