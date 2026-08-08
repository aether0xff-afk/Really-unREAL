from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Sequence

from backend.fusion import EvidenceMessage
from backend.replay import ReplayCase, chronological_split
from backend.replay_hazard import DiscreteHazardModel, select_temporal_model
from backend.simulation.action_policy import Action


@dataclass(frozen=True, slots=True)
class LiveTimingFeatures:
    hour_band: int
    weekend: int
    recent_activity: str
    previous_gap: str
    last_message_kind: str


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


def _message_kind(context: Sequence[EvidenceMessage]) -> str:
    if not context:
        return "none"
    text = context[-1].message.text.strip()
    if "?" in text:
        return "question"
    compact = "".join(text.split())
    if len(compact) <= 4:
        return "very_short"
    return "statement"


def visible_timing_features(
    observed_at: datetime,
    context: Sequence[EvidenceMessage],
) -> LiveTimingFeatures:
    cutoff = observed_at - timedelta(minutes=15)
    recent = sum(cutoff <= item.message.timestamp <= observed_at for item in context)
    previous_gap: float | None = None
    if len(context) >= 2:
        previous_gap = max(
            0.0,
            (context[-1].message.timestamp - context[-2].message.timestamp).total_seconds(),
        )
    return LiveTimingFeatures(
        hour_band=observed_at.hour // 4,
        weekend=int(observed_at.weekday() >= 5),
        recent_activity=_activity_bucket(recent),
        previous_gap=_gap_bucket(previous_gap),
        last_message_kind=_message_kind(context),
    )


def _historical_features(case: ReplayCase) -> LiveTimingFeatures:
    return visible_timing_features(case.observation_end, case.context)


def _observable_action(case: ReplayCase) -> Action:
    if not case.context:
        return case.action
    return (
        Action.INITIATE
        if case.context[-1].sender_person_id == case.person_id
        else Action.REPLY
    )


class ContextualLiveTimingSampler:
    """Deploy live timing from observable current context with safe backoff.

    Fine-grained empirical cells can condition on the visible message itself
    (question / very short / statement) when at least a few matching historical
    events exist. Otherwise the policy uses the validation-gated discrete hazard
    model, then progressively broader relationship/action empirical backoff.
    """

    def __init__(
        self,
        cases: Iterable[ReplayCase],
        *,
        person_id: str,
        seed: int | None = None,
        minimum_context_events: int = 3,
    ) -> None:
        self.cases = tuple(cases)
        if not self.cases:
            raise ValueError("cannot build contextual live timing without replay cases")
        self.person_id = person_id
        self.minimum_context_events = max(1, int(minimum_context_events))
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
            if not case.action_is_ambiguous
            and _observable_action(case) == action
        ]

    def _exact_context_candidates(
        self,
        *,
        conversation_id: str,
        action: Action,
        observed_at: datetime,
        visible_context: Sequence[EvidenceMessage],
    ) -> list[ReplayCase]:
        live = visible_timing_features(observed_at, visible_context)
        pool = [
            case
            for case in self._confident_action_cases(action)
            if case.conversation_id == conversation_id
            and _historical_features(case) == live
        ]
        return pool if len(pool) >= self.minimum_context_events else []

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
                if case.conversation_id == conversation_id
                and feat(case).hour_band == live.hour_band
                and feat(case).recent_activity == live.recent_activity
                and feat(case).previous_gap == live.previous_gap
                and feat(case).last_message_kind == live.last_message_kind
            ],
            [
                case for case in confident
                if case.conversation_id == conversation_id
                and feat(case).recent_activity == live.recent_activity
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
                and feat(case).recent_activity == live.recent_activity
                and feat(case).last_message_kind == live.last_message_kind
            ],
            [case for case in confident if case.platform == platform],
            confident,
        ]
        for index, pool in enumerate(levels):
            if pool and (index >= 3 or len(pool) >= self.minimum_context_events):
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
        if not context:
            return None
        last_sender = context[-1].sender_person_id
        if action == Action.REPLY and last_sender == self.person_id:
            return None
        if action == Action.INITIATE and last_sender != self.person_id:
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
            session_restart=False,
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
        if action == Action.WAIT:
            return None
        observed_at = observed_at or datetime.now()

        # If the same relationship has enough exact observable-context evidence,
        # use it directly. This is where question-vs-statement timing can matter.
        exact = self._exact_context_candidates(
            conversation_id=conversation_id,
            action=action,
            observed_at=observed_at,
            visible_context=visible_context,
        )
        if exact:
            return self._sample_case(exact)

        if self._hazard is not None:
            live_case = self._live_case(
                platform=platform,
                conversation_id=conversation_id,
                action=action,
                observed_at=observed_at,
                visible_context=visible_context,
            )
            if live_case is not None:
                return max(
                    0.0,
                    self._hazard.sample_delay_seconds(
                        live_case,
                        seed=self._rng.randrange(0, 2**32),
                    ),
                )

        candidates = self._contextual_candidates(
            platform=platform,
            conversation_id=conversation_id,
            action=action,
            observed_at=observed_at,
            visible_context=visible_context,
        )
        return self._sample_case(candidates) if candidates else None
