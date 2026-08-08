from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import timedelta
from typing import Iterable, Sequence

from backend.replay import ActionSnapshot, ReplayCase, build_action_snapshots
from backend.replay_baseline import (
    EmpiricalTimingBaseline,
    ReplayBaselineMetrics,
    evaluate_empirical_baseline,
)
from backend.simulation.action_policy import Action


ELAPSED_BINS_SECONDS: tuple[float, ...] = (
    0.0,
    60.0,
    300.0,
    1800.0,
    7200.0,
    21600.0,
    86400.0,
    259200.0,
    604800.0,
    2592000.0,
    7776000.0,
    31536000.0,
)
_SESSION_GAP_SECONDS = 6.0 * 3600.0
_INTERROGATIVE_RE = re.compile(
    r"(^|\s)(뭐|왜|언제|어디|누구|누가|몇|어떻게|어케|얼마|어느)(\s|$)"
)
_QUESTION_STEMS = (
    "뭐함",
    "뭐해",
    "뭐하",
    "몇시",
    "몇명",
    "몇개",
    "어디감",
    "어디가",
    "언제감",
    "언제와",
    "어떻게",
    "어케",
    "얼마",
    "가능함",
    "가능해",
    "맞음",
    "맞아",
)


@dataclass(frozen=True, slots=True)
class HazardMetrics:
    snapshot_count: int
    accuracy: float
    balanced_accuracy: float
    wait_recall: float | None
    reply_recall: float | None
    follow_up_recall: float | None
    initiate_recall: float | None
    mean_interval_error_seconds: float | None
    median_interval_error_seconds: float | None
    confusion: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TemporalModelSelection:
    selected_model: str
    reason: str
    baseline_validation_balanced_accuracy: float
    hazard_validation_balanced_accuracy: float
    hazard_decision_threshold: float
    train_events: int
    validation_events: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _candidate_action(case: ReplayCase, elapsed_seconds: float = 0.0) -> Action:
    if not case.context:
        raise ValueError("ReplayCase has no visible context")
    previous_sender = case.context[-1].sender_person_id
    if previous_sender != case.person_id:
        return Action.REPLY
    return Action.INITIATE if elapsed_seconds > _SESSION_GAP_SECONDS else Action.FOLLOW_UP


def _elapsed_bin(seconds: float) -> int:
    seconds = max(0.0, float(seconds))
    for index, upper in enumerate(ELAPSED_BINS_SECONDS[1:]):
        if seconds < upper:
            return index
    return len(ELAPSED_BINS_SECONDS) - 2


def _activity_bucket(case: ReplayCase, elapsed_seconds: float) -> str:
    observed_at = case.observation_end + timedelta(seconds=elapsed_seconds)
    cutoff = observed_at - timedelta(minutes=15)
    count = sum(
        cutoff <= message.message.timestamp <= observed_at for message in case.context
    )
    if count <= 0:
        return "0"
    if count == 1:
        return "1"
    if count <= 4:
        return "2-4"
    return "5+"


def _previous_gap_bucket(case: ReplayCase) -> str:
    if len(case.context) < 2:
        return "unknown"
    gap = max(
        0.0,
        (
            case.context[-1].message.timestamp
            - case.context[-2].message.timestamp
        ).total_seconds(),
    )
    if gap <= 60:
        return "<=1m"
    if gap <= 300:
        return "<=5m"
    if gap <= 1800:
        return "<=30m"
    if gap <= 7200:
        return "<=2h"
    return ">2h"


def _since_last_bucket(case: ReplayCase, elapsed_seconds: float) -> str:
    if not case.context:
        return "unknown"
    base_gap = max(
        0.0,
        (case.observation_end - case.context[-1].message.timestamp).total_seconds(),
    )
    elapsed = base_gap + max(0.0, float(elapsed_seconds))
    if elapsed <= 60:
        return "<=1m"
    if elapsed <= 300:
        return "<=5m"
    if elapsed <= 1800:
        return "<=30m"
    if elapsed <= 7200:
        return "<=2h"
    if elapsed <= 21600:
        return "<=6h"
    if elapsed <= 86400:
        return "<=1d"
    if elapsed <= 604800:
        return "<=7d"
    return ">7d"


def _message_kind(case: ReplayCase) -> str:
    if not case.context:
        return "none"
    text = case.context[-1].message.text.strip()
    compact = "".join(text.split())
    if (
        "?" in text
        or _INTERROGATIVE_RE.search(text)
        or any(stem in compact for stem in _QUESTION_STEMS)
    ):
        return "question"
    if len(compact) <= 4:
        return "very_short"
    return "statement"


def _feature_tuple(case: ReplayCase, elapsed_seconds: float) -> tuple[object, ...]:
    observed_at = case.observation_end + timedelta(seconds=elapsed_seconds)
    # Action identity must itself be observable at this elapsed time. In
    # particular, a future long-gap INITIATE cannot leak its held-out label into
    # earlier survival bins where it is still only a possible FOLLOW_UP.
    action = _candidate_action(case, elapsed_seconds)
    return (
        action.value,
        case.platform,
        observed_at.hour // 4,
        int(observed_at.weekday() >= 5),
        _activity_bucket(case, elapsed_seconds),
        _previous_gap_bucket(case),
        _since_last_bucket(case, elapsed_seconds),
        _message_kind(case),
    )


def _feature_keys(feature: tuple[object, ...]) -> tuple[tuple[object, ...], ...]:
    action, platform, hour_band, weekend, activity, previous_gap, since_last, kind = feature
    return (
        (),
        (action,),
        (action, kind),
        (action, activity, since_last, kind),
        (action, hour_band, activity, since_last, kind),
        (action, hour_band, weekend, activity, previous_gap, since_last, kind),
        (action, platform, hour_band, weekend, activity, previous_gap, since_last, kind),
    )


def _interval_error_seconds(prediction: float, case: ReplayCase) -> float:
    if prediction < case.delay_lower_seconds:
        return case.delay_lower_seconds - prediction
    if prediction > case.delay_upper_seconds:
        return prediction - case.delay_upper_seconds
    return 0.0


class DiscreteHazardModel:
    """Source-weighted discrete survival model over observable context."""

    def __init__(
        self,
        *,
        minimum_effective_risk: float,
        prior_strength: float,
        tables: tuple[dict[tuple[tuple[object, ...], int], tuple[float, float]], ...],
        global_hazards: tuple[float, ...],
        decision_threshold: float = 0.5,
    ) -> None:
        self.minimum_effective_risk = float(minimum_effective_risk)
        self.prior_strength = float(prior_strength)
        self.tables = tables
        self.global_hazards = global_hazards
        self.decision_threshold = float(decision_threshold)

    @classmethod
    def fit(
        cls,
        cases: Iterable[ReplayCase],
        *,
        minimum_effective_risk: float = 5.0,
        prior_strength: float = 4.0,
    ) -> "DiscreteHazardModel":
        cases = list(cases)
        if not cases:
            raise ValueError("cannot fit hazard model without replay cases")
        if minimum_effective_risk <= 0:
            raise ValueError("minimum_effective_risk must be > 0")
        if prior_strength < 0:
            raise ValueError("prior_strength must be >= 0")

        levels = len(_feature_keys(("x", "x", 0, 0, "x", "x", "x", "x")))
        mutable_tables: list[
            defaultdict[tuple[tuple[object, ...], int], list[float]]
        ] = [defaultdict(lambda: [0.0, 0.0]) for _ in range(levels)]

        for case in cases:
            event_bin = _elapsed_bin(case.observed_delay_seconds)
            weight = max(0.0, float(case.evidence_weight))
            if weight == 0:
                continue
            for bin_index in range(event_bin + 1):
                elapsed = ELAPSED_BINS_SECONDS[bin_index]
                keys = _feature_keys(_feature_tuple(case, elapsed))
                for level, key in enumerate(keys):
                    if level > 0 and case.action_is_ambiguous:
                        continue
                    cell = mutable_tables[level][(key, bin_index)]
                    cell[0] += weight
                    if bin_index == event_bin:
                        cell[1] += weight

        frozen_tables = tuple(
            {key: (value[0], value[1]) for key, value in table.items()}
            for table in mutable_tables
        )
        global_hazards: list[float] = []
        for bin_index in range(len(ELAPSED_BINS_SECONDS) - 1):
            risk, events = frozen_tables[0].get(((), bin_index), (0.0, 0.0))
            global_hazards.append(events / risk if risk > 0 else 0.0)

        return cls(
            minimum_effective_risk=minimum_effective_risk,
            prior_strength=prior_strength,
            tables=frozen_tables,
            global_hazards=tuple(global_hazards),
        )

    def hazard_probability(self, case: ReplayCase, *, elapsed_seconds: float) -> float:
        bin_index = _elapsed_bin(elapsed_seconds)
        keys = _feature_keys(_feature_tuple(case, ELAPSED_BINS_SECONDS[bin_index]))
        global_hazard = self.global_hazards[bin_index]

        for level in range(len(keys) - 1, -1, -1):
            risk, events = self.tables[level].get(
                (keys[level], bin_index),
                (0.0, 0.0),
            )
            if level != 0 and risk < self.minimum_effective_risk:
                continue
            denominator = risk + self.prior_strength
            if denominator <= 0:
                return min(1.0, max(0.0, global_hazard))
            probability = (
                events + self.prior_strength * global_hazard
            ) / denominator
            return min(1.0, max(0.0, probability))

        return min(1.0, max(0.0, global_hazard))

    def predict_action(
        self,
        case: ReplayCase,
        *,
        elapsed_seconds: float,
        threshold: float | None = None,
    ) -> Action:
        threshold = self.decision_threshold if threshold is None else float(threshold)
        probability = self.hazard_probability(case, elapsed_seconds=elapsed_seconds)
        return _candidate_action(case, elapsed_seconds) if probability >= threshold else Action.WAIT

    def predict_median_delay_seconds(self, case: ReplayCase) -> float:
        survival = 1.0
        for bin_index in range(len(ELAPSED_BINS_SECONDS) - 1):
            start = ELAPSED_BINS_SECONDS[bin_index]
            end = ELAPSED_BINS_SECONDS[bin_index + 1]
            hazard = self.hazard_probability(case, elapsed_seconds=start)
            previous_survival = survival
            survival *= 1.0 - hazard
            if survival <= 0.5:
                if hazard <= 0 or previous_survival <= 0.5:
                    return start
                log_step = math.log(max(1e-12, 1.0 - hazard))
                if log_step == 0:
                    return end
                fraction = math.log(0.5 / previous_survival) / log_step
                fraction = min(1.0, max(0.0, fraction))
                return start + fraction * (end - start)
        return ELAPSED_BINS_SECONDS[-1]

    def sample_delay_seconds(
        self,
        case: ReplayCase,
        *,
        seed: int | None = None,
    ) -> float | None:
        """Sample an event time; return None when the survival mass says WAIT."""

        rng = random.Random(seed)
        target = rng.random()
        survival = 1.0
        cumulative = 0.0
        for bin_index in range(len(ELAPSED_BINS_SECONDS) - 1):
            start = ELAPSED_BINS_SECONDS[bin_index]
            end = ELAPSED_BINS_SECONDS[bin_index + 1]
            hazard = self.hazard_probability(case, elapsed_seconds=start)
            event_mass = survival * hazard
            if cumulative + event_mass >= target and event_mass > 0:
                return start + rng.random() * (end - start)
            cumulative += event_mass
            survival *= 1.0 - hazard
        return None

    def tune_decision_threshold(
        self,
        cases: Sequence[ReplayCase],
        snapshots: Sequence[ActionSnapshot] | None = None,
    ) -> float:
        snapshots = list(snapshots or build_action_snapshots(cases))
        if not snapshots:
            return self.decision_threshold
        best_threshold = self.decision_threshold
        best_score = -1.0
        for step in range(1, 100):
            threshold = step / 100.0
            metrics = evaluate_hazard_model(
                self,
                cases,
                snapshots,
                threshold=threshold,
            )
            if metrics.balanced_accuracy > best_score:
                best_score = metrics.balanced_accuracy
                best_threshold = threshold
        self.decision_threshold = best_threshold
        return best_threshold

    def summary_dict(self) -> dict[str, object]:
        return {
            "minimum_effective_risk": self.minimum_effective_risk,
            "prior_strength": self.prior_strength,
            "decision_threshold": self.decision_threshold,
            "elapsed_bins_seconds": list(ELAPSED_BINS_SECONDS),
            "global_hazards": list(self.global_hazards),
        }


def evaluate_hazard_model(
    model: DiscreteHazardModel,
    cases: Sequence[ReplayCase],
    snapshots: Sequence[ActionSnapshot] | None = None,
    *,
    threshold: float | None = None,
) -> HazardMetrics:
    snapshots = list(snapshots or build_action_snapshots(cases))
    by_id = {case.case_id: case for case in cases}
    confusion: dict[str, Counter[str]] = {
        action.value: Counter() for action in Action
    }
    correct = 0

    for snapshot in snapshots:
        case = by_id.get(snapshot.case_id)
        if case is None:
            raise ValueError(
                f"snapshot references unknown replay case {snapshot.case_id!r}"
            )
        predicted = model.predict_action(
            case,
            elapsed_seconds=snapshot.elapsed_seconds,
            threshold=threshold,
        )
        actual = snapshot.expected_action
        confusion[actual.value][predicted.value] += 1
        correct += predicted == actual

    recalls: dict[Action, float | None] = {}
    present_recalls: list[float] = []
    for action in Action:
        row = confusion[action.value]
        total = sum(row.values())
        recall = row[action.value] / total if total else None
        recalls[action] = recall
        if recall is not None:
            present_recalls.append(recall)

    interval_errors = [
        _interval_error_seconds(model.predict_median_delay_seconds(case), case)
        for case in cases
    ]
    ordered_errors = sorted(interval_errors)
    if ordered_errors:
        mean_error = sum(ordered_errors) / len(ordered_errors)
        middle = len(ordered_errors) // 2
        median_error = (
            ordered_errors[middle]
            if len(ordered_errors) % 2
            else (ordered_errors[middle - 1] + ordered_errors[middle]) / 2
        )
    else:
        mean_error = None
        median_error = None

    return HazardMetrics(
        snapshot_count=len(snapshots),
        accuracy=round(correct / len(snapshots), 6) if snapshots else 0.0,
        balanced_accuracy=(
            round(sum(present_recalls) / len(present_recalls), 6)
            if present_recalls
            else 0.0
        ),
        wait_recall=recalls[Action.WAIT],
        reply_recall=recalls[Action.REPLY],
        follow_up_recall=recalls[Action.FOLLOW_UP],
        initiate_recall=recalls[Action.INITIATE],
        mean_interval_error_seconds=(
            round(mean_error, 3) if mean_error is not None else None
        ),
        median_interval_error_seconds=(
            round(median_error, 3) if median_error is not None else None
        ),
        confusion={actual: dict(row) for actual, row in confusion.items()},
    )


def select_temporal_model(
    train_cases: Sequence[ReplayCase],
    validation_cases: Sequence[ReplayCase],
    *,
    minimum_train_events: int = 50,
    minimum_validation_events: int = 10,
    improvement_margin: float = 0.01,
) -> tuple[
    TemporalModelSelection,
    EmpiricalTimingBaseline,
    DiscreteHazardModel,
    ReplayBaselineMetrics,
    HazardMetrics,
]:
    if not train_cases:
        raise ValueError("train_cases must not be empty")
    if not validation_cases:
        raise ValueError("validation_cases must not be empty")

    baseline = EmpiricalTimingBaseline.fit(train_cases)
    hazard = DiscreteHazardModel.fit(train_cases)
    validation_snapshots = build_action_snapshots(validation_cases)
    hazard.tune_decision_threshold(validation_cases, validation_snapshots)

    baseline_metrics = evaluate_empirical_baseline(
        baseline,
        validation_cases,
        validation_snapshots,
    )
    hazard_metrics = evaluate_hazard_model(
        hazard,
        validation_cases,
        validation_snapshots,
    )

    confident_train_events = sum(not case.action_is_ambiguous for case in train_cases)
    confident_validation_events = sum(
        not case.action_is_ambiguous for case in validation_cases
    )
    enough_data = (
        confident_train_events >= minimum_train_events
        and confident_validation_events >= minimum_validation_events
    )
    beats_baseline = (
        hazard_metrics.balanced_accuracy
        > baseline_metrics.balanced_accuracy + improvement_margin
    )

    if not enough_data:
        selected = "empirical"
        reason = "insufficient confident replay events for richer hazard model"
    elif beats_baseline:
        selected = "hazard"
        reason = "hazard improved validation balanced accuracy"
    else:
        selected = "empirical"
        reason = "hazard did not beat empirical validation baseline"

    selection = TemporalModelSelection(
        selected_model=selected,
        reason=reason,
        baseline_validation_balanced_accuracy=baseline_metrics.balanced_accuracy,
        hazard_validation_balanced_accuracy=hazard_metrics.balanced_accuracy,
        hazard_decision_threshold=hazard.decision_threshold,
        train_events=len(train_cases),
        validation_events=len(validation_cases),
    )
    return selection, baseline, hazard, baseline_metrics, hazard_metrics
