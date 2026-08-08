from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

from backend.replay import ActionSnapshot, ReplayCase
from backend.simulation.action_policy import Action, MESSAGE_ACTIONS


_SESSION_GAP_SECONDS = 6.0 * 3600.0


@dataclass(frozen=True, slots=True)
class ReplayBaselineMetrics:
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


def _weighted_quantile(
    values_and_weights: Sequence[tuple[float, float]],
    q: float,
) -> float:
    if not values_and_weights:
        raise ValueError("weighted quantile requires at least one sample")
    if not 0 <= q <= 1:
        raise ValueError("q must be between 0 and 1")

    ordered = sorted(
        (float(value), max(0.0, float(weight))) for value, weight in values_and_weights
    )
    total_weight = sum(weight for _, weight in ordered)
    if total_weight <= 0:
        ordered = [(value, 1.0) for value, _ in ordered]
        total_weight = float(len(ordered))

    target = q * total_weight
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= target:
            return value
    return ordered[-1][0]


def _observable_candidate_action(
    case: ReplayCase,
    *,
    elapsed_seconds: float = 0.0,
) -> Action:
    """Infer an action role using only visible sender order and elapsed silence."""

    if not case.context:
        raise ValueError("ReplayCase has no visible context")
    previous_sender = case.context[-1].sender_person_id
    if previous_sender == case.person_id:
        return (
            Action.INITIATE
            if elapsed_seconds > _SESSION_GAP_SECONDS
            else Action.FOLLOW_UP
        )
    return Action.REPLY


def _representative_training_delay_seconds(case: ReplayCase) -> float:
    return (case.delay_lower_seconds + case.delay_upper_seconds) / 2.0


def _interval_error_seconds(prediction: float, case: ReplayCase) -> float:
    if prediction < case.delay_lower_seconds:
        return case.delay_lower_seconds - prediction
    if prediction > case.delay_upper_seconds:
        return prediction - case.delay_upper_seconds
    return 0.0


class EmpiricalTimingBaseline:
    """Weighted empirical timing floor with relationship/action backoff."""

    def __init__(
        self,
        *,
        quantile: float,
        minimum_bucket_events: int,
        minimum_conversation_events: int,
        conversation_thresholds: dict[tuple[str, Action], float],
        platform_thresholds: dict[tuple[str, Action], float],
        action_thresholds: dict[Action, float],
        global_threshold: float,
    ) -> None:
        self.quantile = quantile
        self.minimum_bucket_events = minimum_bucket_events
        self.minimum_conversation_events = minimum_conversation_events
        self.conversation_thresholds = dict(conversation_thresholds)
        self.platform_thresholds = dict(platform_thresholds)
        self.action_thresholds = dict(action_thresholds)
        self.global_threshold = float(global_threshold)

    @classmethod
    def fit(
        cls,
        cases: Iterable[ReplayCase],
        *,
        quantile: float = 0.5,
        minimum_bucket_events: int = 5,
        minimum_conversation_events: int = 8,
    ) -> "EmpiricalTimingBaseline":
        cases = list(cases)
        if not cases:
            raise ValueError("cannot fit timing baseline without replay cases")
        if minimum_bucket_events < 1:
            raise ValueError("minimum_bucket_events must be >= 1")
        if minimum_conversation_events < 1:
            raise ValueError("minimum_conversation_events must be >= 1")

        global_samples = [
            (_representative_training_delay_seconds(case), case.evidence_weight)
            for case in cases
        ]
        global_threshold = _weighted_quantile(global_samples, quantile)
        confident_cases = [case for case in cases if not case.action_is_ambiguous]

        action_thresholds: dict[Action, float] = {}
        for action in MESSAGE_ACTIONS:
            bucket = [
                (_representative_training_delay_seconds(case), case.evidence_weight)
                for case in confident_cases
                if case.action == action
            ]
            if bucket:
                action_thresholds[action] = _weighted_quantile(bucket, quantile)

        platform_buckets: dict[tuple[str, Action], list[tuple[float, float]]] = {}
        conversation_buckets: dict[tuple[str, Action], list[tuple[float, float]]] = {}
        for case in confident_cases:
            action = case.action
            if action not in MESSAGE_ACTIONS:
                continue
            sample = (_representative_training_delay_seconds(case), case.evidence_weight)
            platform_buckets.setdefault((case.platform, action), []).append(sample)
            conversation_buckets.setdefault((case.conversation_id, action), []).append(sample)

        platform_thresholds = {
            key: _weighted_quantile(bucket, quantile)
            for key, bucket in platform_buckets.items()
            if len(bucket) >= minimum_bucket_events
        }
        conversation_thresholds = {
            key: _weighted_quantile(bucket, quantile)
            for key, bucket in conversation_buckets.items()
            if len(bucket) >= minimum_conversation_events
        }

        return cls(
            quantile=quantile,
            minimum_bucket_events=minimum_bucket_events,
            minimum_conversation_events=minimum_conversation_events,
            conversation_thresholds=conversation_thresholds,
            platform_thresholds=platform_thresholds,
            action_thresholds=action_thresholds,
            global_threshold=global_threshold,
        )

    def delay_for_action(
        self,
        *,
        conversation_id: str,
        platform: str,
        action: Action,
    ) -> float | None:
        if action not in MESSAGE_ACTIONS:
            return None
        return self.conversation_thresholds.get(
            (conversation_id, action),
            self.platform_thresholds.get(
                (platform, action),
                self.action_thresholds.get(action),
            ),
        )

    def predict_delay_seconds(self, case: ReplayCase) -> float:
        action = _observable_candidate_action(case)
        prediction = self.delay_for_action(
            conversation_id=case.conversation_id,
            platform=case.platform,
            action=action,
        )
        return self.global_threshold if prediction is None else prediction

    def predict_action(self, case: ReplayCase, *, elapsed_seconds: float) -> Action:
        candidate = _observable_candidate_action(case, elapsed_seconds=elapsed_seconds)
        threshold = self.delay_for_action(
            conversation_id=case.conversation_id,
            platform=case.platform,
            action=candidate,
        )
        if threshold is None:
            return Action.WAIT
        return candidate if elapsed_seconds >= threshold else Action.WAIT

    def thresholds_dict(self) -> dict[str, object]:
        return {
            "quantile": self.quantile,
            "minimum_bucket_events": self.minimum_bucket_events,
            "minimum_conversation_events": self.minimum_conversation_events,
            "global_seconds": self.global_threshold,
            "by_action_seconds": {
                action.value: value for action, value in self.action_thresholds.items()
            },
            "by_platform_action_seconds": {
                f"{platform}:{action.value}": value
                for (platform, action), value in self.platform_thresholds.items()
            },
            "by_conversation_action_seconds": {
                f"{conversation_id}:{action.value}": value
                for (conversation_id, action), value in self.conversation_thresholds.items()
            },
        }


def evaluate_empirical_baseline(
    baseline: EmpiricalTimingBaseline,
    cases: Sequence[ReplayCase],
    snapshots: Sequence[ActionSnapshot],
) -> ReplayBaselineMetrics:
    by_id = {case.case_id: case for case in cases}
    confusion: dict[str, Counter[str]] = {
        action.value: Counter() for action in Action
    }
    correct = 0

    for snapshot in snapshots:
        case = by_id.get(snapshot.case_id)
        if case is None:
            raise ValueError(f"snapshot references unknown replay case {snapshot.case_id!r}")
        predicted = baseline.predict_action(case, elapsed_seconds=snapshot.elapsed_seconds)
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
        _interval_error_seconds(baseline.predict_delay_seconds(case), case)
        for case in cases
    ]
    ordered_errors = sorted(interval_errors)
    if ordered_errors:
        middle = len(ordered_errors) // 2
        median_error = (
            ordered_errors[middle]
            if len(ordered_errors) % 2
            else (ordered_errors[middle - 1] + ordered_errors[middle]) / 2
        )
        mean_error = sum(ordered_errors) / len(ordered_errors)
    else:
        median_error = None
        mean_error = None

    return ReplayBaselineMetrics(
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
        mean_interval_error_seconds=(round(mean_error, 3) if mean_error is not None else None),
        median_interval_error_seconds=(
            round(median_error, 3) if median_error is not None else None
        ),
        confusion={actual: dict(predictions) for actual, predictions in confusion.items()},
    )
