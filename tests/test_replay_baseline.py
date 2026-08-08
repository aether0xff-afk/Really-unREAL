from dataclasses import replace
from datetime import datetime, timedelta

from backend.fusion import EvidenceContext, EvidenceMessage
from backend.models import ChatMessage
from backend.replay import ActionSnapshot, ReplayCase
from backend.replay_baseline import (
    EmpiricalTimingBaseline,
    evaluate_empirical_baseline,
)
from backend.simulation.action_policy import Action


BASE = datetime(2026, 8, 7, 20, 0)


def _case(
    case_id: str,
    *,
    platform: str,
    previous_person_id: str,
    delay_seconds: float,
    weight: float,
    conversation_id: str = "c",
) -> ReplayCase:
    context_kind = (
        EvidenceContext.KAKAO_DIRECT
        if platform == "kakao"
        else EvidenceContext.INSTAGRAM_DIRECT
    )
    previous = EvidenceMessage(
        message=ChatMessage(BASE, "previous", "x"),
        platform=platform,
        conversation_id=conversation_id,
        context=context_kind,
        sender_person_id=previous_person_id,
        evidence_weight=weight,
    )
    target = EvidenceMessage(
        message=ChatMessage(BASE + timedelta(seconds=delay_seconds), "target", "y"),
        platform=platform,
        conversation_id=conversation_id,
        context=context_kind,
        sender_person_id="target",
        evidence_weight=weight,
    )
    action = Action.REPLY if previous_person_id == "self" else Action.INITIATE
    return ReplayCase(
        case_id=case_id,
        person_id="target",
        platform=platform,
        conversation_id=conversation_id,
        evidence_context=context_kind,
        evidence_weight=weight,
        action=action,
        observation_end=BASE,
        action_at=BASE + timedelta(seconds=delay_seconds),
        observed_delay_seconds=delay_seconds,
        delay_lower_seconds=delay_seconds,
        delay_upper_seconds=delay_seconds,
        context=(previous,),
        target_burst=(target,),
        burst_size=1,
        session_restart=False,
    )


def test_weighted_baseline_keeps_primary_kakao_from_being_outvoted_by_supplement() -> None:
    cases = [
        _case(
            "kakao",
            platform="kakao",
            previous_person_id="self",
            delay_seconds=60,
            weight=1.0,
        ),
        _case(
            "instagram",
            platform="instagram",
            previous_person_id="self",
            delay_seconds=3600,
            weight=0.55,
        ),
    ]

    baseline = EmpiricalTimingBaseline.fit(cases, minimum_bucket_events=5)

    assert baseline.action_thresholds[Action.REPLY] == 60.0


def test_baseline_predicts_wait_until_empirical_threshold() -> None:
    train = [
        _case(
            "a",
            platform="kakao",
            previous_person_id="self",
            delay_seconds=300,
            weight=1.0,
        ),
        _case(
            "b",
            platform="kakao",
            previous_person_id="self",
            delay_seconds=600,
            weight=1.0,
        ),
    ]
    baseline = EmpiricalTimingBaseline.fit(train, minimum_bucket_events=5)
    case = train[0]

    assert baseline.predict_action(case, elapsed_seconds=60) == Action.WAIT
    assert baseline.predict_action(case, elapsed_seconds=300) == Action.REPLY


def test_relationship_threshold_can_override_global_self_twin_timing() -> None:
    fast = [
        _case(
            f"fast-{index}",
            platform="kakao",
            previous_person_id="friend-a",
            delay_seconds=60,
            weight=1.0,
            conversation_id="friend-a",
        )
        for index in range(2)
    ]
    slow = [
        _case(
            f"slow-{index}",
            platform="kakao",
            previous_person_id="friend-b",
            delay_seconds=3600,
            weight=1.0,
            conversation_id="friend-b",
        )
        for index in range(2)
    ]
    # Re-target the synthetic helper cases as SELF_TWIN output. The observable
    # candidate action is still REPLY because the previous sender is not target.
    train = [replace(case, person_id="self", action=Action.REPLY) for case in fast + slow]
    baseline = EmpiricalTimingBaseline.fit(
        train,
        minimum_bucket_events=10,
        minimum_conversation_events=2,
    )

    assert baseline.predict_delay_seconds(train[0]) == 60.0
    assert baseline.predict_delay_seconds(train[-1]) == 3600.0


def test_coarse_same_minute_delay_is_not_fitted_as_literal_zero_seconds() -> None:
    case = _case(
        "coarse",
        platform="kakao",
        previous_person_id="self",
        delay_seconds=0,
        weight=1.0,
    )
    case = replace(
        case,
        delay_lower_seconds=0.0,
        delay_upper_seconds=60.0,
    )

    baseline = EmpiricalTimingBaseline.fit([case])

    assert baseline.action_thresholds[Action.REPLY] == 30.0


def test_ambiguous_long_gap_does_not_distort_action_specific_threshold() -> None:
    normal = _case(
        "normal",
        platform="kakao",
        previous_person_id="self",
        delay_seconds=300,
        weight=1.0,
    )
    long_gap = _case(
        "long-gap",
        platform="kakao",
        previous_person_id="self",
        delay_seconds=86400,
        weight=10.0,
    )
    long_gap = replace(
        long_gap,
        session_restart=True,
        action_is_ambiguous=True,
    )

    baseline = EmpiricalTimingBaseline.fit([normal, long_gap])

    # The long-gap event still informs the global timing floor but cannot claim
    # that one-day silence is a trustworthy REPLY-specific pattern.
    assert baseline.action_thresholds[Action.REPLY] == 300.0
    assert baseline.global_threshold == 86400.0


def test_interval_aware_metric_does_not_penalize_prediction_inside_censored_range() -> None:
    train_case = _case(
        "train",
        platform="kakao",
        previous_person_id="self",
        delay_seconds=120,
        weight=1.0,
    )
    baseline = EmpiricalTimingBaseline.fit([train_case])

    test_case = _case(
        "test",
        platform="kakao",
        previous_person_id="self",
        delay_seconds=180,
        weight=1.0,
    )
    test_case = replace(
        test_case,
        delay_lower_seconds=120.0,
        delay_upper_seconds=240.0,
    )
    snapshots = [
        ActionSnapshot(
            case_id="test",
            observed_at=test_case.action_at,
            expected_action=Action.REPLY,
            elapsed_seconds=180.0,
            remaining_observed_seconds=0.0,
        )
    ]

    metrics = evaluate_empirical_baseline(baseline, [test_case], snapshots)

    assert metrics.median_interval_error_seconds == 0.0
