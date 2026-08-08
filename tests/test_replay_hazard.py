from dataclasses import replace
from datetime import datetime, timedelta

from backend.fusion import EvidenceContext, EvidenceMessage
from backend.models import ChatMessage
from backend.replay import ReplayCase, build_action_snapshots
from backend.replay_hazard import (
    DiscreteHazardModel,
    _message_kind,
    evaluate_hazard_model,
    select_temporal_model,
)
from backend.simulation.action_policy import Action


BASE = datetime(2026, 8, 7, 20, 0)


def _evidence(at: datetime, sender_id: str, text: str = "x") -> EvidenceMessage:
    return EvidenceMessage(
        message=ChatMessage(at, sender_id, text),
        platform="kakao",
        conversation_id="c",
        context=EvidenceContext.KAKAO_DIRECT,
        sender_person_id=sender_id,
        evidence_weight=1.0,
    )


def _case(
    case_id: str,
    *,
    delay_seconds: float,
    prior_gap_seconds: float,
    previous_person_id: str = "self",
) -> ReplayCase:
    previous = _evidence(BASE, previous_person_id, "last")
    earlier_sender = "target" if previous_person_id == "self" else "self"
    earlier = _evidence(
        BASE - timedelta(seconds=prior_gap_seconds),
        earlier_sender,
        "earlier",
    )
    target = _evidence(
        BASE + timedelta(seconds=delay_seconds),
        "target",
        "held out",
    )
    if previous_person_id == "self":
        action = Action.REPLY
    elif delay_seconds > 21600:
        action = Action.INITIATE
    else:
        action = Action.FOLLOW_UP
    return ReplayCase(
        case_id=case_id,
        person_id="target",
        platform="kakao",
        conversation_id="c",
        evidence_context=EvidenceContext.KAKAO_DIRECT,
        evidence_weight=1.0,
        action=action,
        observation_end=BASE,
        action_at=BASE + timedelta(seconds=delay_seconds),
        observed_delay_seconds=delay_seconds,
        delay_lower_seconds=max(0.0, delay_seconds - 60.0),
        delay_upper_seconds=delay_seconds + 60.0,
        context=(earlier, previous),
        target_burst=(target,),
        burst_size=1,
        session_restart=delay_seconds > 21600,
    )


def test_hazard_uses_same_korean_question_classifier_as_live_timing() -> None:
    case = _case("question", delay_seconds=300, prior_gap_seconds=30)
    question = replace(
        case,
        context=(case.context[0], _evidence(BASE, "self", "몇시에 감")),
    )
    assert _message_kind(question) == "question"


def test_hazard_uses_visible_conversation_activity_to_separate_timing() -> None:
    active = [
        _case(f"active-{index}", delay_seconds=60, prior_gap_seconds=30)
        for index in range(12)
    ]
    stale = [
        _case(f"stale-{index}", delay_seconds=7200, prior_gap_seconds=10800)
        for index in range(12)
    ]
    model = DiscreteHazardModel.fit(
        active + stale,
        minimum_effective_risk=3,
        prior_strength=1,
    )

    assert model.predict_median_delay_seconds(active[0]) < 1800
    assert model.predict_median_delay_seconds(stale[0]) >= 1800


def test_hazard_prediction_never_reads_held_out_target_text() -> None:
    cases = [
        _case(f"case-{index}", delay_seconds=300, prior_gap_seconds=30)
        for index in range(8)
    ]
    model = DiscreteHazardModel.fit(cases, minimum_effective_risk=2)
    original = cases[0]
    replacement = ReplayCase(
        case_id=original.case_id,
        person_id=original.person_id,
        platform=original.platform,
        conversation_id=original.conversation_id,
        evidence_context=original.evidence_context,
        evidence_weight=original.evidence_weight,
        action=original.action,
        observation_end=original.observation_end,
        action_at=original.action_at,
        observed_delay_seconds=original.observed_delay_seconds,
        delay_lower_seconds=original.delay_lower_seconds,
        delay_upper_seconds=original.delay_upper_seconds,
        context=original.context,
        target_burst=(
            _evidence(original.action_at, "target", "completely different future"),
        ),
        burst_size=1,
        session_restart=original.session_restart,
    )

    assert model.hazard_probability(original, elapsed_seconds=300) == model.hazard_probability(
        replacement,
        elapsed_seconds=300,
    )
    assert model.predict_median_delay_seconds(original) == model.predict_median_delay_seconds(
        replacement
    )


def test_threshold_tuning_and_evaluation_produce_valid_metrics() -> None:
    train = [
        _case(f"train-{index}", delay_seconds=300, prior_gap_seconds=30)
        for index in range(20)
    ]
    validation = [
        _case(f"validation-{index}", delay_seconds=300, prior_gap_seconds=30)
        for index in range(10)
    ]
    model = DiscreteHazardModel.fit(train)
    snapshots = build_action_snapshots(validation)
    threshold = model.tune_decision_threshold(validation, snapshots)
    metrics = evaluate_hazard_model(model, validation, snapshots)

    assert 0.01 <= threshold <= 0.99
    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.balanced_accuracy <= 1.0


def test_model_selection_keeps_empirical_fallback_for_sparse_history() -> None:
    train = [
        _case(f"train-{index}", delay_seconds=300 + index * 60, prior_gap_seconds=30)
        for index in range(12)
    ]
    validation = [
        _case(f"validation-{index}", delay_seconds=300, prior_gap_seconds=30)
        for index in range(4)
    ]

    selection, *_ = select_temporal_model(
        train,
        validation,
        minimum_train_events=50,
        minimum_validation_events=10,
    )

    assert selection.selected_model == "empirical"
    assert "insufficient" in selection.reason
