from datetime import datetime, timedelta

from backend.fusion import EvidenceContext, EvidenceMessage
from backend.live_timing import ContextualLiveTimingSampler, visible_timing_features
from backend.models import ChatMessage
from backend.replay import ReplayCase
from backend.simulation.action_policy import Action


def _evidence(at: datetime, sender: str, text: str = "x") -> EvidenceMessage:
    return EvidenceMessage(
        message=ChatMessage(at, sender, text),
        platform="kakao",
        conversation_id="c",
        context=EvidenceContext.KAKAO_DIRECT,
        sender_person_id=sender,
        evidence_weight=1.0,
    )


def _case(
    case_id: str,
    observation_end: datetime,
    delay: float,
    *,
    last_text: str = "x",
) -> ReplayCase:
    context = (
        _evidence(observation_end - timedelta(minutes=2), "target"),
        _evidence(observation_end, "self", last_text),
    )
    return ReplayCase(
        case_id=case_id,
        person_id="target",
        platform="kakao",
        conversation_id="c",
        evidence_context=EvidenceContext.KAKAO_DIRECT,
        evidence_weight=1.0,
        action=Action.REPLY,
        observation_end=observation_end,
        action_at=observation_end + timedelta(seconds=delay),
        observed_delay_seconds=delay,
        delay_lower_seconds=delay,
        delay_upper_seconds=delay,
        context=context,
        target_burst=(),
        burst_size=1,
        session_restart=False,
        action_is_ambiguous=False,
    )


def test_visible_timing_features_capture_hour_activity_gap_and_message_kind() -> None:
    now = datetime(2026, 8, 8, 21, 0)
    context = (
        _evidence(now - timedelta(minutes=7), "target"),
        _evidence(now - timedelta(minutes=2), "self"),
        _evidence(now, "self", "낼 학교 감?"),
    )
    features = visible_timing_features(now, context)
    assert features.hour_band == 5
    assert features.weekend == 1
    assert features.recent_activity == "2-4"
    assert features.previous_gap == "<=5m"
    assert features.last_message_kind == "question"


def test_contextual_sampler_changes_distribution_by_time_of_day() -> None:
    cases = [
        _case(f"morning-{i}", datetime(2026, 8, 3 + i, 8, 0), 10.0)
        for i in range(3)
    ] + [
        _case(f"evening-{i}", datetime(2026, 8, 3 + i, 20, 0), 300.0)
        for i in range(3)
    ]
    sampler = ContextualLiveTimingSampler(
        cases,
        person_id="target",
        seed=2,
        minimum_context_events=3,
    )

    morning_at = datetime(2026, 8, 10, 8, 0)
    morning_context = (
        _evidence(morning_at - timedelta(minutes=2), "target"),
        _evidence(morning_at, "self"),
    )
    evening_at = datetime(2026, 8, 10, 20, 0)
    evening_context = (
        _evidence(evening_at - timedelta(minutes=2), "target"),
        _evidence(evening_at, "self"),
    )

    assert sampler.sample_delay_seconds(
        platform="kakao",
        conversation_id="c",
        action=Action.REPLY,
        observed_at=morning_at,
        visible_context=morning_context,
    ) == 10.0
    assert sampler.sample_delay_seconds(
        platform="kakao",
        conversation_id="c",
        action=Action.REPLY,
        observed_at=evening_at,
        visible_context=evening_context,
    ) == 300.0


def test_contextual_sampler_can_distinguish_question_from_statement() -> None:
    # Same relationship, time band, activity and previous gap. Only the visible
    # current message type differs, and each cell has enough observations.
    base_day = datetime(2026, 8, 3, 20, 0)
    cases = [
        _case(
            f"question-{i}",
            base_day + timedelta(days=i),
            12.0,
            last_text="몇시에 감?",
        )
        for i in range(3)
    ] + [
        _case(
            f"statement-{i}",
            base_day + timedelta(days=i),
            240.0,
            last_text="나 집가는중",
        )
        for i in range(3)
    ]
    sampler = ContextualLiveTimingSampler(
        cases,
        person_id="target",
        seed=3,
        minimum_context_events=3,
    )
    now = datetime(2026, 8, 10, 20, 0)

    question_context = (
        _evidence(now - timedelta(minutes=2), "target"),
        _evidence(now, "self", "오늘 몇시에 감?"),
    )
    statement_context = (
        _evidence(now - timedelta(minutes=2), "target"),
        _evidence(now, "self", "나 지금 집가는중"),
    )

    assert sampler.sample_delay_seconds(
        platform="kakao",
        conversation_id="c",
        action=Action.REPLY,
        observed_at=now,
        visible_context=question_context,
    ) == 12.0
    assert sampler.sample_delay_seconds(
        platform="kakao",
        conversation_id="c",
        action=Action.REPLY,
        observed_at=now,
        visible_context=statement_context,
    ) == 240.0


def test_contextual_sampler_falls_back_without_inventing_action() -> None:
    cases = [_case("reply", datetime(2026, 8, 3, 8, 0), 42.0)]
    sampler = ContextualLiveTimingSampler(cases, person_id="target", seed=1)
    now = datetime(2026, 8, 10, 12, 0)
    context = (_evidence(now, "self"),)
    assert sampler.sample_delay_seconds(
        platform="kakao",
        conversation_id="c",
        action=Action.REPLY,
        observed_at=now,
        visible_context=context,
    ) == 42.0
    assert sampler.sample_delay_seconds(
        platform="kakao",
        conversation_id="c",
        action=Action.INITIATE,
        observed_at=now,
        visible_context=context,
    ) is None
