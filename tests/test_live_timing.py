from dataclasses import replace
from datetime import datetime, timedelta

from backend.fusion import EvidenceContext, EvidenceMessage
from backend.live_timing import (
    BurstGapSampler,
    ContextualLiveTimingSampler,
    TimingSampleKind,
    classify_message_kind,
    visible_timing_features,
)
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
    action: Action = Action.REPLY,
) -> ReplayCase:
    previous_sender = "self" if action == Action.REPLY else "target"
    context = (
        _evidence(observation_end - timedelta(minutes=2), "target"),
        _evidence(observation_end, previous_sender, last_text),
    )
    return ReplayCase(
        case_id=case_id,
        person_id="target",
        platform="kakao",
        conversation_id="c",
        evidence_context=EvidenceContext.KAKAO_DIRECT,
        evidence_weight=1.0,
        action=action,
        observation_end=observation_end,
        action_at=observation_end + timedelta(seconds=delay),
        observed_delay_seconds=delay,
        delay_lower_seconds=delay,
        delay_upper_seconds=delay,
        context=context,
        target_burst=(),
        burst_size=1,
        session_restart=(action == Action.INITIATE),
        action_is_ambiguous=False,
    )


def test_visible_timing_features_capture_hour_activity_gap_since_last_and_kind() -> None:
    now = datetime(2026, 8, 8, 21, 0)
    context = (
        _evidence(now - timedelta(minutes=7), "target"),
        _evidence(now - timedelta(minutes=2), "self"),
        _evidence(now - timedelta(seconds=20), "self", "낼 학교 감?"),
    )
    features = visible_timing_features(now, context)
    assert features.hour_band == 5
    assert features.weekend == 1
    assert features.recent_activity == "2-4"
    assert features.previous_gap == "<=5m"
    assert features.since_last == "<=1m"
    assert features.last_message_kind == "question"


def test_korean_question_without_question_mark_is_detected() -> None:
    assert classify_message_kind("몇시에 감") == "question"
    assert classify_message_kind("뭐해") == "question"
    assert classify_message_kind("나 집가는중") == "statement"


def test_time_since_last_visible_changes_even_when_previous_pair_gap_is_same() -> None:
    now = datetime(2026, 8, 10, 20, 0)
    recent = (
        _evidence(now - timedelta(minutes=2), "target"),
        _evidence(now - timedelta(minutes=1), "self", "뭐해"),
    )
    stale = (
        _evidence(now - timedelta(days=3, minutes=1), "target"),
        _evidence(now - timedelta(days=3), "self", "뭐해"),
    )
    # The local conversational rhythm is identical (one-minute prior gap), while
    # the current silence age differs by three days.
    assert visible_timing_features(now, recent).previous_gap == "<=1m"
    assert visible_timing_features(now, stale).previous_gap == "<=1m"
    assert visible_timing_features(now, recent).since_last == "<=1m"
    assert visible_timing_features(now, stale).since_last == "<=7d"


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


def test_contextual_sampler_can_distinguish_question_without_punctuation() -> None:
    base_day = datetime(2026, 8, 3, 20, 0)
    cases = [
        _case(
            f"question-{i}",
            base_day + timedelta(days=i),
            12.0,
            last_text="몇시에 감",
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
        _evidence(now, "self", "오늘 몇시에 감"),
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


def test_action_validity_gate_rejects_semantically_impossible_role() -> None:
    reply = _case("reply", datetime(2026, 8, 3, 8, 0), 42.0)
    follow = _case(
        "follow",
        datetime(2026, 8, 3, 9, 0),
        50.0,
        action=Action.FOLLOW_UP,
    )
    sampler = ContextualLiveTimingSampler([reply, follow], person_id="target", seed=1)
    now = datetime(2026, 8, 10, 12, 0)

    after_user = (_evidence(now, "self", "야"),)
    after_target = (_evidence(now, "target", "ㅇㅇ"),)
    invalid_follow = sampler.sample_timing(
        platform="kakao",
        conversation_id="c",
        action=Action.FOLLOW_UP,
        observed_at=now,
        visible_context=after_user,
    )
    invalid_reply = sampler.sample_timing(
        platform="kakao",
        conversation_id="c",
        action=Action.REPLY,
        observed_at=now,
        visible_context=after_target,
    )
    assert invalid_follow.kind == TimingSampleKind.INVALID
    assert invalid_reply.kind == TimingSampleKind.INVALID
    assert sampler.sample_delay_seconds(
        platform="kakao",
        conversation_id="c",
        action=Action.FOLLOW_UP,
        observed_at=now,
        visible_context=after_user,
    ) is None
    assert sampler.sample_delay_seconds(
        platform="kakao",
        conversation_id="c",
        action=Action.REPLY,
        observed_at=now,
        visible_context=after_target,
    ) is None


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


def test_burst_gap_sampler_uses_observed_internal_gaps_instead_of_fixed_one_second() -> None:
    base = datetime(2026, 8, 3, 8, 0)
    case = _case("burst", base, 10.0)
    first = _evidence(base + timedelta(seconds=10), "target", "a")
    second = _evidence(base + timedelta(seconds=17), "target", "b")
    case = replace(case, target_burst=(first, second), burst_size=2)
    sampler = BurstGapSampler([case], seed=1)
    assert sampler.sample_gaps(conversation_id="c", count=2) == (7.0,)


class _TinyReplyHazard:
    """Only ten percent total event mass, all in the first bin."""

    def hazard_probability(self, case, *, elapsed_seconds):
        return 0.1 if elapsed_seconds == 0.0 else 0.0


def test_live_hazard_is_conditioned_on_already_selected_reply_event() -> None:
    case = _case("reply", datetime(2026, 8, 3, 8, 0), 42.0)
    sampler = ContextualLiveTimingSampler([case], person_id="target", seed=7)
    sampler._hazard = _TinyReplyHazard()  # force the richer path for this regression
    now = datetime(2026, 8, 10, 12, 0)
    context = (_evidence(now, "self", "질문"),)

    # Pre-v1.2.1 hazard sampling could return None from residual survival mass,
    # silently turning timing into a second REPLY-vs-WAIT policy. Once behavior
    # already selected REPLY, live timing must condition on event occurrence.
    for _ in range(20):
        sampled = sampler.sample_timing(
            platform="kakao",
            conversation_id="c",
            action=Action.REPLY,
            observed_at=now,
            visible_context=context,
        )
        assert sampled.kind == TimingSampleKind.SAMPLED
        assert sampled.delay_seconds is not None
        assert 0.0 <= sampled.delay_seconds <= 60.0


class _ImpossibleEarlyHazard:
    def hazard_probability(self, case, *, elapsed_seconds):
        return 1.0 if elapsed_seconds == 0.0 else 0.0


def test_initiate_timing_never_uses_early_hazard_mass() -> None:
    observation = datetime(2026, 8, 3, 8, 0)
    case = _case(
        "initiate",
        observation,
        8 * 3600.0,
        action=Action.INITIATE,
    )
    sampler = ContextualLiveTimingSampler([case], person_id="target", seed=9)
    sampler._hazard = _ImpossibleEarlyHazard()
    now = datetime(2026, 8, 10, 12, 0)
    context = (_evidence(now, "target", "마지막 상대 메시지"),)

    sampled = sampler.sample_timing(
        platform="kakao",
        conversation_id="c",
        action=Action.INITIATE,
        observed_at=now,
        visible_context=context,
    )
    assert sampled.kind == TimingSampleKind.SAMPLED
    assert sampled.delay_seconds is not None
    assert sampled.delay_seconds > 6 * 3600.0


def test_follow_up_timing_never_crosses_new_session_boundary() -> None:
    observation = datetime(2026, 8, 3, 8, 0)
    case = _case(
        "follow",
        observation,
        2 * 3600.0,
        action=Action.FOLLOW_UP,
    )
    sampler = ContextualLiveTimingSampler([case], person_id="target", seed=10)
    sampler._hazard = _ImpossibleEarlyHazard()
    now = datetime(2026, 8, 10, 12, 0)
    context = (_evidence(now, "target", "마지막 상대 메시지"),)

    sampled = sampler.sample_timing(
        platform="kakao",
        conversation_id="c",
        action=Action.FOLLOW_UP,
        observed_at=now,
        visible_context=context,
    )
    assert sampled.kind == TimingSampleKind.SAMPLED
    assert sampled.delay_seconds is not None
    assert sampled.delay_seconds <= 6 * 3600.0
