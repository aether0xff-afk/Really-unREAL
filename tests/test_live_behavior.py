from datetime import datetime, timedelta

from backend.fusion import EvidenceContext, EvidenceConversation, EvidenceMessage, PersonEvidence
from backend.live_behavior import LiveResponsePolicy, TargetContinuationPolicy
from backend.models import ChatMessage
from backend.simulation.action_policy import Action


BASE = datetime(2026, 8, 8, 12, 0)


def _message(at, sender, text):
    return EvidenceMessage(
        message=ChatMessage(at, sender, text),
        platform="kakao",
        conversation_id="c",
        context=EvidenceContext.KAKAO_DIRECT,
        sender_person_id=sender,
        evidence_weight=1.0,
    )


def _evidence(messages) -> PersonEvidence:
    return PersonEvidence(
        "target",
        (
            EvidenceConversation(
                platform="kakao",
                conversation_id="c",
                context=EvidenceContext.KAKAO_DIRECT,
                messages=tuple(messages),
            ),
        ),
    )


def test_response_policy_learns_that_some_user_bursts_receive_no_reply() -> None:
    messages = [
        _message(BASE, "self", "뭐해"),
        _message(BASE + timedelta(minutes=1), "target", "집"),
        # User writes again twice before target responds: first burst becomes a
        # negative observation rather than forcing a future reply.
        _message(BASE + timedelta(minutes=10), "self", "야"),
        _message(BASE + timedelta(minutes=13), "self", "있냐"),
        _message(BASE + timedelta(minutes=14), "target", "왜"),
        _message(BASE + timedelta(minutes=20), "self", "끝"),
        _message(BASE + timedelta(minutes=21), "target", "ㅇㅇ"),
    ]
    policy = LiveResponsePolicy.from_evidence(
        _evidence(messages),
        self_person_id="self",
        focus_conversation_id="c",
        burst_gap_seconds=120,
        seed=1,
    )
    assert 0.0 < policy.global_reply_probability < 1.0


def test_final_user_burst_is_right_censored_not_fake_wait() -> None:
    messages = [
        _message(BASE, "self", "뭐해"),
        _message(BASE + timedelta(minutes=1), "target", "집"),
        _message(BASE + timedelta(minutes=10), "self", "마지막 기록"),
    ]
    policy = LiveResponsePolicy.from_evidence(
        _evidence(messages),
        self_person_id="self",
        focus_conversation_id="c",
        seed=1,
    )
    # Only the first complete user→target observation contributes.
    assert policy.global_reply_probability > 0.5


def test_continuation_policy_keeps_follow_up_and_new_session_separate() -> None:
    messages = [
        _message(BASE, "target", "a"),
        _message(BASE + timedelta(minutes=10), "target", "b"),
        _message(BASE + timedelta(minutes=11), "self", "ㅇㅇ"),
        _message(BASE + timedelta(days=1), "target", "야"),
        _message(BASE + timedelta(days=2), "target", "또"),
        _message(BASE + timedelta(days=2, minutes=1), "self", "왜"),
    ]
    policy = TargetContinuationPolicy.from_evidence(
        _evidence(messages),
        focus_conversation_id="c",
        seed=2,
    )
    assert policy.follow_up.global_probability > 0.0
    assert policy.initiate.global_probability > 0.0
    assert policy.follow_up is not policy.initiate


def test_response_policy_can_return_wait_with_deterministic_low_probability_policy() -> None:
    # Build many explicit negatives so the sampled choice is observably allowed to
    # be WAIT rather than treating REPLY as a mandatory action.
    messages = []
    cursor = BASE
    for index in range(8):
        messages.append(_message(cursor, "self", f"u{index}"))
        cursor += timedelta(minutes=3)
        messages.append(_message(cursor, "self", f"u{index}-again"))
        cursor += timedelta(minutes=3)
    messages.append(_message(cursor, "target", "finally"))
    policy = LiveResponsePolicy.from_evidence(
        _evidence(messages),
        self_person_id="self",
        focus_conversation_id="c",
        burst_gap_seconds=60,
        seed=4,
    )
    now = cursor + timedelta(minutes=1)
    context = (_message(now, "self", "새 메시지"),)
    choices = {
        policy.choose_after_counterpart_message(observed_at=now, visible_context=context)
        for _ in range(20)
    }
    assert Action.WAIT in choices
