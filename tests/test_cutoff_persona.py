from datetime import datetime, timedelta

from backend.fusion import EvidenceContext, EvidenceConversation, EvidenceMessage, PersonEvidence
from backend.models import ChatMessage
from backend.persona.cutoff import build_cutoff_language_profile, target_messages_before


BASE = datetime(2026, 8, 7, 20, 0)


def _message(
    at: datetime,
    text: str,
    *,
    platform: str = "kakao",
    weight: float = 1.0,
    conversation_id: str | None = None,
) -> EvidenceMessage:
    context = (
        EvidenceContext.KAKAO_DIRECT
        if platform == "kakao"
        else EvidenceContext.INSTAGRAM_DIRECT
    )
    return EvidenceMessage(
        message=ChatMessage(at, "target", text),
        platform=platform,
        conversation_id=conversation_id or platform,
        context=context,
        sender_person_id="target",
        evidence_weight=weight,
    )


def _evidence(messages: tuple[EvidenceMessage, ...]) -> PersonEvidence:
    conversations = tuple(
        EvidenceConversation(
            platform=message.platform,
            conversation_id=message.conversation_id,
            context=message.context,
            messages=(message,),
        )
        for message in messages
    )
    return PersonEvidence(person_id="target", conversations=conversations)


def test_future_messages_do_not_affect_cutoff_profile() -> None:
    past = _message(BASE - timedelta(days=1), "ㅋㅋ 짧음")
    future = _message(BASE + timedelta(days=1), "미래에 생긴 완전히 다른 말버릇입니다")
    evidence = _evidence((past, future))

    profile = build_cutoff_language_profile(evidence, BASE)

    assert profile.message_count == 1
    assert profile.platform_message_counts == {"kakao": 1}
    assert any(token == "ㅋㅋ" for token, _ in profile.frequent_tokens)
    assert all("미래" not in token for token, _ in profile.frequent_tokens)


def test_same_timestamp_as_cutoff_is_excluded() -> None:
    same = _message(BASE, "경계")
    evidence = _evidence((same,))

    assert target_messages_before(evidence, BASE) == []


def test_instagram_is_supplemental_in_weighted_language_profile() -> None:
    kakao = _message(
        BASE - timedelta(days=2),
        "짧아",
        platform="kakao",
        weight=1.0,
    )
    instagram = _message(
        BASE - timedelta(days=1),
        "이 메시지는 인스타그램에서 훨씬 더 길게 작성된 보조 자료",
        platform="instagram",
        weight=0.55,
    )
    evidence = _evidence((kakao, instagram))

    profile = build_cutoff_language_profile(evidence, BASE)
    unweighted_mean = (len(kakao.message.text) + len(instagram.message.text)) / 2

    assert profile.effective_message_weight == 1.55
    assert profile.weighted_mean_char_length < unweighted_mean


def test_current_relationship_can_shift_self_twin_style_without_erasing_global_fallback() -> None:
    focused = _message(
        BASE - timedelta(days=2),
        "ㅇㅇ",
        conversation_id="friend-a",
    )
    global_other = _message(
        BASE - timedelta(days=1),
        "이쪽 대화에서는 문장을 꽤 길게 작성하는 편임",
        conversation_id="friend-b",
    )
    evidence = _evidence((focused, global_other))

    global_profile = build_cutoff_language_profile(evidence, BASE)
    focused_profile = build_cutoff_language_profile(
        evidence,
        BASE,
        focus_conversation_id="friend-a",
        focus_platform="kakao",
        focus_weight_multiplier=4.0,
    )

    assert focused_profile.profile_scope == "relationship_blend"
    assert focused_profile.focused_message_count == 1
    assert focused_profile.message_count == 2
    assert focused_profile.weighted_mean_char_length < global_profile.weighted_mean_char_length
