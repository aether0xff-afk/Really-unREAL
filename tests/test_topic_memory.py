from datetime import datetime, timedelta

from backend.fusion import EvidenceContext, EvidenceConversation, EvidenceMessage, PersonEvidence
from backend.models import ChatMessage
from backend.topic_memory import build_topic_memory


BASE = datetime(2026, 8, 8, 12, 0)


def _message(at, sender, text, conversation_id):
    return EvidenceMessage(
        message=ChatMessage(at, sender, text),
        platform="kakao",
        conversation_id=conversation_id,
        context=EvidenceContext.KAKAO_DIRECT,
        sender_person_id=sender,
        evidence_weight=1.0,
    )


def test_topic_memory_is_cutoff_safe_and_relationship_focused() -> None:
    focus = EvidenceConversation(
        platform="kakao",
        conversation_id="friend-a",
        context=EvidenceContext.KAKAO_DIRECT,
        messages=(
            _message(BASE - timedelta(days=3), "self", "선형대수 과제 함?", "friend-a"),
            _message(BASE - timedelta(days=2), "friend", "선형대수 아직", "friend-a"),
            _message(BASE + timedelta(days=1), "self", "미래비밀 주제", "friend-a"),
        ),
    )
    other = EvidenceConversation(
        platform="kakao",
        conversation_id="friend-b",
        context=EvidenceContext.KAKAO_DIRECT,
        messages=(
            _message(BASE - timedelta(days=1), "other", "치킨 치킨 치킨", "friend-b"),
        ),
    )
    evidence = PersonEvidence("self", (focus, other))

    snapshot = build_topic_memory(
        evidence,
        BASE,
        focus_conversation_id="friend-a",
        focus_platform="kakao",
        focus_weight_multiplier=4.0,
        top_k=5,
    )

    tokens = [cue.token for cue in snapshot.cues]
    assert "선형대수" in tokens
    assert "미래비밀" not in tokens
    linear = next(cue for cue in snapshot.cues if cue.token == "선형대수")
    assert linear.focused_mention_count == 2
