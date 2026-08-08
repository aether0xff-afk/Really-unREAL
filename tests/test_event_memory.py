from datetime import datetime, timedelta

from backend.event_memory import build_event_memory
from backend.fusion import EvidenceContext, EvidenceConversation, EvidenceMessage, PersonEvidence
from backend.models import ChatMessage


BASE = datetime(2026, 8, 8, 16, 0)


def _message(at: datetime, sender_id: str, text: str) -> EvidenceMessage:
    return EvidenceMessage(
        message=ChatMessage(at, sender_id, text),
        platform="kakao",
        conversation_id="c",
        context=EvidenceContext.KAKAO_DIRECT,
        sender_person_id=sender_id,
        evidence_weight=1.0,
    )


def test_event_memory_resolves_relative_date_without_future_leakage() -> None:
    past = _message(BASE - timedelta(days=1), "target", "내일 9시에 시험 있음")
    future = _message(BASE + timedelta(hours=1), "target", "모레 여행감")
    evidence = PersonEvidence(
        "target",
        (
            EvidenceConversation(
                "kakao",
                "c",
                EvidenceContext.KAKAO_DIRECT,
                (past, future),
            ),
        ),
    )

    memory = build_event_memory(
        evidence,
        BASE,
        focus_conversation_id="c",
        focus_platform="kakao",
    )

    assert len(memory.cues) == 1
    cue = memory.cues[0]
    assert cue.event_at.startswith("2026-08-08T09:00")
    assert "시험" in cue.label_tokens
    assert all("여행" not in item.label_tokens for item in memory.cues)


def test_recent_past_event_remains_available_for_followup() -> None:
    mentioned = _message(BASE - timedelta(days=2), "self", "내일 발표 3시")
    evidence = PersonEvidence(
        "target",
        (
            EvidenceConversation(
                "kakao",
                "c",
                EvidenceContext.KAKAO_DIRECT,
                (mentioned,),
            ),
        ),
    )

    memory = build_event_memory(evidence, BASE, focus_conversation_id="c")

    assert memory.cues
    assert "발표" in memory.cues[0].label_tokens
