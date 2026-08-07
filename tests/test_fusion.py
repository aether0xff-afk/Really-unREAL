from datetime import datetime

from backend.fusion import (
    EvidenceContext,
    canonical_target_messages,
    collect_person_evidence,
)
from backend.identity import IdentityMap, PersonEntity
from backend.ingest.archive import ConversationExport
from backend.ingest.instagram import InstagramThread
from backend.models import ChatMessage


def _message(minute: int, sender: str, text: str) -> ChatMessage:
    return ChatMessage(datetime(2026, 8, 7, 20, minute), sender, text)


def test_collects_direct_and_group_evidence_without_flattening_context() -> None:
    identities = IdentityMap(
        [
            PersonEntity(
                "self",
                {"kakao": ("나",), "instagram": ("self_ig",)},
                is_self=True,
            ),
            PersonEntity(
                "target",
                {"kakao": ("상대",), "instagram": ("target_ig",)},
            ),
        ]
    )

    kakao_direct = ConversationExport(
        chat_name="상대",
        source_archive="target.zip",
        source_text="Talk.txt",
        messages=(
            _message(0, "나", "야"),
            _message(1, "상대", "왜"),
        ),
    )
    kakao_group = ConversationExport(
        chat_name="단톡",
        source_archive="group.zip",
        source_text="Talk.txt",
        messages=(
            _message(2, "나", "ㅎㅇ"),
            _message(3, "상대", "ㅎㅇ"),
            _message(4, "다른사람", "ㅇ"),
        ),
    )
    instagram_direct = InstagramThread(
        thread_id="target_thread",
        participants=("self_ig", "target_ig"),
        messages=(
            _message(5, "self_ig", "reel"),
            _message(6, "target_ig", "ㅋㅋ"),
        ),
    )

    evidence = collect_person_evidence(
        "target",
        identities,
        kakao_conversations=[kakao_direct, kakao_group],
        instagram_threads=[instagram_direct],
    )

    assert len(evidence.conversations) == 3
    assert evidence.counts_by_context() == {
        EvidenceContext.KAKAO_DIRECT.value: 1,
        EvidenceContext.KAKAO_GROUP.value: 1,
        EvidenceContext.INSTAGRAM_DIRECT.value: 1,
        EvidenceContext.INSTAGRAM_GROUP.value: 0,
    }

    target_messages = canonical_target_messages(evidence)
    assert [message.text for message in target_messages] == ["왜", "ㅎㅇ", "ㅋㅋ"]
    assert all(message.sender == "target" for message in target_messages)
    assert target_messages[0].metadata["evidence_weight"] == 1.0
    assert target_messages[1].metadata["evidence_weight"] == 0.35
    assert target_messages[2].metadata["platform"] == "instagram"


def test_unmapped_similar_alias_is_not_silently_fused() -> None:
    identities = IdentityMap(
        [
            PersonEntity("target", {"kakao": ("임명민",)}),
        ]
    )
    instagram = InstagramThread(
        thread_id="myeongmin",
        participants=("명민", "self"),
        messages=(_message(0, "명민", "안녕"),),
    )

    evidence = collect_person_evidence(
        "target",
        identities,
        instagram_threads=[instagram],
    )

    assert evidence.conversations == ()
