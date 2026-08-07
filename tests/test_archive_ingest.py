from __future__ import annotations

import io
import zipfile

from backend.audit import audit_archive
from backend.ingest.archive import load_kakao_archive


def _chat_zip(name: str, text: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, text.encode("utf-8-sig"))
        archive.writestr("ignored.png", b"not-an-image-but-ignored")
    return buffer.getvalue()


def test_loads_outer_bundle_without_extracting_attachments(tmp_path) -> None:
    first = _chat_zip(
        "Talk_first.txt",
        """2026년 8월 7일 금요일
2026. 8. 7. 오후 9:00, 나 : 안녕
2026. 8. 7. 오후 9:01, 친구A : ㅎㅇ
""",
    )
    second = _chat_zip(
        "Talk_second.txt",
        """2026년 8월 8일 토요일
2026. 8. 8. 오전 10:00, 나 : 굿모닝
2026. 8. 8. 오전 10:02, 친구B : 굿모닝
""",
    )

    bundle = tmp_path / "all.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("folder/Kakaotalk_Chat_친구A_20260807.zip", first)
        archive.writestr("folder/Kakaotalk_Chat_친구B_20260808.zip", second)

    conversations = load_kakao_archive(bundle)

    assert len(conversations) == 2
    assert [conversation.chat_name for conversation in conversations] == [
        "친구A_20260807",
        "친구B_20260808",
    ]
    assert [len(conversation.messages) for conversation in conversations] == [2, 2]

    audit = audit_archive(bundle)
    assert audit["conversation_count"] == 2
    assert audit["message_count"] == 4
    assert audit["direct_conversation_count"] == 2
    assert audit["group_conversation_count"] == 0
