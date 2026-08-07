from __future__ import annotations

import io
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from backend.ingest.kakao import parse_kakao_text
from backend.models import ChatMessage


@dataclass(frozen=True, slots=True)
class ConversationExport:
    chat_name: str
    source_archive: str
    source_text: str
    messages: tuple[ChatMessage, ...]

    @property
    def participants(self) -> tuple[str, ...]:
        return tuple(sorted({message.sender for message in self.messages}))


def _normalize_name(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def _chat_name_from_archive(filename: str) -> str:
    stem = PurePosixPath(filename).stem
    stem = _normalize_name(stem)
    prefix = "Kakaotalk_Chat_"
    if stem.startswith(prefix):
        stem = stem[len(prefix) :]
    return stem


def _read_text_member(archive: zipfile.ZipFile, member: zipfile.ZipInfo) -> str:
    raw = archive.read(member)
    # KakaoTalk exports observed in the wild are UTF-8 with or without BOM.
    # cp949 is kept as a conservative fallback for older Korean exports.
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", raw, 0, min(len(raw), 1), "unsupported KakaoTalk text encoding")


def _parse_chat_zip(data: bytes, archive_name: str) -> list[ConversationExport]:
    conversations: list[ConversationExport] = []
    with zipfile.ZipFile(io.BytesIO(data)) as chat_zip:
        text_members = [
            member
            for member in chat_zip.infolist()
            if not member.is_dir() and member.filename.lower().endswith(".txt")
        ]
        for member in text_members:
            text = _read_text_member(chat_zip, member)
            messages = tuple(parse_kakao_text(text))
            if not messages:
                continue
            conversations.append(
                ConversationExport(
                    chat_name=_chat_name_from_archive(archive_name),
                    source_archive=_normalize_name(archive_name),
                    source_text=_normalize_name(member.filename),
                    messages=messages,
                )
            )
    return conversations


def load_kakao_archive(path: str | Path) -> list[ConversationExport]:
    """Load a KakaoTalk export ZIP without extracting private media to disk.

    Two layouts are supported:

    1. a single KakaoTalk chat ZIP containing ``Talk_*.txt``;
    2. an outer ZIP containing multiple per-chat KakaoTalk ZIP files.

    Non-text attachments are deliberately ignored during Phase 1.
    """

    path = Path(path)
    conversations: list[ConversationExport] = []

    with zipfile.ZipFile(path) as outer:
        text_members = [
            member
            for member in outer.infolist()
            if not member.is_dir() and member.filename.lower().endswith(".txt")
        ]
        nested_zips = [
            member
            for member in outer.infolist()
            if not member.is_dir() and member.filename.lower().endswith(".zip")
        ]

        # Single-chat export ZIP.
        if text_members:
            for member in text_members:
                text = _read_text_member(outer, member)
                messages = tuple(parse_kakao_text(text))
                if not messages:
                    continue
                conversations.append(
                    ConversationExport(
                        chat_name=_chat_name_from_archive(path.name),
                        source_archive=_normalize_name(path.name),
                        source_text=_normalize_name(member.filename),
                        messages=messages,
                    )
                )

        # Bundle of per-chat ZIPs, as produced when several exports are packed
        # together for local analysis.
        for member in nested_zips:
            conversations.extend(_parse_chat_zip(outer.read(member), member.filename))

    conversations.sort(
        key=lambda conversation: (
            conversation.messages[0].timestamp,
            conversation.chat_name,
        )
    )
    return conversations
