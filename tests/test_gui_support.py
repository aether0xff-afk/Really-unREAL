from datetime import datetime, timedelta
from pathlib import Path

import pytest

from backend.gui_entry import bundle_smoke
from backend.gui_support import (
    build_quick_identity_map,
    direct_targets_for_self,
    load_quick_kakao,
    rank_self_aliases,
)
from backend.ingest.archive import ConversationExport
from backend.models import ChatMessage


BASE = datetime(2026, 8, 8, 12, 0)


def _conversation(name: str, pairs: list[tuple[str, str]]) -> ConversationExport:
    messages = tuple(
        ChatMessage(BASE + timedelta(minutes=index), sender, text)
        for index, (sender, text) in enumerate(pairs)
    )
    return ConversationExport(
        chat_name=name,
        source_archive=f"{name}.zip",
        source_text="Talk.txt",
        messages=messages,
    )


def test_rank_self_aliases_prefers_name_seen_across_chats() -> None:
    conversations = [
        _conversation("A", [("나", "1"), ("A", "2"), ("나", "3")]),
        _conversation("B", [("나", "1"), ("B", "2")]),
        _conversation("C", [("나", "1"), ("C", "2")]),
    ]

    assert rank_self_aliases(conversations)[0] == "나"


def test_direct_targets_are_sorted_by_target_message_volume() -> None:
    conversations = [
        _conversation("A", [("나", "x"), ("A", "1"), ("A", "2")]),
        _conversation("B", [("나", "x"), ("B", "1")]),
    ]

    assert direct_targets_for_self(conversations, "나") == ["A", "B"]


def test_quick_identity_map_marks_confirmed_alias_as_self() -> None:
    conversations = [_conversation("A", [("나", "x"), ("A", "1")])]

    identities = build_quick_identity_map(conversations, "나")

    assert identities.self_person_id == "self"
    assert identities.resolve("kakao", "나") == "self"
    assert identities.resolve("kakao", "A") is not None
    assert identities.resolve("kakao", "A") != "self"


def test_load_quick_kakao_combines_multiple_selected_archives(monkeypatch) -> None:
    first = _conversation("A", [("나", "x"), ("A", "1")])
    second = _conversation("B", [("나", "y"), ("B", "2")])
    by_name = {"first.zip": [first], "second.zip": [second]}

    def fake_load(path):
        return by_name[Path(path).name]

    monkeypatch.setattr("backend.gui_support.load_kakao_archive", fake_load)

    loaded = load_quick_kakao(["first.zip", "second.zip"])

    assert {conversation.chat_name for conversation in loaded} == {"A", "B"}


def test_load_quick_kakao_deduplicates_same_conversation_from_multiple_archives(
    monkeypatch,
) -> None:
    duplicate = _conversation("A", [("나", "x"), ("A", "1")])

    monkeypatch.setattr(
        "backend.gui_support.load_kakao_archive",
        lambda _path: [duplicate],
    )

    loaded = load_quick_kakao(["first.zip", "duplicate-copy.zip"])

    assert len(loaded) == 1
    assert loaded[0].chat_name == "A"


def test_load_quick_kakao_does_not_load_same_selected_path_twice(monkeypatch) -> None:
    conversation = _conversation("A", [("나", "x"), ("A", "1")])
    calls: list[str] = []

    def fake_load(path):
        calls.append(str(path))
        return [conversation]

    monkeypatch.setattr("backend.gui_support.load_kakao_archive", fake_load)

    load_quick_kakao(["same.zip", "same.zip"])

    assert len(calls) == 1


def test_load_quick_kakao_requires_at_least_one_archive() -> None:
    with pytest.raises(ValueError, match="하나 이상"):
        load_quick_kakao([])


def test_bundle_smoke_imports_packaged_resources() -> None:
    bundle_smoke()
