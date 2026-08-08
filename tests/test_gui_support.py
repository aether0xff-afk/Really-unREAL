from datetime import datetime, timedelta

from backend.gui_entry import bundle_smoke
from backend.gui_support import (
    build_quick_identity_map,
    direct_targets_for_self,
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


def test_bundle_smoke_imports_packaged_resources() -> None:
    bundle_smoke()
