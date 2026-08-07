import json
import zipfile
from datetime import datetime, timezone

from backend.ingest.instagram import load_instagram_export, repair_meta_text
from backend.models import MessageType


def test_repairs_meta_mojibake() -> None:
    assert repair_meta_text("ì\x9d´ì\x9d\x80ì\x84¸") == "이은세"
    assert repair_meta_text("plain_ascii") == "plain_ascii"
    assert repair_meta_text("이미 정상") == "이미 정상"


def test_loads_dm_and_activity_counts(tmp_path) -> None:
    path = tmp_path / "instagram.zip"
    timestamp_ms = int(datetime(2026, 8, 7, 12, 0, tzinfo=timezone.utc).timestamp() * 1000)
    thread = {
        "participants": [{"name": "ì\x9d´ì\x9d\x80ì\x84¸"}, {"name": "friend"}],
        "messages": [
            {
                "sender_name": "friend",
                "timestamp_ms": timestamp_ms + 1000,
                "photos": [{"uri": "photo.jpg"}],
            },
            {
                "sender_name": "ì\x9d´ì\x9d\x80ì\x84¸",
                "timestamp_ms": timestamp_ms,
                "content": "ì\x95\x88ë\x85\x95",
                "share": {"link": "https://example.invalid/post"},
            },
        ],
    }

    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "your_instagram_activity/messages/inbox/friend_123/message_1.json",
            json.dumps(thread),
        )
        archive.writestr(
            "your_instagram_activity/likes/liked_posts.json",
            json.dumps([{}, {}]),
        )
        archive.writestr(
            "connections/followers_and_following/followers_1.json",
            json.dumps([{}]),
        )
        archive.writestr(
            "connections/followers_and_following/following.json",
            json.dumps({"relationships_following": [{}, {}, {}]}),
        )
        archive.writestr(
            "your_instagram_activity/media/stories.json",
            json.dumps({"ig_stories": [{}, {}]}),
        )

    export = load_instagram_export(path)
    assert export.activity.dm_threads == 1
    assert export.activity.dm_messages == 2
    assert export.activity.text_messages == 1
    assert export.activity.shared_items == 1
    assert export.activity.photo_messages == 1
    assert export.activity.liked_posts == 2
    assert export.activity.followers == 1
    assert export.activity.following == 3
    assert export.activity.stories == 2

    parsed = export.threads[0]
    assert parsed.participants == ("이은세", "friend")
    assert [message.text for message in parsed.messages] == ["안녕", ""]
    assert parsed.messages[0].timestamp == datetime(2026, 8, 7, 21, 0)
    assert parsed.messages[1].message_type == MessageType.MEDIA
