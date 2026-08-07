from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from backend.models import ChatMessage, MemorySource, MessageType


@dataclass(frozen=True, slots=True)
class InstagramThread:
    thread_id: str
    participants: tuple[str, ...]
    messages: tuple[ChatMessage, ...]


@dataclass(frozen=True, slots=True)
class InstagramActivitySummary:
    dm_threads: int
    dm_messages: int
    text_messages: int
    shared_items: int
    photo_messages: int
    reaction_messages: int
    liked_posts: int
    saved_posts: int
    saved_music: int
    stories: int
    other_content: int
    followers: int
    following: int
    close_friends: int


@dataclass(frozen=True, slots=True)
class InstagramExport:
    threads: tuple[InstagramThread, ...]
    activity: InstagramActivitySummary


def repair_meta_text(value: str) -> str:
    """Repair the common UTF-8-as-Latin-1 mojibake found in Meta JSON exports.

    ASCII and already-correct Unicode are returned unchanged.
    """

    try:
        repaired = value.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value
    return repaired


def _local_datetime(timestamp_ms: int, timezone_name: str) -> datetime:
    utc = datetime.fromtimestamp(timestamp_ms / 1000, tz=ZoneInfo("UTC"))
    return utc.astimezone(ZoneInfo(timezone_name)).replace(tzinfo=None)


def _count_json(
    archive: zipfile.ZipFile,
    path: str,
    *,
    key: str | None = None,
) -> int:
    try:
        data = json.loads(archive.read(path))
    except KeyError:
        return 0
    if key is not None:
        if not isinstance(data, dict):
            return 0
        data = data.get(key, [])
    return len(data) if isinstance(data, (list, dict)) else 0


def _message_files(archive: zipfile.ZipFile) -> list[str]:
    prefixes = (
        "your_instagram_activity/messages/inbox/",
        "your_instagram_activity/messages/hidden_threads/",
    )
    return sorted(
        name
        for name in archive.namelist()
        if name.endswith(".json") and name.startswith(prefixes)
    )


def load_instagram_export(
    path: str | Path,
    *,
    timezone_name: str = "Asia/Seoul",
) -> InstagramExport:
    """Load a Meta/Instagram information-download ZIP locally.

    Only JSON metadata is read. Photos and videos remain inside the archive and
    are referenced by their relative URI in ``ChatMessage.metadata``. DM
    timestamps are converted from epoch milliseconds into the selected local
    timezone so they can be compared with local messenger exports.
    """

    path = Path(path)
    threads: list[InstagramThread] = []
    text_messages = 0
    shared_items = 0
    photo_messages = 0
    reaction_messages = 0

    with zipfile.ZipFile(path) as archive:
        for filename in _message_files(archive):
            raw_thread = json.loads(archive.read(filename))
            participants = tuple(
                repair_meta_text(item.get("name", ""))
                for item in raw_thread.get("participants", [])
            )
            messages: list[ChatMessage] = []

            # Meta exports newest-first; normalize to chronological order.
            for raw in reversed(raw_thread.get("messages", [])):
                has_text = "content" in raw and bool(raw.get("content"))
                has_media = "photos" in raw or "share" in raw
                if "content" in raw:
                    text_messages += 1
                if "share" in raw:
                    shared_items += 1
                if "photos" in raw:
                    photo_messages += 1
                if "reactions" in raw:
                    reaction_messages += 1

                metadata: dict[str, object] = {
                    "platform": "instagram",
                    "thread_id": filename.split("/")[-2],
                }
                for key in ("share", "photos", "reactions"):
                    if key in raw:
                        metadata[key] = raw[key]

                messages.append(
                    ChatMessage(
                        timestamp=_local_datetime(raw["timestamp_ms"], timezone_name),
                        sender=repair_meta_text(raw.get("sender_name", "")),
                        text=repair_meta_text(raw.get("content", "")),
                        source=MemorySource.REAL,
                        message_type=MessageType.TEXT if has_text else (
                            MessageType.MEDIA if has_media else MessageType.SYSTEM
                        ),
                        metadata=metadata,
                    )
                )

            threads.append(
                InstagramThread(
                    thread_id=filename.split("/")[-2],
                    participants=participants,
                    messages=tuple(messages),
                )
            )

        activity = InstagramActivitySummary(
            dm_threads=len(threads),
            dm_messages=sum(len(thread.messages) for thread in threads),
            text_messages=text_messages,
            shared_items=shared_items,
            photo_messages=photo_messages,
            reaction_messages=reaction_messages,
            liked_posts=_count_json(
                archive, "your_instagram_activity/likes/liked_posts.json"
            ),
            saved_posts=_count_json(
                archive, "your_instagram_activity/saved/saved_posts.json"
            ),
            saved_music=_count_json(
                archive, "your_instagram_activity/saved/saved_music.json"
            ),
            stories=_count_json(
                archive,
                "your_instagram_activity/media/stories.json",
                key="ig_stories",
            ),
            other_content=_count_json(
                archive, "your_instagram_activity/media/other_content.json"
            ),
            followers=_count_json(
                archive, "connections/followers_and_following/followers_1.json"
            ),
            following=_count_json(
                archive,
                "connections/followers_and_following/following.json",
                key="relationships_following",
            ),
            close_friends=_count_json(
                archive, "connections/followers_and_following/close_friends.json"
            ),
        )

    threads.sort(
        key=lambda thread: thread.messages[0].timestamp
        if thread.messages
        else datetime.max
    )
    return InstagramExport(threads=tuple(threads), activity=activity)
