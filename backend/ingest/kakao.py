from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path

from backend.models import ChatMessage, MemorySource

_DATE_SEPARATOR = re.compile(
    r"^-+\s*(?P<year>\d{4})년\s*(?P<month>\d{1,2})월\s*(?P<day>\d{1,2})일.*?-+$"
)

_PLAIN_DATE_HEADER = re.compile(
    r"^(?P<year>\d{4})년\s*(?P<month>\d{1,2})월\s*(?P<day>\d{1,2})일(?:\s+.+)?$"
)

_BRACKET_MESSAGE = re.compile(
    r"^\[(?P<sender>.+?)\]\s*\[(?P<ampm>오전|오후)\s*(?P<hour>\d{1,2}):(?P<minute>\d{2})\]\s?(?P<text>.*)$"
)

_INLINE_MESSAGE = re.compile(
    r"^(?P<year>\d{4})년\s*(?P<month>\d{1,2})월\s*(?P<day>\d{1,2})일\s*"
    r"(?P<ampm>오전|오후)\s*(?P<hour>\d{1,2}):(?P<minute>\d{2}),\s*"
    r"(?P<sender>.+?)\s*:\s*(?P<text>.*)$"
)

_DOTTED_MESSAGE = re.compile(
    r"^(?P<year>\d{4})\.\s*(?P<month>\d{1,2})\.\s*(?P<day>\d{1,2})\.\s*"
    r"(?P<ampm>오전|오후)\s*(?P<hour>\d{1,2}):(?P<minute>\d{2}),\s*"
    r"(?P<sender>.+?)\s*:\s*(?P<text>.*)$"
)

_SYSTEM_EVENT = re.compile(
    r"^\d{4}\.\s*\d{1,2}\.\s*\d{1,2}\.\s*(?:오전|오후)\s*\d{1,2}:\d{2}:\s*.+$"
)


def _hour24(ampm: str, hour: int) -> int:
    if not 1 <= hour <= 12:
        raise ValueError(f"invalid 12-hour clock hour: {hour}")
    if ampm == "오전":
        return 0 if hour == 12 else hour
    return 12 if hour == 12 else hour + 12


def _timestamp(day: date, ampm: str, hour: str, minute: str) -> datetime:
    return datetime(
        day.year,
        day.month,
        day.day,
        _hour24(ampm, int(hour)),
        int(minute),
    )


def _day_from_match(match: re.Match[str]) -> date:
    return date(
        int(match["year"]),
        int(match["month"]),
        int(match["day"]),
    )


def parse_kakao_text(text: str) -> list[ChatMessage]:
    """Parse common Korean KakaoTalk text-export formats.

    Supported formats include mobile-style bracket exports and desktop exports
    such as ``2026. 8. 7. 오후 9:11, 이름 : 메시지``. KakaoTalk preamble,
    day headers, and membership/system events are not emitted as user messages.
    Lines belonging to a multiline message are preserved, while trailing blank
    separator lines are removed when the message is flushed.

    Kakao exports carry only minute-resolution timestamps. That precision is
    recorded in metadata so replay evaluation can treat delay as an interval
    instead of pretending a displayed ``12:03`` is exact to the second.
    """

    messages: list[ChatMessage] = []
    current_date: date | None = None
    pending: dict[str, object] | None = None

    def flush() -> None:
        nonlocal pending
        if pending is None:
            return
        message_text = "\n".join(pending["lines"]).rstrip()  # type: ignore[arg-type]
        messages.append(
            ChatMessage(
                timestamp=pending["timestamp"],  # type: ignore[arg-type]
                sender=pending["sender"],  # type: ignore[arg-type]
                text=message_text,
                source=MemorySource.REAL,
                metadata={
                    "platform": "kakao",
                    "timestamp_precision_seconds": 60.0,
                },
            )
        )
        pending = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip("\r\n")
        stripped = line.strip()

        date_match = _DATE_SEPARATOR.match(stripped) or _PLAIN_DATE_HEADER.match(stripped)
        if date_match:
            flush()
            current_date = _day_from_match(date_match)
            continue

        inline = _INLINE_MESSAGE.match(line) or _DOTTED_MESSAGE.match(line)
        if inline:
            flush()
            day = _day_from_match(inline)
            pending = {
                "timestamp": _timestamp(day, inline["ampm"], inline["hour"], inline["minute"]),
                "sender": inline["sender"].strip(),
                "lines": [inline["text"]],
            }
            current_date = day
            continue

        bracket = _BRACKET_MESSAGE.match(line)
        if bracket and current_date is not None:
            flush()
            pending = {
                "timestamp": _timestamp(
                    current_date,
                    bracket["ampm"],
                    bracket["hour"],
                    bracket["minute"],
                ),
                "sender": bracket["sender"].strip(),
                "lines": [bracket["text"]],
            }
            continue

        if _SYSTEM_EVENT.match(line):
            flush()
            continue

        if stripped.startswith("Talk_") or stripped.startswith("저장한 날짜"):
            continue

        if pending is not None:
            pending["lines"].append(line)  # type: ignore[union-attr]

    flush()
    return messages


def parse_kakao_file(path: str | Path) -> list[ChatMessage]:
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    return parse_kakao_text(text)
