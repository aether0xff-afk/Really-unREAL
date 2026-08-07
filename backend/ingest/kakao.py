from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path

from backend.models import ChatMessage, MemorySource

_DATE_SEPARATOR = re.compile(
    r"^-+\s*(?P<year>\d{4})년\s*(?P<month>\d{1,2})월\s*(?P<day>\d{1,2})일.*?-+$"
)

_BRACKET_MESSAGE = re.compile(
    r"^\[(?P<sender>.+?)\]\s*\[(?P<ampm>오전|오후)\s*(?P<hour>\d{1,2}):(?P<minute>\d{2})\]\s?(?P<text>.*)$"
)

_INLINE_MESSAGE = re.compile(
    r"^(?P<year>\d{4})년\s*(?P<month>\d{1,2})월\s*(?P<day>\d{1,2})일\s*"
    r"(?P<ampm>오전|오후)\s*(?P<hour>\d{1,2}):(?P<minute>\d{2}),\s*"
    r"(?P<sender>.+?)\s*:\s*(?P<text>.*)$"
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


def parse_kakao_text(text: str) -> list[ChatMessage]:
    """Parse common Korean KakaoTalk text-export formats.

    The parser currently supports both the date-separator + bracket form and
    the inline timestamp form. Unknown preamble lines are ignored. Lines after
    a recognized message that do not start another record are preserved as
    multiline message content.
    """

    messages: list[ChatMessage] = []
    current_date: date | None = None
    pending: dict[str, object] | None = None

    def flush() -> None:
        nonlocal pending
        if pending is None:
            return
        messages.append(
            ChatMessage(
                timestamp=pending["timestamp"],  # type: ignore[arg-type]
                sender=pending["sender"],  # type: ignore[arg-type]
                text="\n".join(pending["lines"]),  # type: ignore[arg-type]
                source=MemorySource.REAL,
            )
        )
        pending = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip("\r\n")

        date_match = _DATE_SEPARATOR.match(line.strip())
        if date_match:
            flush()
            current_date = date(
                int(date_match["year"]),
                int(date_match["month"]),
                int(date_match["day"]),
            )
            continue

        inline = _INLINE_MESSAGE.match(line)
        if inline:
            flush()
            day = date(int(inline["year"]), int(inline["month"]), int(inline["day"]))
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

        if pending is not None:
            pending["lines"].append(line)  # type: ignore[union-attr]

    flush()
    return messages


def parse_kakao_file(path: str | Path) -> list[ChatMessage]:
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    return parse_kakao_text(text)
