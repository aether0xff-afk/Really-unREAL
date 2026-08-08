from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta

from backend.fusion import PersonEvidence


_ABSOLUTE_DATE_RE = re.compile(
    r"(?:(?P<year>20\d{2})[.\-/년]\s*)?"
    r"(?P<month>\d{1,2})[.\-/월]\s*(?P<day>\d{1,2})(?:일)?"
)
_TIME_RE = re.compile(
    r"(?:(?P<ampm>오전|오후)\s*)?(?P<hour>\d{1,2})(?::(?P<minute>\d{2})|시(?:\s*(?P<minute2>\d{1,2})분)?)"
)
_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+")
_RELATIVE_DAYS = {"오늘": 0, "내일": 1, "모레": 2, "글피": 3, "어제": -1}
_STOPWORDS = {
    "오늘", "내일", "모레", "글피", "어제", "오전", "오후", "시", "분",
    "ㅋㅋ", "ㅎㅎ", "ㅠㅠ", "ㅜㅜ", "ㅇㅇ", "ㄴㄴ", "그거", "이거", "저거",
    "나", "너", "우리", "근데", "그냥", "진짜", "뭐", "왜",
}


@dataclass(frozen=True, slots=True)
class ObservableEventCue:
    """A date/time cue explicitly supported by a past message.

    The cue stores only a resolved timestamp and a few nearby content tokens. It
    is not a claim that the event definitely happened; it is evidence that the
    conversation mentioned an event around that time.
    """

    event_at: str
    label_tokens: tuple[str, ...]
    mention_count: int
    focused_mention_count: int
    last_mentioned_at: str
    score: float


@dataclass(frozen=True, slots=True)
class EventMemorySnapshot:
    cutoff: str
    cues: tuple[ObservableEventCue, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _hour24(ampm: str | None, hour: int) -> int:
    if ampm is None:
        return min(23, max(0, hour))
    if not 1 <= hour <= 12:
        return min(23, max(0, hour))
    if ampm == "오전":
        return 0 if hour == 12 else hour
    return 12 if hour == 12 else hour + 12


def _time_from_text(text: str) -> time | None:
    match = _TIME_RE.search(text)
    if not match:
        return None
    hour = _hour24(match["ampm"], int(match["hour"]))
    minute = int(match["minute"] or match["minute2"] or 0)
    if minute > 59:
        return None
    return time(hour, minute)


def _resolved_dates(text: str, mentioned_at: datetime) -> list[date]:
    output: list[date] = []
    for token, offset in _RELATIVE_DAYS.items():
        if token in text:
            output.append((mentioned_at + timedelta(days=offset)).date())

    for match in _ABSOLUTE_DATE_RE.finditer(text):
        year = int(match["year"] or mentioned_at.year)
        month = int(match["month"])
        day = int(match["day"])
        try:
            candidate = date(year, month, day)
        except ValueError:
            continue
        # A yearless date mentioned late in the year may refer to next year.
        if match["year"] is None and candidate < mentioned_at.date() - timedelta(days=180):
            try:
                candidate = date(year + 1, month, day)
            except ValueError:
                pass
        output.append(candidate)
    return output


def _label_tokens(text: str, *, limit: int = 4) -> tuple[str, ...]:
    tokens: list[str] = []
    for token in _TOKEN_RE.findall(text.lower()):
        if token in _STOPWORDS or token.isdigit():
            continue
        if len(token) == 1 and not token.isascii():
            continue
        if token not in tokens:
            tokens.append(token)
        if len(tokens) >= limit:
            break
    return tuple(tokens)


def build_event_memory(
    evidence: PersonEvidence,
    cutoff: datetime,
    *,
    focus_conversation_id: str | None = None,
    focus_platform: str | None = None,
    history_days: float = 180.0,
    future_window_days: float = 60.0,
    recent_past_days: float = 14.0,
    focus_weight_multiplier: float = 3.0,
    top_k: int = 8,
) -> EventMemorySnapshot:
    """Extract explicit date/time mentions from observable history only.

    Upcoming cues and very recent past cues are retained because both can drive
    natural continuation ("내일 시험" before it, "시험 어땠어" after it). Nothing
    from messages at or after the replay cutoff is used.
    """

    if history_days <= 0 or future_window_days < 0 or recent_past_days < 0:
        raise ValueError("event-memory windows must be non-negative and history > 0")
    if focus_weight_multiplier < 1.0:
        raise ValueError("focus_weight_multiplier must be >= 1")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    aggregate: dict[tuple[datetime, tuple[str, ...]], list[object]] = {}

    for conversation in evidence.conversations:
        focused = (
            focus_conversation_id is not None
            and conversation.conversation_id == focus_conversation_id
            and (focus_platform is None or conversation.platform == focus_platform)
        )
        for item in conversation.messages:
            mentioned_at = item.message.timestamp
            if mentioned_at >= cutoff:
                continue
            age_days = (cutoff - mentioned_at).total_seconds() / 86400.0
            if age_days > history_days:
                continue

            text = item.message.text
            dates = _resolved_dates(text, mentioned_at)
            if not dates:
                continue
            mentioned_time = _time_from_text(text) or time(12, 0)
            labels = _label_tokens(text)
            for day in dates:
                event_at = datetime.combine(day, mentioned_time)
                delta_days = (event_at - cutoff).total_seconds() / 86400.0
                if delta_days > future_window_days or delta_days < -recent_past_days:
                    continue
                recency = 1.0 / (1.0 + age_days / 30.0)
                proximity = 1.0 / (1.0 + abs(delta_days) / 7.0)
                weight = max(0.0, float(item.evidence_weight)) * recency * proximity
                if focused:
                    weight *= focus_weight_multiplier
                key = (event_at, labels)
                if key not in aggregate:
                    aggregate[key] = [0.0, 0, 0, mentioned_at]
                cell = aggregate[key]
                cell[0] = float(cell[0]) + weight
                cell[1] = int(cell[1]) + 1
                cell[2] = int(cell[2]) + int(focused)
                if mentioned_at > cell[3]:  # type: ignore[operator]
                    cell[3] = mentioned_at

    ordered = sorted(
        aggregate.items(),
        key=lambda item: (
            -float(item[1][0]),
            abs((item[0][0] - cutoff).total_seconds()),
            item[0][0],
        ),
    )[:top_k]
    cues = tuple(
        ObservableEventCue(
            event_at=event_at.isoformat(),
            label_tokens=labels,
            mention_count=int(cell[1]),
            focused_mention_count=int(cell[2]),
            last_mentioned_at=cell[3].isoformat(),  # type: ignore[union-attr]
            score=round(float(cell[0]), 6),
        )
        for (event_at, labels), cell in ordered
    )
    return EventMemorySnapshot(cutoff=cutoff.isoformat(), cues=cues)
