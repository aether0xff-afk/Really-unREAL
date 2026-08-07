from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import asdict, dataclass

from backend.models import ChatMessage


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


@dataclass(frozen=True, slots=True)
class TemporalProfile:
    sender: str
    message_count: int
    active_hour_counts: dict[int, int]
    reply_delay_seconds: tuple[float, ...]
    reply_delay_median_seconds: float | None
    reply_delay_p25_seconds: float | None
    reply_delay_p75_seconds: float | None
    initiated_sessions: int
    total_sessions: int
    initiation_rate: float
    session_gap_hours: float

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["reply_delay_seconds"] = list(self.reply_delay_seconds)
        return data


def build_temporal_profile(
    messages: list[ChatMessage],
    sender: str,
    *,
    session_gap_hours: float = 6.0,
    max_reply_gap_hours: float = 24.0,
) -> TemporalProfile:
    ordered = sorted(messages, key=lambda message: message.timestamp)
    own = [message for message in ordered if message.sender == sender]
    if not own:
        raise ValueError(f"no messages found for sender {sender!r}")

    hour_counts = Counter(message.timestamp.hour for message in own)

    reply_delays: list[float] = []
    max_reply_seconds = max_reply_gap_hours * 3600
    for previous, current in zip(ordered, ordered[1:]):
        if current.sender != sender or previous.sender == sender:
            continue
        delay = (current.timestamp - previous.timestamp).total_seconds()
        if 0 <= delay <= max_reply_seconds:
            reply_delays.append(delay)

    session_gap_seconds = session_gap_hours * 3600
    session_starts: list[ChatMessage] = []
    for index, message in enumerate(ordered):
        if index == 0:
            session_starts.append(message)
            continue
        gap = (message.timestamp - ordered[index - 1].timestamp).total_seconds()
        if gap > session_gap_seconds:
            session_starts.append(message)

    total_sessions = len(session_starts)
    initiated_sessions = sum(message.sender == sender for message in session_starts)

    return TemporalProfile(
        sender=sender,
        message_count=len(own),
        active_hour_counts={hour: hour_counts.get(hour, 0) for hour in range(24)},
        reply_delay_seconds=tuple(reply_delays),
        reply_delay_median_seconds=(
            float(statistics.median(reply_delays)) if reply_delays else None
        ),
        reply_delay_p25_seconds=_quantile(reply_delays, 0.25),
        reply_delay_p75_seconds=_quantile(reply_delays, 0.75),
        initiated_sessions=initiated_sessions,
        total_sessions=total_sessions,
        initiation_rate=round(initiated_sessions / total_sessions, 4) if total_sessions else 0.0,
        session_gap_hours=session_gap_hours,
    )
