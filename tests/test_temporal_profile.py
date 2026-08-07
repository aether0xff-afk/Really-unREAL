from datetime import datetime, timedelta

from backend.models import ChatMessage
from backend.persona.temporal import build_temporal_profile


def test_temporal_profile_reply_delay_and_session_initiation() -> None:
    start = datetime(2026, 8, 7, 10, 0)
    messages = [
        ChatMessage(start, "나", "첫 세션 시작"),
        ChatMessage(start + timedelta(minutes=5), "상대", "답장"),
        ChatMessage(start + timedelta(minutes=6), "상대", "연속 메시지"),
        ChatMessage(start + timedelta(hours=8), "상대", "둘째 세션 선톡"),
        ChatMessage(start + timedelta(hours=8, minutes=2), "나", "답"),
    ]

    profile = build_temporal_profile(messages, "상대", session_gap_hours=6)

    assert profile.reply_delay_seconds == (300.0,)
    assert profile.reply_delay_median_seconds == 300.0
    assert profile.total_sessions == 2
    assert profile.initiated_sessions == 1
    assert profile.initiation_rate == 0.5
    assert profile.active_hour_counts[10] == 2
    assert profile.active_hour_counts[18] == 1
