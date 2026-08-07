from datetime import datetime, timedelta

from backend.models import ChatMessage
from backend.persona.language import build_language_profile


def test_language_profile_uses_only_target_sender() -> None:
    base = datetime(2026, 8, 7, 20, 0)
    messages = [
        ChatMessage(base, "상대", "ㅋㅋㅋ"),
        ChatMessage(base + timedelta(minutes=1), "나", "긴 문장을 일부러 넣는다"),
        ChatMessage(base + timedelta(minutes=2), "상대", "아니ㅋㅋ"),
        ChatMessage(base + timedelta(minutes=3), "상대", "ㅠㅠ"),
    ]

    profile = build_language_profile(messages, "상대")

    assert profile.message_count == 3
    assert profile.laugh_expression_ratio == 0.6667
    assert profile.cry_expression_ratio == 0.3333
    assert profile.short_message_ratio == 1.0
