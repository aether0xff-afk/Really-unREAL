from datetime import datetime

from backend.ingest.kakao import parse_kakao_text


def test_parses_bracket_export_and_multiline_message() -> None:
    text = """--------------- 2026년 8월 7일 금요일 ---------------
[나] [오후 11:03] 야
[상대] [오후 11:04] 왜ㅋㅋ
두번째 줄
[나] [오전 12:05] 자냐
"""

    messages = parse_kakao_text(text)

    assert len(messages) == 3
    assert messages[0].timestamp == datetime(2026, 8, 7, 23, 3)
    assert messages[1].sender == "상대"
    assert messages[1].text == "왜ㅋㅋ\n두번째 줄"
    assert messages[2].timestamp == datetime(2026, 8, 7, 0, 5)


def test_parses_inline_export() -> None:
    text = """2026년 8월 7일 오후 9:11, 나 : 뭐해
2026년 8월 7일 오후 9:13, 상대 : 집
"""

    messages = parse_kakao_text(text)

    assert [(message.sender, message.text) for message in messages] == [
        ("나", "뭐해"),
        ("상대", "집"),
    ]
    assert messages[1].timestamp == datetime(2026, 8, 7, 21, 13)
