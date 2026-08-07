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


def test_parses_desktop_export_preamble_day_header_and_system_event() -> None:
    text = """Talk_2026.8.7 13:01-1.txt
저장한 날짜 : 2026. 8. 7. 오후 11:01

2026년 8월 7일 금요일
2026. 8. 7. 오후 1:01, 나 : 안녕
2026. 8. 7. 오후 1:02: 상대님이 나님을 초대했습니다.
2026. 8. 7. 오후 1:03, 상대 : 어 ㅋㅋ

2026년 8월 8일 토요일
2026. 8. 8. 오전 12:04, 나 : 잘자
"""

    messages = parse_kakao_text(text)

    assert len(messages) == 3
    assert [(message.sender, message.text) for message in messages] == [
        ("나", "안녕"),
        ("상대", "어 ㅋㅋ"),
        ("나", "잘자"),
    ]
    assert messages[0].timestamp == datetime(2026, 8, 7, 13, 1)
    assert messages[2].timestamp == datetime(2026, 8, 8, 0, 4)


def test_desktop_export_preserves_internal_multiline_but_drops_day_separator_blanks() -> None:
    text = """2026년 8월 7일 금요일
2026. 8. 7. 오후 9:11, 나 : 첫 줄

https://example.invalid

2026년 8월 8일 토요일
2026. 8. 8. 오전 9:00, 상대 : 다음날
"""

    messages = parse_kakao_text(text)

    assert messages[0].text == "첫 줄\n\nhttps://example.invalid"
    assert messages[1].text == "다음날"
