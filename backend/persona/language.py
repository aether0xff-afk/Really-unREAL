from __future__ import annotations

import re
import statistics
from collections import Counter
from dataclasses import asdict, dataclass

from backend.models import ChatMessage

_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣ㅋㅎㅠㅜ]+")
_LAUGH_RE = re.compile(r"ㅋ{2,}|ㅎ{2,}")
_CRY_RE = re.compile(r"ㅠ{2,}|ㅜ{2,}")


@dataclass(frozen=True, slots=True)
class LanguageProfile:
    sender: str
    message_count: int
    mean_char_length: float
    median_char_length: float
    short_message_ratio: float
    multiline_ratio: float
    laugh_expression_ratio: float
    cry_expression_ratio: float
    frequent_tokens: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["frequent_tokens"] = [list(item) for item in self.frequent_tokens]
        return data


def build_language_profile(
    messages: list[ChatMessage], sender: str, *, top_k: int = 12
) -> LanguageProfile:
    own = [message for message in messages if message.sender == sender]
    if not own:
        raise ValueError(f"no messages found for sender {sender!r}")

    lengths = [len(message.text) for message in own]
    tokens = Counter(
        token
        for message in own
        for token in _TOKEN_RE.findall(message.text.lower())
        if token.strip()
    )
    n = len(own)

    return LanguageProfile(
        sender=sender,
        message_count=n,
        mean_char_length=round(statistics.fmean(lengths), 3),
        median_char_length=float(statistics.median(lengths)),
        short_message_ratio=round(sum(length <= 5 for length in lengths) / n, 4),
        multiline_ratio=round(sum("\n" in message.text for message in own) / n, 4),
        laugh_expression_ratio=round(sum(bool(_LAUGH_RE.search(message.text)) for message in own) / n, 4),
        cry_expression_ratio=round(sum(bool(_CRY_RE.search(message.text)) for message in own) / n, 4),
        frequent_tokens=tuple(tokens.most_common(top_k)),
    )
