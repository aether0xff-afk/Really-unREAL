from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Iterable

from backend.fusion import EvidenceMessage, PersonEvidence


_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣ㅋㅎㅠㅜ]+")
_LAUGH_RE = re.compile(r"ㅋ{2,}|ㅎ{2,}")
_CRY_RE = re.compile(r"ㅠ{2,}|ㅜ{2,}")


@dataclass(frozen=True, slots=True)
class CutoffLanguageProfile:
    person_id: str
    cutoff: str
    message_count: int
    effective_message_weight: float
    weighted_mean_char_length: float | None
    weighted_short_message_ratio: float | None
    weighted_laugh_expression_ratio: float | None
    weighted_cry_expression_ratio: float | None
    frequent_tokens: tuple[tuple[str, float], ...]
    platform_message_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["frequent_tokens"] = [list(item) for item in self.frequent_tokens]
        return data


def target_messages_before(
    evidence: PersonEvidence,
    cutoff: datetime,
) -> list[EvidenceMessage]:
    """Return target observations that are strictly older than ``cutoff``.

    Strict comparison is deliberate for coarse Kakao timestamps: a message with
    the same displayed minute as the replay cutoff is not assumed to precede it.
    """

    return sorted(
        (
            item
            for item in evidence.target_messages()
            if item.message.timestamp < cutoff
        ),
        key=lambda item: item.message.timestamp,
    )


def build_cutoff_language_profile(
    evidence: PersonEvidence,
    cutoff: datetime,
    *,
    top_k: int = 12,
) -> CutoffLanguageProfile:
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    messages = target_messages_before(evidence, cutoff)
    total_weight = sum(max(0.0, item.evidence_weight) for item in messages)
    platform_counts: defaultdict[str, int] = defaultdict(int)
    token_weights: defaultdict[str, float] = defaultdict(float)

    weighted_length = 0.0
    short_weight = 0.0
    laugh_weight = 0.0
    cry_weight = 0.0

    for item in messages:
        weight = max(0.0, float(item.evidence_weight))
        platform_counts[item.platform] += 1
        text = item.message.text
        weighted_length += len(text) * weight
        short_weight += (len(text) <= 5) * weight
        laugh_weight += bool(_LAUGH_RE.search(text)) * weight
        cry_weight += bool(_CRY_RE.search(text)) * weight
        for token in _TOKEN_RE.findall(text.lower()):
            if token.strip():
                token_weights[token] += weight

    if total_weight > 0:
        mean_length = weighted_length / total_weight
        short_ratio = short_weight / total_weight
        laugh_ratio = laugh_weight / total_weight
        cry_ratio = cry_weight / total_weight
    else:
        mean_length = None
        short_ratio = None
        laugh_ratio = None
        cry_ratio = None

    frequent_tokens = tuple(
        sorted(
            token_weights.items(),
            key=lambda item: (-item[1], item[0]),
        )[:top_k]
    )

    return CutoffLanguageProfile(
        person_id=evidence.person_id,
        cutoff=cutoff.isoformat(),
        message_count=len(messages),
        effective_message_weight=round(total_weight, 4),
        weighted_mean_char_length=(
            round(mean_length, 3) if mean_length is not None else None
        ),
        weighted_short_message_ratio=(
            round(short_ratio, 4) if short_ratio is not None else None
        ),
        weighted_laugh_expression_ratio=(
            round(laugh_ratio, 4) if laugh_ratio is not None else None
        ),
        weighted_cry_expression_ratio=(
            round(cry_ratio, 4) if cry_ratio is not None else None
        ),
        frequent_tokens=frequent_tokens,
        platform_message_counts=dict(platform_counts),
    )
