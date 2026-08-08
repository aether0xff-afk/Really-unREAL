from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime

from backend.fusion import PersonEvidence


_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+")
_STOPWORDS = {
    "ㅋㅋ",
    "ㅋㅋㅋ",
    "ㅎㅎ",
    "ㅠㅠ",
    "ㅜㅜ",
    "ㅇㅇ",
    "ㄴㄴ",
    "아",
    "어",
    "응",
    "엉",
    "근데",
    "그냥",
    "진짜",
    "너",
    "나",
    "내",
    "네",
    "뭐",
    "왜",
    "이거",
    "그거",
    "저거",
}


@dataclass(frozen=True, slots=True)
class ObservableTopicCue:
    """A topic-like token supported only by observable past messages."""

    token: str
    score: float
    mention_count: int
    focused_mention_count: int
    last_seen_at: str


@dataclass(frozen=True, slots=True)
class TopicMemorySnapshot:
    cutoff: str
    horizon_days: float
    cues: tuple[ObservableTopicCue, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _tokens(text: str) -> tuple[str, ...]:
    output: list[str] = []
    for token in _TOKEN_RE.findall(text.lower()):
        token = token.strip()
        if not token or token in _STOPWORDS:
            continue
        # One-character Korean chat particles/interjections are too noisy. Keep
        # one-character digits/ASCII only when they carry explicit information.
        if len(token) == 1 and not token.isascii():
            continue
        output.append(token)
    return tuple(output)


def build_topic_memory(
    evidence: PersonEvidence,
    cutoff: datetime,
    *,
    focus_conversation_id: str | None = None,
    focus_platform: str | None = None,
    horizon_days: float = 120.0,
    focus_weight_multiplier: float = 3.0,
    top_k: int = 8,
) -> TopicMemorySnapshot:
    """Build long-term topic cues without inferring hidden interests.

    Only messages strictly before ``cutoff`` are considered. Scores combine
    source evidence weight, recency, and a relationship focus boost. The output
    contains token-level cues rather than raw historical sentences, so the LLM
    gets continuity hints without receiving another answer to copy.
    """

    if horizon_days <= 0:
        raise ValueError("horizon_days must be > 0")
    if focus_weight_multiplier < 1.0:
        raise ValueError("focus_weight_multiplier must be >= 1")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    scores: defaultdict[str, float] = defaultdict(float)
    mentions: defaultdict[str, int] = defaultdict(int)
    focused_mentions: defaultdict[str, int] = defaultdict(int)
    last_seen: dict[str, datetime] = {}

    for conversation in evidence.conversations:
        is_focus_conversation = (
            focus_conversation_id is not None
            and conversation.conversation_id == focus_conversation_id
            and (focus_platform is None or conversation.platform == focus_platform)
        )
        for item in conversation.messages:
            timestamp = item.message.timestamp
            if timestamp >= cutoff:
                continue
            age_days = max(0.0, (cutoff - timestamp).total_seconds() / 86400.0)
            if age_days > horizon_days:
                continue

            recency = 1.0 / (1.0 + age_days / 30.0)
            weight = max(0.0, float(item.evidence_weight)) * recency
            if is_focus_conversation:
                weight *= focus_weight_multiplier
            # Target-authored and counterpart-authored text are both observable
            # evidence of what the relationship actually talked about. We do not
            # convert either into a hidden preference or interest score.
            for token in set(_tokens(item.message.text)):
                scores[token] += weight
                mentions[token] += 1
                if is_focus_conversation:
                    focused_mentions[token] += 1
                if token not in last_seen or timestamp > last_seen[token]:
                    last_seen[token] = timestamp

    ordered = sorted(
        scores,
        key=lambda token: (
            -scores[token],
            -last_seen[token].timestamp(),
            token,
        ),
    )[:top_k]

    cues = tuple(
        ObservableTopicCue(
            token=token,
            score=round(scores[token], 6),
            mention_count=mentions[token],
            focused_mention_count=focused_mentions[token],
            last_seen_at=last_seen[token].isoformat(),
        )
        for token in ordered
    )
    return TopicMemorySnapshot(
        cutoff=cutoff.isoformat(),
        horizon_days=float(horizon_days),
        cues=cues,
    )
