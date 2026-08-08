from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime

from backend.fusion import EvidenceMessage, PersonEvidence


_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣ㅋㅎㅠㅜ]+")
_LAUGH_RE = re.compile(r"ㅋ{2,}|ㅎ{2,}")
_CRY_RE = re.compile(r"ㅠ{2,}|ㅜ{2,}")
_TERMINAL_PUNCTUATION = ".?!~…"


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
    weighted_question_ratio: float | None = None
    weighted_exclamation_ratio: float | None = None
    weighted_no_terminal_punctuation_ratio: float | None = None
    weighted_multiline_ratio: float | None = None
    frequent_endings: tuple[tuple[str, float], ...] = ()
    profile_scope: str = "global"
    focused_message_count: int = 0
    focus_weight_multiplier: float = 1.0

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["frequent_tokens"] = [list(item) for item in self.frequent_tokens]
        data["frequent_endings"] = [list(item) for item in self.frequent_endings]
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


def _ending_fragment(text: str) -> str:
    compact = re.sub(r"\s+", " ", text.strip())
    if not compact:
        return ""
    # Keep punctuation because whether a person habitually ends with ?, ~, etc.
    # is itself observable style. Two characters avoid exposing whole long
    # historical responses as persona examples.
    return compact[-2:]


def build_cutoff_language_profile(
    evidence: PersonEvidence,
    cutoff: datetime,
    *,
    top_k: int = 12,
    ending_top_k: int = 8,
    focus_conversation_id: str | None = None,
    focus_platform: str | None = None,
    focus_weight_multiplier: float = 2.0,
) -> CutoffLanguageProfile:
    """Build a leakage-safe style profile with optional relationship focus.

    SELF_TWIN needs this distinction most: the same user can write very
    differently to different people. The profile therefore keeps all older
    target messages as a global fallback while giving messages from the current
    conversation an additional weight. This avoids a brittle hard split when a
    relationship has only a few historical messages.

    The focus multiplier is a baseline hyperparameter, not a psychological
    score. Historical Replay should tune it if relationship-conditioned replay
    shows that another value generalizes better.
    """

    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if ending_top_k < 1:
        raise ValueError("ending_top_k must be >= 1")
    if focus_weight_multiplier < 1.0:
        raise ValueError("focus_weight_multiplier must be >= 1")

    messages = target_messages_before(evidence, cutoff)
    platform_counts: defaultdict[str, int] = defaultdict(int)
    token_weights: defaultdict[str, float] = defaultdict(float)
    ending_weights: defaultdict[str, float] = defaultdict(float)

    weighted_length = 0.0
    short_weight = 0.0
    laugh_weight = 0.0
    cry_weight = 0.0
    question_weight = 0.0
    exclamation_weight = 0.0
    no_terminal_punctuation_weight = 0.0
    multiline_weight = 0.0
    total_weight = 0.0
    focused_message_count = 0

    for item in messages:
        is_focus = (
            focus_conversation_id is not None
            and item.conversation_id == focus_conversation_id
            and (focus_platform is None or item.platform == focus_platform)
        )
        weight = max(0.0, float(item.evidence_weight))
        if is_focus:
            weight *= focus_weight_multiplier
            focused_message_count += 1
        total_weight += weight
        platform_counts[item.platform] += 1
        text = item.message.text
        stripped = text.strip()
        weighted_length += len(text) * weight
        short_weight += (len(text) <= 5) * weight
        laugh_weight += bool(_LAUGH_RE.search(text)) * weight
        cry_weight += bool(_CRY_RE.search(text)) * weight
        question_weight += ("?" in text) * weight
        exclamation_weight += ("!" in text) * weight
        no_terminal_punctuation_weight += (
            bool(stripped) and stripped[-1] not in _TERMINAL_PUNCTUATION
        ) * weight
        multiline_weight += ("\n" in text) * weight
        for token in _TOKEN_RE.findall(text.lower()):
            if token.strip():
                token_weights[token] += weight
        ending = _ending_fragment(text)
        if ending:
            ending_weights[ending] += weight

    if total_weight > 0:
        mean_length = weighted_length / total_weight
        short_ratio = short_weight / total_weight
        laugh_ratio = laugh_weight / total_weight
        cry_ratio = cry_weight / total_weight
        question_ratio = question_weight / total_weight
        exclamation_ratio = exclamation_weight / total_weight
        no_terminal_punctuation_ratio = no_terminal_punctuation_weight / total_weight
        multiline_ratio = multiline_weight / total_weight
    else:
        mean_length = None
        short_ratio = None
        laugh_ratio = None
        cry_ratio = None
        question_ratio = None
        exclamation_ratio = None
        no_terminal_punctuation_ratio = None
        multiline_ratio = None

    frequent_tokens = tuple(
        sorted(
            token_weights.items(),
            key=lambda item: (-item[1], item[0]),
        )[:top_k]
    )
    frequent_endings = tuple(
        sorted(
            ending_weights.items(),
            key=lambda item: (-item[1], item[0]),
        )[:ending_top_k]
    )

    def rounded(value: float | None, digits: int = 4) -> float | None:
        return round(value, digits) if value is not None else None

    return CutoffLanguageProfile(
        person_id=evidence.person_id,
        cutoff=cutoff.isoformat(),
        message_count=len(messages),
        effective_message_weight=round(total_weight, 4),
        weighted_mean_char_length=rounded(mean_length, 3),
        weighted_short_message_ratio=rounded(short_ratio),
        weighted_laugh_expression_ratio=rounded(laugh_ratio),
        weighted_cry_expression_ratio=rounded(cry_ratio),
        frequent_tokens=frequent_tokens,
        platform_message_counts=dict(platform_counts),
        weighted_question_ratio=rounded(question_ratio),
        weighted_exclamation_ratio=rounded(exclamation_ratio),
        weighted_no_terminal_punctuation_ratio=rounded(no_terminal_punctuation_ratio),
        weighted_multiline_ratio=rounded(multiline_ratio),
        frequent_endings=frequent_endings,
        profile_scope=(
            "relationship_blend" if focused_message_count else "global"
        ),
        focused_message_count=focused_message_count,
        focus_weight_multiplier=(
            focus_weight_multiplier if focused_message_count else 1.0
        ),
    )
