from __future__ import annotations

import re
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Iterable

from backend.fusion import PersonEvidence
from backend.retrieval import HistoricalExample
from backend.simulation.action_policy import Action


_JAMO_RUN_RE = re.compile(r"[ㄱ-ㅎㅏ-ㅣ]{2,}")
_ELLIPSIS_RE = re.compile(r"\.{2,}|…+")
_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣ㅋㅎㅠㅜ]+")


@dataclass(frozen=True, slots=True)
class StyleFingerprint:
    message_count: int
    focused_message_count: int
    median_char_length: float | None
    p25_char_length: float | None
    p75_char_length: float | None
    space_presence_ratio: float | None
    jamo_run_ratio: float | None
    ellipsis_ratio: float | None
    repeated_question_ratio: float | None
    repeated_exclamation_ratio: float | None
    frequent_first_tokens: tuple[tuple[str, float], ...]
    frequent_punctuation_shapes: tuple[tuple[str, float], ...]

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["frequent_first_tokens"] = [list(item) for item in self.frequent_first_tokens]
        data["frequent_punctuation_shapes"] = [list(item) for item in self.frequent_punctuation_shapes]
        return data


@dataclass(frozen=True, slots=True)
class BurstBehaviorProfile:
    event_count: int
    focused_event_count: int
    action_event_count: int
    weighted_mean_burst_size: float | None
    single_message_burst_ratio: float | None
    weighted_mean_total_chars: float | None
    short_total_burst_ratio: float | None
    burst_size_histogram: tuple[tuple[int, float], ...]

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["burst_size_histogram"] = [list(item) for item in self.burst_size_histogram]
        return data


def _weighted_quantile(values: list[tuple[float, float]], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted((value, max(0.0, weight)) for value, weight in values)
    total = sum(weight for _, weight in ordered)
    if total <= 0:
        ordered = [(value, 1.0) for value, _ in ordered]
        total = float(len(ordered))
    target = q * total
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= target:
            return float(value)
    return float(ordered[-1][0])


def _punctuation_shape(text: str) -> str:
    output: list[str] = []
    for char in text.strip():
        if char in "?!~.…":
            output.append(char)
    shape = "".join(output)
    return shape[-4:] if shape else "none"


def build_style_fingerprint(
    evidence: PersonEvidence,
    cutoff: datetime,
    *,
    focus_conversation_id: str | None = None,
    focus_platform: str | None = None,
    focus_multiplier: float = 2.5,
    top_k: int = 6,
) -> StyleFingerprint:
    rows = [item for item in evidence.target_messages() if item.message.timestamp < cutoff]
    lengths: list[tuple[float, float]] = []
    first_tokens: Counter[str] = Counter()
    punctuation: Counter[str] = Counter()
    weighted_space = weighted_jamo = weighted_ellipsis = 0.0
    weighted_repeat_q = weighted_repeat_e = total_weight = 0.0
    focused = 0

    for item in rows:
        is_focus = (
            focus_conversation_id is not None
            and item.conversation_id == focus_conversation_id
            and (focus_platform is None or item.platform == focus_platform)
        )
        weight = max(0.0, float(item.evidence_weight)) * (focus_multiplier if is_focus else 1.0)
        if is_focus:
            focused += 1
        text = item.message.text.strip()
        lengths.append((float(len(text)), weight))
        total_weight += weight
        weighted_space += (" " in text) * weight
        weighted_jamo += bool(_JAMO_RUN_RE.search(text)) * weight
        weighted_ellipsis += bool(_ELLIPSIS_RE.search(text)) * weight
        weighted_repeat_q += ("??" in text) * weight
        weighted_repeat_e += ("!!" in text) * weight
        tokens = _TOKEN_RE.findall(text.lower())
        if tokens:
            first_tokens[tokens[0]] += weight
        punctuation[_punctuation_shape(text)] += weight

    ratio = lambda value: round(value / total_weight, 4) if total_weight > 0 else None
    return StyleFingerprint(
        message_count=len(rows),
        focused_message_count=focused,
        median_char_length=(round(_weighted_quantile(lengths, 0.5), 3) if lengths else None),
        p25_char_length=(round(_weighted_quantile(lengths, 0.25), 3) if lengths else None),
        p75_char_length=(round(_weighted_quantile(lengths, 0.75), 3) if lengths else None),
        space_presence_ratio=ratio(weighted_space),
        jamo_run_ratio=ratio(weighted_jamo),
        ellipsis_ratio=ratio(weighted_ellipsis),
        repeated_question_ratio=ratio(weighted_repeat_q),
        repeated_exclamation_ratio=ratio(weighted_repeat_e),
        frequent_first_tokens=tuple(first_tokens.most_common(top_k)),
        frequent_punctuation_shapes=tuple(punctuation.most_common(top_k)),
    )


def build_burst_behavior_profile(
    examples: Iterable[HistoricalExample],
    cutoff: datetime,
    *,
    focus_conversation_id: str | None,
    platform: str | None,
    action: Action | None,
    focus_multiplier: float = 2.5,
) -> BurstBehaviorProfile:
    rows = [example for example in examples if example.action_at < cutoff]
    weighted_total = weighted_burst = weighted_single = weighted_chars = weighted_short = 0.0
    histogram: Counter[int] = Counter()
    focused = action_count = 0

    for example in rows:
        is_focus = focus_conversation_id is not None and example.conversation_id == focus_conversation_id
        if platform is not None:
            is_focus = is_focus and example.platform == platform
        weight = max(0.0, float(example.evidence_weight)) * (focus_multiplier if is_focus else 1.0)
        if is_focus:
            focused += 1
        if action is not None and not example.action_is_ambiguous and example.action == action:
            weight *= 1.35
            action_count += 1
        total_chars = sum(len(text) for text in example.target_texts)
        weighted_total += weight
        weighted_burst += example.burst_size * weight
        weighted_single += (example.burst_size == 1) * weight
        weighted_chars += total_chars * weight
        weighted_short += (total_chars <= 8) * weight
        histogram[example.burst_size] += weight

    if weighted_total <= 0:
        mean_burst = single_ratio = mean_chars = short_ratio = None
    else:
        mean_burst = round(weighted_burst / weighted_total, 3)
        single_ratio = round(weighted_single / weighted_total, 4)
        mean_chars = round(weighted_chars / weighted_total, 3)
        short_ratio = round(weighted_short / weighted_total, 4)

    return BurstBehaviorProfile(
        event_count=len(rows),
        focused_event_count=focused,
        action_event_count=action_count,
        weighted_mean_burst_size=mean_burst,
        single_message_burst_ratio=single_ratio,
        weighted_mean_total_chars=mean_chars,
        short_total_burst_ratio=short_ratio,
        burst_size_histogram=tuple(sorted((size, round(weight, 3)) for size, weight in histogram.items())),
    )
