from __future__ import annotations

import re
from dataclasses import replace
from typing import Iterable

from backend.generation import BurstLanguageModel, GeneratedBurst
from backend.generation_context import GenerationContextPacket


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def _ngrams(text: str, n: int = 3) -> set[str]:
    text = _normalize(text)
    if not text:
        return set()
    if len(text) <= n:
        return {text}
    return {text[index : index + n] for index in range(len(text) - n + 1)}


def text_copy_similarity(left: str, right: str) -> float:
    """Character 3-gram Jaccard used only as an anti-copy safety heuristic."""

    left_set = _ngrams(left)
    right_set = _ngrams(right)
    if not left_set and not right_set:
        return 1.0
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def historical_style_references(
    packet: GenerationContextPacket,
    *,
    min_reference_chars: int = 8,
) -> tuple[str, ...]:
    references: list[str] = []
    for example in packet.retrieved_examples:
        for text in example.response_texts:
            normalized = _normalize(text)
            if len(normalized) >= min_reference_chars:
                references.append(text)
    return tuple(references)


def max_historical_copy_similarity(
    burst: GeneratedBurst,
    references: Iterable[str],
) -> float:
    generated = "\n".join(burst.messages)
    values = [text_copy_similarity(generated, reference) for reference in references]
    return max(values, default=0.0)


class GuardedBurstLanguageModel:
    """Retry suspicious long exemplar copies while preserving ordinary phrasing.

    Very short historical replies are deliberately excluded from copy detection:
    expressions such as `ㅇㅇ`, `ㄴㄴ`, `오케`, or `ㅋㅋ` are legitimate recurring
    observable habits and should not be treated as memorization. Longer style
    exemplars may be shown to the provider, but near-verbatim reproduction causes
    a retry with an explicit rewrite directive. If every candidate remains close,
    the least-copying candidate is returned instead of erasing the scheduled
    behavior.
    """

    def __init__(
        self,
        base_model: BurstLanguageModel,
        *,
        max_attempts: int = 2,
        copy_threshold: float = 0.82,
        min_reference_chars: int = 8,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if not 0.0 <= copy_threshold <= 1.0:
            raise ValueError("copy_threshold must be between 0 and 1")
        if min_reference_chars < 1:
            raise ValueError("min_reference_chars must be >= 1")
        self.base_model = base_model
        self.max_attempts = int(max_attempts)
        self.copy_threshold = float(copy_threshold)
        self.min_reference_chars = int(min_reference_chars)

    @property
    def model(self) -> str:
        return str(getattr(self.base_model, "model", type(self.base_model).__name__))

    def generate_burst(self, packet: GenerationContextPacket) -> GeneratedBurst:
        references = historical_style_references(
            packet,
            min_reference_chars=self.min_reference_chars,
        )
        if not references:
            return self.base_model.generate_burst(packet)

        candidates: list[tuple[float, GeneratedBurst]] = []
        current_packet = packet
        for attempt in range(self.max_attempts):
            burst = self.base_model.generate_burst(current_packet)
            similarity = max_historical_copy_similarity(burst, references)
            candidates.append((similarity, burst))
            if similarity < self.copy_threshold:
                return burst

            if attempt + 1 < self.max_attempts:
                current_packet = replace(
                    packet,
                    generation_directives=packet.generation_directives
                    + (
                        "The previous candidate was too close to an older style exemplar. "
                        "Answer the CURRENT visible message again in this person's style, "
                        "but use independently composed wording and do not reuse a long "
                        "historical phrase.",
                    ),
                )

        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]
