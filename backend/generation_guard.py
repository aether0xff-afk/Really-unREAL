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
    """Detect both overall near-copy and a historical phrase embedded in output."""

    left_set = _ngrams(left)
    right_set = _ngrams(right)
    if not left_set and not right_set:
        return 1.0
    if not left_set or not right_set:
        return 0.0
    overlap = len(left_set & right_set)
    jaccard = overlap / len(left_set | right_set)
    containment = overlap / min(len(left_set), len(right_set))
    return max(jaccard, containment)


def historical_style_references(
    packet: GenerationContextPacket,
    *,
    min_reference_chars: int = 8,
) -> tuple[str, ...]:
    references: list[str] = []
    for example in packet.retrieved_examples:
        for text in example.response_texts:
            if len(_normalize(text)) >= min_reference_chars:
                references.append(text)
    return tuple(references)


def max_historical_copy_similarity(
    burst: GeneratedBurst,
    references: Iterable[str],
) -> float:
    generated = "\n".join(burst.messages)
    values = [text_copy_similarity(generated, reference) for reference in references]
    return max(values, default=0.0)


def _without_raw_exemplars(packet: GenerationContextPacket) -> GenerationContextPacket:
    examples = tuple(
        replace(example, response_texts=()) for example in packet.retrieved_examples
    )
    return replace(
        packet,
        retrieved_examples=examples,
        generation_directives=packet.generation_directives
        + (
            "Historical response wording has been removed for this attempt. "
            "Use only the style fingerprint, burst profile and current visible context; "
            "compose the response independently.",
        ),
    )


class GuardedBurstLanguageModel:
    """Regenerate suspicious exemplar reuse and fall back without raw wording."""

    def __init__(
        self,
        base_model: BurstLanguageModel,
        *,
        max_attempts: int = 3,
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
        stripped = False
        for attempt in range(self.max_attempts):
            burst = self.base_model.generate_burst(current_packet)
            similarity = max_historical_copy_similarity(burst, references)
            candidates.append((similarity, burst))
            if similarity < self.copy_threshold:
                return burst

            if attempt + 1 >= self.max_attempts:
                break
            if attempt + 2 == self.max_attempts:
                # Final attempt removes the raw historical wording entirely, so
                # the guard is not merely asking the same prompt to "try harder".
                current_packet = _without_raw_exemplars(packet)
                stripped = True
            else:
                current_packet = replace(
                    packet,
                    generation_directives=packet.generation_directives
                    + (
                        "The previous candidate reused too much historical wording. "
                        "Answer the CURRENT visible message again in this person's style "
                        "with independently composed phrasing.",
                    ),
                )

        # A provider can still independently emit the same phrase after exemplars
        # are removed. Preserve the scheduled behavior but return the least-copying
        # candidate rather than silently deleting the event.
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]
