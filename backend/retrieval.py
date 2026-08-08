from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Protocol, Sequence

from backend.fusion import EvidenceContext
from backend.replay import ReplayCase
from backend.simulation.action_policy import Action


_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣ㅋㅎㅠㅜ]+")


class EmbeddingProvider(Protocol):
    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        ...


@dataclass(frozen=True, slots=True)
class HistoricalExample:
    case_id: str
    action_at: datetime
    platform: str
    conversation_id: str
    evidence_context: EvidenceContext
    evidence_weight: float
    action: Action
    context_texts: tuple[str, ...]
    target_texts: tuple[str, ...]
    burst_size: int
    action_is_ambiguous: bool = False


@dataclass(frozen=True, slots=True)
class RetrievedExample:
    example: HistoricalExample
    score: float
    # Hybrid semantic score when an embedding provider is configured; lexical
    # fallback otherwise. Kept under the old field name for caller compatibility.
    semantic_similarity: float
    recency_score: float
    lexical_similarity: float = 0.0
    embedding_similarity: float | None = None


def historical_examples_from_replay(
    cases: Iterable[ReplayCase],
    *,
    context_messages: int = 6,
) -> list[HistoricalExample]:
    if context_messages < 1:
        raise ValueError("context_messages must be >= 1")
    examples = [
        HistoricalExample(
            case_id=case.case_id,
            action_at=case.action_at,
            platform=case.platform,
            conversation_id=case.conversation_id,
            evidence_context=case.evidence_context,
            evidence_weight=case.evidence_weight,
            action=case.action,
            context_texts=tuple(
                message.message.text
                for message in case.context[-context_messages:]
                if message.message.text
            ),
            target_texts=tuple(
                message.message.text
                for message in case.target_burst
                if message.message.text
            ),
            burst_size=case.burst_size,
            action_is_ambiguous=case.action_is_ambiguous,
        )
        for case in cases
    ]
    examples.sort(key=lambda example: (example.action_at, example.case_id))
    return examples


def _normalized_text(texts: Sequence[str]) -> str:
    tokens = _TOKEN_RE.findall(" ".join(texts).lower())
    return " ".join(tokens)


def _word_jaccard(left: str, right: str) -> float:
    left_tokens = set(_TOKEN_RE.findall(left))
    right_tokens = set(_TOKEN_RE.findall(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _char_ngrams(text: str, n: int = 2) -> Counter[str]:
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return Counter()
    if len(compact) < n:
        return Counter({compact: 1})
    return Counter(compact[index : index + n] for index in range(len(compact) - n + 1))


def _counter_cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(key, 0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _vector_cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding vectors must have the same dimension")
    if not left:
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))


def _lexical_similarity(query_text: str, candidate_text: str) -> float:
    """Cheap local similarity fallback, deliberately not called an embedding."""

    char_similarity = _counter_cosine(
        _char_ngrams(query_text),
        _char_ngrams(candidate_text),
    )
    token_similarity = _word_jaccard(query_text, candidate_text)
    return 0.65 * char_similarity + 0.35 * token_similarity


def _recency_score(candidate_at: datetime, cutoff: datetime) -> float:
    age_days = max(0.0, (cutoff - candidate_at).total_seconds() / 86400.0)
    return 1.0 / (1.0 + age_days / 30.0)


class CutoffExampleIndex:
    """Historical examples with a hard temporal retrieval cutoff.

    The index may be constructed from the full replay corpus for convenience,
    but ``search`` only considers examples whose real target action occurred
    strictly before the requested cutoff.

    With no embedding provider, retrieval remains dependency-free and lexical.
    When a provider is configured, context-only dense embeddings are precomputed
    and blended with lexical similarity. Target responses are never embedded for
    ranking, preserving the anti-leakage contract.
    """

    def __init__(
        self,
        examples: Iterable[HistoricalExample],
        *,
        embedding_provider: EmbeddingProvider | None = None,
        embedding_weight: float = 0.70,
    ) -> None:
        if not 0.0 <= embedding_weight <= 1.0:
            raise ValueError("embedding_weight must be between 0 and 1")
        self.examples = tuple(
            sorted(examples, key=lambda example: (example.action_at, example.case_id))
        )
        self.embedding_provider = embedding_provider
        self.embedding_weight = float(embedding_weight)
        self._embeddings: dict[str, tuple[float, ...]] = {}

        if embedding_provider is not None and self.examples:
            texts = [_normalized_text(example.context_texts) for example in self.examples]
            vectors = list(embedding_provider.embed(texts))
            if len(vectors) != len(self.examples):
                raise ValueError("embedding provider returned unexpected vector count")
            dimension: int | None = None
            for example, vector in zip(self.examples, vectors):
                values = tuple(float(value) for value in vector)
                if not values:
                    raise ValueError("embedding vectors must not be empty")
                if dimension is None:
                    dimension = len(values)
                elif len(values) != dimension:
                    raise ValueError("embedding vectors must have consistent dimensions")
                self._embeddings[example.case_id] = values

    @classmethod
    def from_replay_cases(
        cls,
        cases: Iterable[ReplayCase],
        *,
        context_messages: int = 6,
        embedding_provider: EmbeddingProvider | None = None,
        embedding_weight: float = 0.70,
    ) -> "CutoffExampleIndex":
        return cls(
            historical_examples_from_replay(
                cases,
                context_messages=context_messages,
            ),
            embedding_provider=embedding_provider,
            embedding_weight=embedding_weight,
        )

    def search(
        self,
        case: ReplayCase,
        *,
        cutoff: datetime | None = None,
        k: int = 5,
        context_messages: int = 6,
        minimum_score: float = 0.0,
        action: Action | None = None,
    ) -> list[RetrievedExample]:
        if k < 1:
            return []
        if context_messages < 1:
            raise ValueError("context_messages must be >= 1")
        cutoff = cutoff or case.observation_end

        query_text = _normalized_text(
            tuple(
                message.message.text
                for message in case.context[-context_messages:]
                if message.message.text
            )
        )
        query_embedding: tuple[float, ...] | None = None
        if self.embedding_provider is not None:
            vectors = list(self.embedding_provider.embed([query_text]))
            if len(vectors) != 1:
                raise ValueError("embedding provider must return one query vector")
            query_embedding = tuple(float(value) for value in vectors[0])

        visible_timestamps = {
            message.message.timestamp
            for message in case.context
            if message.conversation_id == case.conversation_id
        }

        results: list[RetrievedExample] = []
        for example in self.examples:
            if example.case_id == case.case_id:
                continue
            # Strictly earlier is intentional. With minute-resolution Kakao
            # timestamps, an example stamped at the same minute as the cutoff
            # cannot safely be assumed to have happened first.
            if example.action_at >= cutoff:
                continue
            if action is not None:
                # Long-gap examples cannot safely tell us REPLY vs INITIATE, so
                # they are excluded when the caller requests an action-specific
                # retrieval bucket.
                if example.action_is_ambiguous or example.action != action:
                    continue
            if (
                example.conversation_id == case.conversation_id
                and example.action_at in visible_timestamps
            ):
                # The response is already present in the visible live context;
                # returning it again as a historical example adds no information.
                continue

            candidate_text = _normalized_text(example.context_texts)
            lexical = _lexical_similarity(query_text, candidate_text)
            dense: float | None = None
            semantic = lexical
            if query_embedding is not None:
                dense = _vector_cosine(
                    query_embedding,
                    self._embeddings[example.case_id],
                )
                semantic = (
                    self.embedding_weight * dense
                    + (1.0 - self.embedding_weight) * lexical
                )

            recency = _recency_score(example.action_at, cutoff)
            source_weight = max(0.0, float(example.evidence_weight))
            score = source_weight * (0.85 * semantic + 0.15 * recency)
            if example.platform == case.platform:
                score *= 1.05
            if score < minimum_score:
                continue
            results.append(
                RetrievedExample(
                    example=example,
                    score=score,
                    semantic_similarity=semantic,
                    recency_score=recency,
                    lexical_similarity=lexical,
                    embedding_similarity=dense,
                )
            )

        results.sort(
            key=lambda result: (
                -result.score,
                -result.example.action_at.timestamp(),
                result.example.case_id,
            )
        )
        return results[:k]
