from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Sequence

from backend.fusion import PersonEvidence
from backend.persona.cutoff import CutoffLanguageProfile, build_cutoff_language_profile
from backend.replay import ReplayCase
from backend.retrieval import CutoffExampleIndex, RetrievedExample
from backend.simulation.action_policy import Action


_LAUGH_RE = re.compile(r"ㅋ{2,}|ㅎ{2,}")
_CRY_RE = re.compile(r"ㅠ{2,}|ㅜ{2,}")


@dataclass(frozen=True, slots=True)
class VisibleGenerationMessage:
    timestamp: str
    sender_person_id: str | None
    text: str
    platform: str


@dataclass(frozen=True, slots=True)
class RetrievedResponseShape:
    message_lengths: tuple[int, ...]
    question_count: int
    laugh_expression_count: int
    cry_expression_count: int
    endings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RetrievedGenerationExample:
    platform: str
    action: str
    context_texts: tuple[str, ...]
    burst_size: int
    retrieval_score: float
    response_shape: RetrievedResponseShape
    # Raw historical responses are withheld by default to prevent nearest-
    # neighbour copying. This field exists only for explicit ablation/debug runs.
    response_texts: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GenerationContextPacket:
    person_id: str
    observation_end: str
    chosen_action: str
    visible_context: tuple[VisibleGenerationMessage, ...]
    language_profile: CutoffLanguageProfile
    retrieved_examples: tuple[RetrievedGenerationExample, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _ending(text: str) -> str:
    stripped = re.sub(r"\s+", " ", text.strip())
    return stripped[-2:] if stripped else ""


def _response_shape(texts: Sequence[str]) -> RetrievedResponseShape:
    return RetrievedResponseShape(
        message_lengths=tuple(len(text) for text in texts),
        question_count=sum("?" in text for text in texts),
        laugh_expression_count=sum(bool(_LAUGH_RE.search(text)) for text in texts),
        cry_expression_count=sum(bool(_CRY_RE.search(text)) for text in texts),
        endings=tuple(ending for text in texts if (ending := _ending(text))),
    )


def _generation_examples(
    retrieved: Sequence[RetrievedExample],
    *,
    raw_response_examples: int = 0,
) -> tuple[RetrievedGenerationExample, ...]:
    if raw_response_examples < 0:
        raise ValueError("raw_response_examples must be >= 0")

    output: list[RetrievedGenerationExample] = []
    for index, item in enumerate(retrieved):
        raw = item.example.target_texts if index < raw_response_examples else ()
        output.append(
            RetrievedGenerationExample(
                platform=item.example.platform,
                action=item.example.action.value,
                context_texts=item.example.context_texts,
                burst_size=item.example.burst_size,
                retrieval_score=round(item.score, 6),
                response_shape=_response_shape(item.example.target_texts),
                response_texts=raw,
            )
        )
    return tuple(output)


def build_generation_context(
    case: ReplayCase,
    evidence: PersonEvidence,
    index: CutoffExampleIndex,
    *,
    chosen_action: Action,
    retrieval_k: int = 5,
    visible_context_messages: int = 12,
    raw_response_examples: int = 0,
) -> GenerationContextPacket:
    """Build everything a future language model may see for one action.

    The caller must supply ``chosen_action`` from the temporal policy. The real
    held-out ``case.action`` and ``case.target_burst`` are intentionally never
    read here, keeping temporal choice and language generation separated.
    Persona statistics and RAG examples are both cut off strictly before the
    replay observation time.

    Historical *response text* is withheld by default. Retrieval exposes the
    old context plus response shape/style statistics, which gives the model
    behavioural evidence without handing it a sentence to copy. Set
    ``raw_response_examples`` only for an explicit ablation.
    """

    if evidence.person_id != case.person_id:
        raise ValueError("case and evidence refer to different people")
    if visible_context_messages < 1:
        raise ValueError("visible_context_messages must be >= 1")
    if raw_response_examples < 0:
        raise ValueError("raw_response_examples must be >= 0")

    visible = tuple(
        VisibleGenerationMessage(
            timestamp=item.message.timestamp.isoformat(),
            sender_person_id=item.sender_person_id,
            text=item.message.text,
            platform=item.platform,
        )
        for item in case.context[-visible_context_messages:]
    )
    profile = build_cutoff_language_profile(evidence, case.observation_end)
    retrieved = index.search(
        case,
        cutoff=case.observation_end,
        k=retrieval_k,
        action=chosen_action,
    )

    return GenerationContextPacket(
        person_id=case.person_id,
        observation_end=case.observation_end.isoformat(),
        chosen_action=chosen_action.value,
        visible_context=visible,
        language_profile=profile,
        retrieved_examples=_generation_examples(
            retrieved,
            raw_response_examples=raw_response_examples,
        ),
    )
