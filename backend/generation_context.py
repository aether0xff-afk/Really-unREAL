from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from backend.fusion import PersonEvidence
from backend.persona.cutoff import CutoffLanguageProfile, build_cutoff_language_profile
from backend.replay import ReplayCase
from backend.retrieval import CutoffExampleIndex, RetrievedExample
from backend.simulation.action_policy import Action


@dataclass(frozen=True, slots=True)
class VisibleGenerationMessage:
    timestamp: str
    sender_person_id: str | None
    text: str
    platform: str


@dataclass(frozen=True, slots=True)
class RetrievedGenerationExample:
    platform: str
    action: str
    context_texts: tuple[str, ...]
    response_texts: tuple[str, ...]
    burst_size: int
    retrieval_score: float


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


def _generation_examples(
    retrieved: Sequence[RetrievedExample],
) -> tuple[RetrievedGenerationExample, ...]:
    return tuple(
        RetrievedGenerationExample(
            platform=item.example.platform,
            action=item.example.action.value,
            context_texts=item.example.context_texts,
            response_texts=item.example.target_texts,
            burst_size=item.example.burst_size,
            retrieval_score=round(item.score, 6),
        )
        for item in retrieved
    )


def build_generation_context(
    case: ReplayCase,
    evidence: PersonEvidence,
    index: CutoffExampleIndex,
    *,
    chosen_action: Action,
    retrieval_k: int = 5,
    visible_context_messages: int = 12,
) -> GenerationContextPacket:
    """Build everything a future language model may see for one action.

    The caller must supply ``chosen_action`` from the temporal policy. The real
    held-out ``case.action`` and ``case.target_burst`` are intentionally never
    read here, keeping temporal choice and language generation separated.
    Persona statistics and RAG examples are both cut off strictly before the
    replay observation time.
    """

    if evidence.person_id != case.person_id:
        raise ValueError("case and evidence refer to different people")
    if visible_context_messages < 1:
        raise ValueError("visible_context_messages must be >= 1")

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
    )

    return GenerationContextPacket(
        person_id=case.person_id,
        observation_end=case.observation_end.isoformat(),
        chosen_action=chosen_action.value,
        visible_context=visible,
        language_profile=profile,
        retrieved_examples=_generation_examples(retrieved),
    )
