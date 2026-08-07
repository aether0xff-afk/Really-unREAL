from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Iterable

from backend.identity import IdentityMap
from backend.ingest.archive import ConversationExport
from backend.ingest.instagram import InstagramThread
from backend.models import ChatMessage


class EvidenceContext(StrEnum):
    KAKAO_DIRECT = "kakao_direct"
    KAKAO_GROUP = "kakao_group"
    INSTAGRAM_DIRECT = "instagram_direct"
    INSTAGRAM_GROUP = "instagram_group"


@dataclass(frozen=True, slots=True)
class EvidenceWeights:
    kakao_direct: float = 1.0
    instagram_direct: float = 1.0
    kakao_group: float = 0.35
    instagram_group: float = 0.45

    def for_context(self, context: EvidenceContext) -> float:
        return {
            EvidenceContext.KAKAO_DIRECT: self.kakao_direct,
            EvidenceContext.KAKAO_GROUP: self.kakao_group,
            EvidenceContext.INSTAGRAM_DIRECT: self.instagram_direct,
            EvidenceContext.INSTAGRAM_GROUP: self.instagram_group,
        }[context]


@dataclass(frozen=True, slots=True)
class EvidenceMessage:
    message: ChatMessage
    platform: str
    conversation_id: str
    context: EvidenceContext
    sender_person_id: str | None
    evidence_weight: float


@dataclass(frozen=True, slots=True)
class EvidenceConversation:
    platform: str
    conversation_id: str
    context: EvidenceContext
    messages: tuple[EvidenceMessage, ...]


@dataclass(frozen=True, slots=True)
class PersonEvidence:
    person_id: str
    conversations: tuple[EvidenceConversation, ...]

    def target_messages(self) -> tuple[EvidenceMessage, ...]:
        return tuple(
            evidence
            for conversation in self.conversations
            for evidence in conversation.messages
            if evidence.sender_person_id == self.person_id
        )

    def counts_by_context(self) -> dict[str, int]:
        counts = {context.value: 0 for context in EvidenceContext}
        for message in self.target_messages():
            counts[message.context.value] += 1
        return counts


def _resolved_participants(
    platform: str,
    aliases: Iterable[str],
    identity_map: IdentityMap,
) -> set[str]:
    return {
        person_id
        for alias in aliases
        if (person_id := identity_map.resolve(platform, alias)) is not None
    }


def _context(platform: str, participant_count: int) -> EvidenceContext:
    if platform == "kakao":
        return (
            EvidenceContext.KAKAO_DIRECT
            if participant_count == 2
            else EvidenceContext.KAKAO_GROUP
        )
    return (
        EvidenceContext.INSTAGRAM_DIRECT
        if participant_count == 2
        else EvidenceContext.INSTAGRAM_GROUP
    )


def _wrap_messages(
    *,
    platform: str,
    conversation_id: str,
    context: EvidenceContext,
    messages: Iterable[ChatMessage],
    identity_map: IdentityMap,
    weights: EvidenceWeights,
) -> tuple[EvidenceMessage, ...]:
    weight = weights.for_context(context)
    return tuple(
        EvidenceMessage(
            message=message,
            platform=platform,
            conversation_id=conversation_id,
            context=context,
            sender_person_id=identity_map.resolve(platform, message.sender),
            evidence_weight=weight,
        )
        for message in messages
    )


def collect_person_evidence(
    person_id: str,
    identity_map: IdentityMap,
    *,
    kakao_conversations: Iterable[ConversationExport] = (),
    instagram_threads: Iterable[InstagramThread] = (),
    weights: EvidenceWeights | None = None,
) -> PersonEvidence:
    """Collect source-aware evidence for one person without flattening context.

    A conversation is included only when the explicit identity map resolves the
    target as one of its participants. The full conversation is retained for
    retrieval/context, but only messages whose sender resolves to ``person_id``
    count as target persona evidence.
    """

    identity_map.get(person_id)  # validate early
    weights = weights or EvidenceWeights()
    conversations: list[EvidenceConversation] = []

    for conversation in kakao_conversations:
        participant_aliases = conversation.participants
        participant_ids = _resolved_participants("kakao", participant_aliases, identity_map)
        if person_id not in participant_ids:
            continue
        context = _context("kakao", len(participant_aliases))
        conversations.append(
            EvidenceConversation(
                platform="kakao",
                conversation_id=conversation.source_archive,
                context=context,
                messages=_wrap_messages(
                    platform="kakao",
                    conversation_id=conversation.source_archive,
                    context=context,
                    messages=conversation.messages,
                    identity_map=identity_map,
                    weights=weights,
                ),
            )
        )

    for thread in instagram_threads:
        participant_ids = _resolved_participants("instagram", thread.participants, identity_map)
        if person_id not in participant_ids:
            continue
        context = _context("instagram", len(thread.participants))
        conversations.append(
            EvidenceConversation(
                platform="instagram",
                conversation_id=thread.thread_id,
                context=context,
                messages=_wrap_messages(
                    platform="instagram",
                    conversation_id=thread.thread_id,
                    context=context,
                    messages=thread.messages,
                    identity_map=identity_map,
                    weights=weights,
                ),
            )
        )

    conversations.sort(
        key=lambda conversation: (
            conversation.messages[0].message.timestamp
            if conversation.messages
            else datetime.max
        )
    )
    return PersonEvidence(person_id=person_id, conversations=tuple(conversations))


def canonical_target_messages(evidence: PersonEvidence) -> list[ChatMessage]:
    """Return target utterances with a stable sender ID and source metadata.

    This is convenient for the existing profile code, while the original sender
    name and source context remain available in metadata. Evidence weights are
    annotations only; callers must not duplicate messages to simulate weights.
    """

    output: list[ChatMessage] = []
    for item in evidence.target_messages():
        metadata = dict(item.message.metadata)
        metadata.update(
            {
                "original_sender": item.message.sender,
                "platform": item.platform,
                "conversation_id": item.conversation_id,
                "evidence_context": item.context.value,
                "evidence_weight": item.evidence_weight,
            }
        )
        output.append(
            ChatMessage(
                timestamp=item.message.timestamp,
                sender=evidence.person_id,
                text=item.message.text,
                source=item.message.source,
                message_type=item.message.message_type,
                metadata=metadata,
            )
        )
    output.sort(key=lambda message: message.timestamp)
    return output
