from __future__ import annotations

from datetime import datetime, timedelta

from backend.fusion import EvidenceContext, EvidenceConversation, EvidenceMessage, PersonEvidence
from backend.generation import GeneratedBurst
from backend.generation_context import (
    GenerationContextPacket,
    RetrievedGenerationExample,
    RetrievedResponseShape,
    VisibleGenerationMessage,
)
from backend.generation_guard import GuardedBurstLanguageModel
from backend.models import ChatMessage
from backend.persona.cutoff import CutoffLanguageProfile
from backend.persona.style_fingerprint import build_burst_behavior_profile, build_style_fingerprint
from backend.retrieval import HistoricalExample
from backend.simulation.action_policy import Action


BASE = datetime(2026, 8, 8, 12, 0)


def _emessage(at: datetime, sender: str, text: str, conversation_id: str) -> EvidenceMessage:
    return EvidenceMessage(
        message=ChatMessage(at, sender, text),
        platform="kakao",
        conversation_id=conversation_id,
        context=EvidenceContext.KAKAO_DIRECT,
        sender_person_id=sender,
        evidence_weight=1.0,
    )


def test_style_fingerprint_is_cutoff_safe_and_relationship_focused() -> None:
    c1 = EvidenceConversation(
        platform="kakao",
        conversation_id="c1",
        context=EvidenceContext.KAKAO_DIRECT,
        messages=(
            _emessage(BASE - timedelta(days=3), "target", "ㅇㅇㅋㅋ", "c1"),
            _emessage(BASE - timedelta(days=2), "target", "아니??", "c1"),
            _emessage(BASE + timedelta(days=1), "target", "FUTURE NEVER USE", "c1"),
        ),
    )
    c2 = EvidenceConversation(
        platform="kakao",
        conversation_id="c2",
        context=EvidenceContext.KAKAO_DIRECT,
        messages=(_emessage(BASE - timedelta(days=1), "target", "안녕하세요...", "c2"),),
    )
    evidence = PersonEvidence("target", (c1, c2))
    profile = build_style_fingerprint(
        evidence,
        BASE,
        focus_conversation_id="c1",
        focus_platform="kakao",
        focus_multiplier=3.0,
    )
    assert profile.message_count == 3
    assert profile.focused_message_count == 2
    assert profile.repeated_question_ratio is not None
    assert profile.repeated_question_ratio > 0
    assert all(token != "future" for token, _ in profile.frequent_first_tokens)


def test_burst_profile_uses_only_examples_before_cutoff() -> None:
    old = HistoricalExample(
        case_id="old",
        action_at=BASE - timedelta(days=1),
        platform="kakao",
        conversation_id="c",
        evidence_context=EvidenceContext.KAKAO_DIRECT,
        evidence_weight=1.0,
        action=Action.REPLY,
        context_texts=("뭐함",),
        target_texts=("집", "ㅋㅋ"),
        burst_size=2,
    )
    future = HistoricalExample(
        case_id="future",
        action_at=BASE + timedelta(days=1),
        platform="kakao",
        conversation_id="c",
        evidence_context=EvidenceContext.KAKAO_DIRECT,
        evidence_weight=1.0,
        action=Action.REPLY,
        context_texts=("뭐함",),
        target_texts=("future",),
        burst_size=1,
    )
    profile = build_burst_behavior_profile(
        [old, future],
        BASE,
        focus_conversation_id="c",
        platform="kakao",
        action=Action.REPLY,
    )
    assert profile.event_count == 1
    assert profile.focused_event_count == 1
    assert profile.weighted_mean_burst_size == 2.0
    assert profile.burst_size_histogram == ((2, profile.burst_size_histogram[0][1]),)


def _packet(reference: str) -> GenerationContextPacket:
    return GenerationContextPacket(
        person_id="target",
        observation_end=BASE.isoformat(),
        chosen_action="REPLY",
        visible_context=(
            VisibleGenerationMessage(
                timestamp=BASE.isoformat(),
                sender_person_id="self",
                text="오늘 뭐해",
                platform="kakao",
            ),
        ),
        language_profile=CutoffLanguageProfile(
            person_id="target",
            cutoff=BASE.isoformat(),
            message_count=20,
            effective_message_weight=20.0,
            weighted_mean_char_length=5.0,
            weighted_short_message_ratio=0.8,
            weighted_laugh_expression_ratio=0.3,
            weighted_cry_expression_ratio=0.0,
            frequent_tokens=(),
            platform_message_counts={"kakao": 20},
        ),
        retrieved_examples=(
            RetrievedGenerationExample(
                platform="kakao",
                action="REPLY",
                context_texts=("예전 질문",),
                burst_size=1,
                retrieval_score=0.9,
                response_shape=RetrievedResponseShape(
                    message_lengths=(len(reference),),
                    question_count=0,
                    laugh_expression_count=0,
                    cry_expression_count=0,
                    endings=(reference[-2:],),
                ),
                response_texts=(reference,),
            ),
        ),
    )


class SequenceModel:
    model = "sequence"

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls: list[GenerationContextPacket] = []

    def generate_burst(self, packet: GenerationContextPacket) -> GeneratedBurst:
        self.calls.append(packet)
        return GeneratedBurst((self.outputs[len(self.calls) - 1],))


def test_copy_guard_retries_long_historical_verbatim_copy() -> None:
    reference = "뚜레쥬르 앞으로 와"
    base = SequenceModel([reference, "그쪽으로 와 ㅋㅋ"])
    model = GuardedBurstLanguageModel(
        base,
        max_attempts=2,
        copy_threshold=0.82,
        min_reference_chars=8,
    )
    result = model.generate_burst(_packet(reference))
    assert result.messages == ("그쪽으로 와 ㅋㅋ",)
    assert len(base.calls) == 2
    assert base.calls[1].generation_directives
    assert "historical" in base.calls[1].generation_directives[-1].lower()


def test_copy_guard_final_attempt_removes_raw_historical_wording() -> None:
    reference = "뚜레쥬르 앞으로 와"
    base = SequenceModel([reference, reference, "완전 다른 문장"])
    model = GuardedBurstLanguageModel(
        base,
        max_attempts=3,
        copy_threshold=0.82,
        min_reference_chars=8,
    )
    result = model.generate_burst(_packet(reference))
    assert result.messages == ("완전 다른 문장",)
    assert len(base.calls) == 3
    assert base.calls[2].retrieved_examples[0].response_texts == ()
    assert "removed" in base.calls[2].generation_directives[-1].lower()


def test_copy_guard_detects_reference_embedded_inside_longer_output() -> None:
    reference = "뚜레쥬르 앞으로 와"
    base = SequenceModel([f"아 그럼 {reference} ㅋㅋ", "그쪽으로 보자"])
    model = GuardedBurstLanguageModel(base, max_attempts=2, copy_threshold=0.82)
    assert model.generate_burst(_packet(reference)).messages == ("그쪽으로 보자",)


def test_copy_guard_allows_common_short_repetition() -> None:
    reference = "ㅇㅇ"
    base = SequenceModel([reference])
    model = GuardedBurstLanguageModel(base, max_attempts=2, min_reference_chars=8)
    assert model.generate_burst(_packet(reference)).messages == ("ㅇㅇ",)
    assert len(base.calls) == 1
