from datetime import datetime, timedelta

from backend.fusion import EvidenceContext, EvidenceConversation, EvidenceMessage, PersonEvidence
from backend.generation_context import build_generation_context
from backend.models import ChatMessage
from backend.replay import ReplayCase
from backend.retrieval import CutoffExampleIndex
from backend.simulation.action_policy import Action


BASE = datetime(2026, 8, 7, 20, 0)


def _message(at: datetime, sender: str, text: str) -> EvidenceMessage:
    return EvidenceMessage(
        message=ChatMessage(at, sender, text),
        platform="kakao",
        conversation_id="c",
        context=EvidenceContext.KAKAO_DIRECT,
        sender_person_id=sender,
        evidence_weight=1.0,
    )


def _case(case_id: str, target_text: str, *, ambiguous: bool = False) -> ReplayCase:
    visible = _message(BASE, "self", "시험 어땠어")
    target = _message(BASE + timedelta(minutes=5), "target", target_text)
    return ReplayCase(
        case_id=case_id,
        person_id="target",
        platform="kakao",
        conversation_id="c",
        evidence_context=EvidenceContext.KAKAO_DIRECT,
        evidence_weight=1.0,
        action=Action.REPLY,
        observation_end=BASE,
        action_at=BASE + timedelta(minutes=5),
        observed_delay_seconds=300.0,
        delay_lower_seconds=240.0,
        delay_upper_seconds=360.0,
        context=(visible,),
        target_burst=(target,),
        burst_size=1,
        session_restart=ambiguous,
        action_is_ambiguous=ambiguous,
    )


def _historical() -> ReplayCase:
    return ReplayCase(
        case_id="old",
        person_id="target",
        platform="kakao",
        conversation_id="old-c",
        evidence_context=EvidenceContext.KAKAO_DIRECT,
        evidence_weight=1.0,
        action=Action.REPLY,
        observation_end=BASE - timedelta(days=2, minutes=5),
        action_at=BASE - timedelta(days=2),
        observed_delay_seconds=300.0,
        delay_lower_seconds=240.0,
        delay_upper_seconds=360.0,
        context=(_message(BASE - timedelta(days=2, minutes=5), "self", "시험 어땠어"),),
        target_burst=(_message(BASE - timedelta(days=2), "target", "망함ㅋㅋ"),),
        burst_size=1,
        session_restart=False,
    )


def _evidence() -> PersonEvidence:
    past_target = _message(BASE - timedelta(days=3), "target", "ㅋㅋ")
    future_target = _message(BASE + timedelta(days=1), "target", "future style")
    return PersonEvidence(
        person_id="target",
        conversations=(
            EvidenceConversation(
                platform="kakao",
                conversation_id="persona",
                context=EvidenceContext.KAKAO_DIRECT,
                messages=(past_target, future_target),
            ),
        ),
    )


def test_generation_packet_does_not_depend_on_hidden_target_or_real_action() -> None:
    historical = _historical()
    query_a = _case("query-a", "hidden answer A")
    query_b = _case("query-b", "completely different hidden answer B")
    evidence = _evidence()
    index = CutoffExampleIndex.from_replay_cases([historical])

    packet_a = build_generation_context(
        query_a,
        evidence,
        index,
        chosen_action=Action.REPLY,
    )
    packet_b = build_generation_context(
        query_b,
        evidence,
        index,
        chosen_action=Action.REPLY,
    )

    assert packet_a.to_dict() == packet_b.to_dict()
    assert packet_a.chosen_action == "REPLY"
    assert packet_a.language_profile.message_count == 1
    assert packet_a.retrieved_examples[0].response_texts == ()
    assert packet_a.retrieved_examples[0].response_shape.message_lengths == (4,)
    assert packet_a.retrieved_examples[0].response_shape.laugh_expression_count == 1


def test_raw_historical_response_requires_explicit_ablation_opt_in() -> None:
    index = CutoffExampleIndex.from_replay_cases([_historical()])
    packet = build_generation_context(
        _case("query", "hidden"),
        _evidence(),
        index,
        chosen_action=Action.REPLY,
        raw_response_examples=1,
    )

    assert packet.retrieved_examples[0].response_texts == ("망함ㅋㅋ",)


def test_ambiguous_long_gap_can_disable_action_specific_retrieval() -> None:
    historical = _historical()
    index = CutoffExampleIndex.from_replay_cases([historical])
    packet = build_generation_context(
        _case("query", "hidden", ambiguous=True),
        _evidence(),
        index,
        chosen_action=Action.INITIATE,
        action_specific_retrieval=False,
    )

    assert packet.action_role_ambiguous is True
    assert len(packet.retrieved_examples) == 1
