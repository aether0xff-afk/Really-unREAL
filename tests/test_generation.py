from datetime import datetime, timedelta

import pytest

from backend.fusion import EvidenceContext, EvidenceMessage
from backend.generation import GeneratedBurst, evaluate_generated_burst
from backend.models import ChatMessage
from backend.replay import ReplayCase
from backend.simulation.action_policy import Action


BASE = datetime(2026, 8, 7, 20, 0)


def _evidence(at: datetime, sender: str, text: str) -> EvidenceMessage:
    return EvidenceMessage(
        message=ChatMessage(at, sender, text),
        platform="kakao",
        conversation_id="c",
        context=EvidenceContext.KAKAO_DIRECT,
        sender_person_id=sender,
        evidence_weight=1.0,
    )


def _case() -> ReplayCase:
    visible = _evidence(BASE, "self", "뭐함")
    first = _evidence(BASE + timedelta(minutes=2), "target", "집ㅋㅋ")
    second = _evidence(BASE + timedelta(minutes=2, seconds=30), "target", "왜")
    return ReplayCase(
        case_id="case",
        person_id="target",
        platform="kakao",
        conversation_id="c",
        evidence_context=EvidenceContext.KAKAO_DIRECT,
        evidence_weight=1.0,
        action=Action.REPLY,
        observation_end=BASE,
        action_at=BASE + timedelta(minutes=2),
        observed_delay_seconds=120.0,
        delay_lower_seconds=60.0,
        delay_upper_seconds=180.0,
        context=(visible,),
        target_burst=(first, second),
        burst_size=2,
        session_restart=False,
    )


def test_generated_burst_accepts_messages_only_json() -> None:
    burst = GeneratedBurst.from_json('{"messages": ["집ㅋㅋ", "왜"]}')

    assert burst.messages == ("집ㅋㅋ", "왜")


def test_generated_burst_rejects_empty_or_wrong_shape() -> None:
    with pytest.raises(ValueError):
        GeneratedBurst.from_json('{"text": "hi"}')
    with pytest.raises(ValueError):
        GeneratedBurst.from_json('{"messages": []}')


def test_generation_metrics_are_computed_only_after_output_exists() -> None:
    metrics = evaluate_generated_burst(
        GeneratedBurst(("집ㅋㅋ", "왜")),
        _case(),
    )

    assert metrics.burst_size_absolute_error == 0
    assert metrics.total_char_length_absolute_error == 0
    assert metrics.char_bigram_f1 == 1.0
    assert metrics.laugh_presence_match is True
