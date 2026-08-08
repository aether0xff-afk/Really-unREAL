from datetime import datetime, timedelta

from backend.fusion import EvidenceContext, EvidenceMessage
from backend.models import ChatMessage
from backend.replay import ReplayCase
from backend.replay_sampling import EmpiricalTimingSampler
from backend.simulation.action_policy import Action


BASE = datetime(2026, 8, 8, 12, 0)


def _case(
    case_id: str,
    *,
    conversation_id: str,
    action: Action,
    lower: float,
    upper: float,
    weight: float = 1.0,
) -> ReplayCase:
    previous_sender = "target" if action == Action.INITIATE else "self"
    previous = EvidenceMessage(
        message=ChatMessage(BASE, previous_sender, "x"),
        platform="kakao",
        conversation_id=conversation_id,
        context=EvidenceContext.KAKAO_DIRECT,
        sender_person_id=previous_sender,
        evidence_weight=1.0,
    )
    midpoint = (lower + upper) / 2.0
    return ReplayCase(
        case_id=case_id,
        person_id="target",
        platform="kakao",
        conversation_id=conversation_id,
        evidence_context=EvidenceContext.KAKAO_DIRECT,
        evidence_weight=weight,
        action=action,
        observation_end=BASE,
        action_at=BASE + timedelta(seconds=midpoint),
        observed_delay_seconds=midpoint,
        delay_lower_seconds=lower,
        delay_upper_seconds=upper,
        context=(previous,),
        target_burst=(),
        burst_size=1,
        session_restart=False,
        action_is_ambiguous=False,
    )


def test_live_sampler_varies_inside_observed_kakao_intervals() -> None:
    sampler = EmpiricalTimingSampler(
        [
            _case("same-minute", conversation_id="c1", action=Action.REPLY, lower=0, upper=60),
            _case("later", conversation_id="c1", action=Action.REPLY, lower=120, upper=180),
            _case("other-chat", conversation_id="c2", action=Action.REPLY, lower=600, upper=600),
        ],
        seed=7,
    )

    values = [
        sampler.sample_delay_seconds(
            platform="kakao",
            conversation_id="c1",
            action=Action.REPLY,
        )
        for _ in range(20)
    ]

    assert all(value is not None for value in values)
    numeric = [float(value) for value in values if value is not None]
    assert len({round(value, 3) for value in numeric}) > 1
    assert all((0 <= value <= 60) or (120 <= value <= 180) for value in numeric)


def test_live_sampler_uses_exact_observation_when_interval_is_exact() -> None:
    sampler = EmpiricalTimingSampler(
        [_case("exact", conversation_id="c", action=Action.REPLY, lower=42, upper=42)],
        seed=1,
    )
    assert sampler.sample_delay_seconds(
        platform="kakao",
        conversation_id="c",
        action=Action.REPLY,
    ) == 42


def test_live_sampler_does_not_invent_missing_action_evidence() -> None:
    sampler = EmpiricalTimingSampler(
        [_case("reply", conversation_id="c", action=Action.REPLY, lower=0, upper=60)],
        seed=1,
    )
    assert sampler.sample_delay_seconds(
        platform="kakao",
        conversation_id="c",
        action=Action.INITIATE,
    ) is None
    assert sampler.sample_delay_seconds(
        platform="kakao",
        conversation_id="c",
        action=Action.WAIT,
    ) is None
