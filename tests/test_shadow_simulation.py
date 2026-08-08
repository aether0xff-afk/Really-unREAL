from datetime import datetime, timedelta

from backend.fusion import EvidenceContext, EvidenceConversation, EvidenceMessage, PersonEvidence
from backend.generation import GeneratedBurst
from backend.models import ChatMessage
from backend.replay import ReplayCase
from backend.replay_baseline import EmpiricalTimingBaseline
from backend.simulation.action_policy import Action
from backend.simulation.shadow import run_shadow_simulation


BASE = datetime(2026, 8, 8, 12, 0)


def _message(at: datetime, sender_id: str, text: str) -> EvidenceMessage:
    return EvidenceMessage(
        message=ChatMessage(at, sender_id, text),
        platform="kakao",
        conversation_id="c",
        context=EvidenceContext.KAKAO_DIRECT,
        sender_person_id=sender_id,
        evidence_weight=1.0,
    )


def _case(
    case_id: str,
    *,
    previous: EvidenceMessage,
    target: EvidenceMessage,
    action: Action,
) -> ReplayCase:
    delay = (target.message.timestamp - previous.message.timestamp).total_seconds()
    return ReplayCase(
        case_id=case_id,
        person_id="target",
        platform="kakao",
        conversation_id="c",
        evidence_context=EvidenceContext.KAKAO_DIRECT,
        evidence_weight=1.0,
        action=action,
        observation_end=previous.message.timestamp,
        action_at=target.message.timestamp,
        observed_delay_seconds=delay,
        delay_lower_seconds=delay,
        delay_upper_seconds=delay,
        context=(previous,),
        target_burst=(target,),
        burst_size=1,
        session_restart=False,
    )


class FutureBarrierModel:
    def generate_burst(self, packet):
        serialized = str(packet.to_dict())
        assert "REAL_FUTURE" not in serialized
        return GeneratedBurst(("sim",))


def test_shadow_replaces_hidden_target_history_with_simulated_output() -> None:
    old_other = _message(BASE - timedelta(hours=1), "other", "old ping")
    old_target = _message(BASE - timedelta(minutes=59), "target", "old reply")
    old_follow = _message(BASE - timedelta(minutes=49), "target", "old follow")
    incoming = _message(BASE + timedelta(minutes=2), "other", "hello")
    real_future = _message(BASE + timedelta(minutes=3), "target", "REAL_FUTURE")

    evidence = PersonEvidence(
        "target",
        (
            EvidenceConversation(
                "kakao",
                "c",
                EvidenceContext.KAKAO_DIRECT,
                (old_other, old_target, old_follow, incoming, real_future),
            ),
        ),
    )
    pre_reply = _case(
        "pre-reply",
        previous=old_other,
        target=old_target,
        action=Action.REPLY,
    )
    pre_init = _case(
        "pre-init",
        previous=old_target,
        target=old_follow,
        action=Action.INITIATE,
    )
    held_out = _case(
        "held-out",
        previous=incoming,
        target=real_future,
        action=Action.REPLY,
    )
    cases = [pre_reply, pre_init, held_out]
    timing = EmpiricalTimingBaseline.fit([pre_reply, pre_init])

    report, predicted = run_shadow_simulation(
        evidence=evidence,
        replay_cases=cases,
        timing=timing,
        language_model=FutureBarrierModel(),
        conversation_id="c",
        start_at=BASE,
        end_at=BASE + timedelta(minutes=4),
    )

    assert len(predicted) == 1
    assert predicted[0].action == Action.REPLY
    assert predicted[0].at == BASE + timedelta(minutes=3)
    assert report.matched_event_count == 1
    assert report.event_precision == 1.0
    assert report.event_recall == 1.0
    assert report.median_absolute_timing_error_seconds == 0.0
