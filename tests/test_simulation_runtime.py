from datetime import datetime, timedelta

from backend.fusion import EvidenceContext, EvidenceConversation, EvidenceMessage, PersonEvidence
from backend.generation import GeneratedBurst
from backend.models import ChatMessage, MemorySource
from backend.replay import ReplayCase
from backend.replay_baseline import EmpiricalTimingBaseline
from backend.retrieval import CutoffExampleIndex
from backend.simulation.action_policy import Action
from backend.simulation.runtime import LiveSimulationEngine
from backend.simulation.store import SQLiteSimulationStore


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


def _case(case_id: str, action: Action, delay: float, previous_sender: str) -> ReplayCase:
    previous = _message(BASE, previous_sender, "x")
    target = _message(BASE + timedelta(seconds=delay), "target", "y")
    return ReplayCase(
        case_id=case_id,
        person_id="target",
        platform="kakao",
        conversation_id="c",
        evidence_context=EvidenceContext.KAKAO_DIRECT,
        evidence_weight=1.0,
        action=action,
        observation_end=BASE,
        action_at=BASE + timedelta(seconds=delay),
        observed_delay_seconds=delay,
        delay_lower_seconds=delay,
        delay_upper_seconds=delay,
        context=(previous,),
        target_burst=(target,),
        burst_size=1,
        session_restart=False,
    )


class FakeModel:
    def generate_burst(self, packet):
        return GeneratedBurst(("sim",))


def test_live_runtime_persists_due_events_and_simulation_memory(tmp_path) -> None:
    train = [
        _case("reply", Action.REPLY, 60.0, "other"),
        _case("init", Action.INITIATE, 300.0, "target"),
    ]
    timing = EmpiricalTimingBaseline.fit(train)
    evidence = PersonEvidence(
        "target",
        (
            EvidenceConversation(
                "kakao",
                "c",
                EvidenceContext.KAKAO_DIRECT,
                (_message(BASE - timedelta(minutes=1), "other", "뭐함"),),
            ),
        ),
    )
    index = CutoffExampleIndex.from_replay_cases(train)
    path = tmp_path / "simulation.db"
    store = SQLiteSimulationStore(path)
    engine = LiveSimulationEngine(
        twin_person_id="target",
        platform="kakao",
        conversation_id="c",
        evidence=evidence,
        retrieval_index=index,
        timing=timing,
        language_model=FakeModel(),
        store=store,
    )

    scheduled = engine.observe_counterpart_message(observed_at=BASE)
    assert scheduled.action == Action.REPLY
    assert scheduled.due_at == BASE + timedelta(seconds=60)
    assert engine.process_due(
        now=BASE + timedelta(seconds=30),
        visible_context=evidence.conversations[0].messages,
    ) == []

    emissions = engine.process_due(
        now=BASE + timedelta(seconds=61),
        visible_context=evidence.conversations[0].messages,
    )
    assert len(emissions) == 1
    assert emissions[0].action == Action.REPLY
    stored = store.simulation_messages(
        twin_person_id="target",
        platform="kakao",
        conversation_id="c",
    )
    assert stored[0].source == MemorySource.SIMULATION
    assert stored[0].text == "sim"

    reopened = SQLiteSimulationStore(path)
    pending = reopened.pending_events()
    assert len(pending) == 1
    assert pending[0].action == Action.INITIATE

    recovered_engine = LiveSimulationEngine(
        twin_person_id="target",
        platform="kakao",
        conversation_id="c",
        evidence=evidence,
        retrieval_index=index,
        timing=timing,
        language_model=FakeModel(),
        store=reopened,
    )
    recovered = recovered_engine.recover(
        now=BASE + timedelta(seconds=400),
        visible_context=evidence.conversations[0].messages,
    )
    assert len(recovered) == 1
    assert recovered[0].action == Action.INITIATE
    assert reopened.pending_events() == []


def test_recovery_filters_context_that_arrived_after_event_due_time(tmp_path) -> None:
    reply_case = _case("reply", Action.REPLY, 60.0, "other")
    timing = EmpiricalTimingBaseline.fit([reply_case])
    incoming = _message(BASE, "other", "hello")
    future = _message(BASE + timedelta(seconds=120), "other", "FUTURE_MESSAGE")
    evidence = PersonEvidence(
        "target",
        (
            EvidenceConversation(
                "kakao",
                "c",
                EvidenceContext.KAKAO_DIRECT,
                (incoming, future),
            ),
        ),
    )

    class CausalityModel:
        def generate_burst(self, packet):
            assert "FUTURE_MESSAGE" not in str(packet.to_dict())
            return GeneratedBurst(("ok",))

    store = SQLiteSimulationStore(tmp_path / "causal.db")
    engine = LiveSimulationEngine(
        twin_person_id="target",
        platform="kakao",
        conversation_id="c",
        evidence=evidence,
        retrieval_index=CutoffExampleIndex.from_replay_cases([reply_case]),
        timing=timing,
        language_model=CausalityModel(),
        store=store,
    )
    engine.observe_counterpart_message(observed_at=BASE)

    recovered = engine.recover(
        now=BASE + timedelta(seconds=180),
        visible_context=(incoming, future),
    )

    assert len(recovered) == 1
    assert recovered[0].due_at == BASE + timedelta(seconds=60)
