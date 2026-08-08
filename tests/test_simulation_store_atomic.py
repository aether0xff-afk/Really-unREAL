from datetime import datetime, timedelta

from backend.simulation.action_policy import Action
from backend.simulation.store import SQLiteSimulationStore


BASE = datetime(2026, 8, 9, 0, 0)


def _scheduled(store: SQLiteSimulationStore):
    return store.schedule(
        twin_person_id="target",
        platform="kakao",
        conversation_id="c",
        action=Action.REPLY,
        due_at=BASE,
        created_at=BASE - timedelta(seconds=30),
    )


def test_due_event_can_only_be_claimed_once_across_two_store_instances(tmp_path) -> None:
    path = tmp_path / "atomic.db"
    first_store = SQLiteSimulationStore(path)
    second_store = SQLiteSimulationStore(path)
    event = _scheduled(first_store)

    first_claim = first_store.claim_due_events(
        now=BASE,
        twin_person_id="target",
        platform="kakao",
        conversation_id="c",
    )
    second_claim = second_store.claim_due_events(
        now=BASE,
        twin_person_id="target",
        platform="kakao",
        conversation_id="c",
    )

    assert [item.event_id for item in first_claim] == [event.event_id]
    assert second_claim == []
    assert first_store.event(event.event_id).status == "CLAIMED"


def test_atomic_completion_writes_messages_and_processes_event_together(tmp_path) -> None:
    store = SQLiteSimulationStore(tmp_path / "complete.db")
    event = _scheduled(store)
    claimed = store.claim_due_events(
        now=BASE,
        twin_person_id="target",
        platform="kakao",
        conversation_id="c",
    )[0]

    store.complete_claimed_event_with_messages(
        event_id=claimed.event_id,
        twin_person_id="target",
        platform="kakao",
        conversation_id="c",
        sender_person_id="target",
        messages=((BASE, "답장"),),
        metadata={"event_id": claimed.event_id},
    )

    assert store.event(event.event_id).status == "PROCESSED"
    assert [message.text for message in store.simulation_messages(
        twin_person_id="target", platform="kakao", conversation_id="c"
    )] == ["답장"]


def test_reset_while_claimed_prevents_stale_generation_from_reappearing(tmp_path) -> None:
    store = SQLiteSimulationStore(tmp_path / "reset.db")
    event = _scheduled(store)
    store.claim_due_events(
        now=BASE,
        twin_person_id="target",
        platform="kakao",
        conversation_id="c",
    )
    store.clear_conversation(
        twin_person_id="target",
        platform="kakao",
        conversation_id="c",
    )

    try:
        store.complete_claimed_event_with_messages(
            event_id=event.event_id,
            twin_person_id="target",
            platform="kakao",
            conversation_id="c",
            sender_person_id="target",
            messages=((BASE, "늦게 도착한 답장"),),
        )
    except KeyError:
        pass
    else:
        raise AssertionError("cancelled claim must reject stale completion")

    assert store.simulation_messages(
        twin_person_id="target", platform="kakao", conversation_id="c"
    ) == []


def test_stale_claim_is_recovered_to_retry(tmp_path) -> None:
    store = SQLiteSimulationStore(tmp_path / "stale.db")
    event = _scheduled(store)
    store.claim_due_events(
        now=BASE,
        twin_person_id="target",
        platform="kakao",
        conversation_id="c",
    )
    recovered = store.recover_stale_claims(
        now=BASE + timedelta(minutes=10),
        stale_after_seconds=60,
    )
    assert recovered == 1
    current = store.event(event.event_id)
    assert current.status == "RETRY"
    assert current.ready_at == BASE + timedelta(minutes=10)
