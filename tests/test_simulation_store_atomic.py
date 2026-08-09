from datetime import datetime, timedelta

from backend.simulation.action_policy import Action
from backend.simulation.store import SQLiteSimulationStore


BASE = datetime(2026, 8, 9, 0, 0)


def _scheduled(store: SQLiteSimulationStore, action: Action = Action.REPLY):
    return store.schedule(
        twin_person_id="target",
        platform="kakao",
        conversation_id="c",
        action=action,
        due_at=BASE,
        created_at=BASE - timedelta(seconds=30),
    )


def _claim(store: SQLiteSimulationStore, *, now: datetime = BASE):
    return store.claim_due_events(
        now=now,
        twin_person_id="target",
        platform="kakao",
        conversation_id="c",
    )[0]


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
    assert first_claim[0].claim_token
    assert second_claim == []
    assert first_store.event(event.event_id).status == "CLAIMED"


def test_atomic_completion_writes_messages_and_processes_event_together(tmp_path) -> None:
    store = SQLiteSimulationStore(tmp_path / "complete.db")
    event = _scheduled(store)
    claimed = _claim(store)
    assert claimed.claim_token is not None

    store.complete_claimed_event_with_messages(
        event_id=claimed.event_id,
        claim_token=claimed.claim_token,
        twin_person_id="target",
        platform="kakao",
        conversation_id="c",
        sender_person_id="target",
        messages=((BASE, "답장"),),
        metadata={"event_id": claimed.event_id},
    )

    assert store.event(event.event_id).status == "PROCESSED"
    assert [
        message.text
        for message in store.simulation_messages(
            twin_person_id="target", platform="kakao", conversation_id="c"
        )
    ] == ["답장"]


def test_reset_while_claimed_prevents_stale_generation_from_reappearing(tmp_path) -> None:
    store = SQLiteSimulationStore(tmp_path / "reset.db")
    event = _scheduled(store)
    claimed = _claim(store)
    assert claimed.claim_token is not None
    store.clear_conversation(
        twin_person_id="target",
        platform="kakao",
        conversation_id="c",
    )

    try:
        store.complete_claimed_event_with_messages(
            event_id=event.event_id,
            claim_token=claimed.claim_token,
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

    assert (
        store.simulation_messages(
            twin_person_id="target", platform="kakao", conversation_id="c"
        )
        == []
    )


def test_stale_claim_is_recovered_to_retry(tmp_path) -> None:
    store = SQLiteSimulationStore(tmp_path / "stale.db")
    event = _scheduled(store)
    _claim(store)
    recovered = store.recover_stale_claims(
        now=BASE + timedelta(minutes=10),
        stale_after_seconds=60,
    )
    assert recovered == 1
    current = store.event(event.event_id)
    assert current.status == "RETRY"
    assert current.claim_token is None
    assert current.ready_at == BASE + timedelta(minutes=10)


def test_recovered_and_reclaimed_event_rejects_old_worker_completion(tmp_path) -> None:
    path = tmp_path / "generation-owner.db"
    first_store = SQLiteSimulationStore(path)
    second_store = SQLiteSimulationStore(path)
    event = _scheduled(first_store)

    old_claim = _claim(first_store)
    assert old_claim.claim_token is not None

    recovered_at = BASE + timedelta(minutes=10)
    assert first_store.recover_stale_claims(
        now=recovered_at,
        stale_after_seconds=60,
    ) == 1
    new_claim = _claim(second_store, now=recovered_at)
    assert new_claim.event_id == event.event_id
    assert new_claim.claim_token is not None
    assert new_claim.claim_token != old_claim.claim_token

    try:
        first_store.complete_claimed_event_with_messages(
            event_id=event.event_id,
            claim_token=old_claim.claim_token,
            twin_person_id="target",
            platform="kakao",
            conversation_id="c",
            sender_person_id="target",
            messages=((recovered_at, "OLD WORKER MUST NOT COMMIT"),),
        )
    except KeyError:
        pass
    else:
        raise AssertionError("stale generation must not steal a newer claim")

    assert (
        first_store.simulation_messages(
            twin_person_id="target", platform="kakao", conversation_id="c"
        )
        == []
    )
    assert first_store.event(event.event_id).claim_token == new_claim.claim_token

    second_store.complete_claimed_event_with_messages(
        event_id=event.event_id,
        claim_token=new_claim.claim_token,
        twin_person_id="target",
        platform="kakao",
        conversation_id="c",
        sender_person_id="target",
        messages=((recovered_at, "NEW WORKER"),),
    )
    assert [
        message.text
        for message in second_store.simulation_messages(
            twin_person_id="target", platform="kakao", conversation_id="c"
        )
    ] == ["NEW WORKER"]


def test_recovered_and_reclaimed_read_rejects_old_worker_side_effect(tmp_path) -> None:
    path = tmp_path / "read-owner.db"
    first_store = SQLiteSimulationStore(path)
    second_store = SQLiteSimulationStore(path)
    event = _scheduled(first_store, Action.READ)
    first_store.append_simulation_messages(
        twin_person_id="target",
        platform="kakao",
        conversation_id="c",
        sender_person_id="self",
        messages=((BASE - timedelta(seconds=5), "안읽음"),),
        metadata={"read_status": "UNREAD"},
    )

    old_claim = _claim(first_store)
    assert old_claim.claim_token is not None
    recovered_at = BASE + timedelta(minutes=10)
    first_store.recover_stale_claims(now=recovered_at, stale_after_seconds=60)
    new_claim = _claim(second_store, now=recovered_at)
    assert new_claim.claim_token is not None

    try:
        first_store.complete_claimed_read_event(
            event_id=event.event_id,
            claim_token=old_claim.claim_token,
            twin_person_id="target",
            platform="kakao",
            conversation_id="c",
            sender_person_id="self",
            read_at=BASE,
        )
    except KeyError:
        pass
    else:
        raise AssertionError("stale READ claim must not mutate message metadata")

    message = first_store.simulation_messages(
        twin_person_id="target", platform="kakao", conversation_id="c"
    )[0]
    assert message.metadata.get("read_at") is None

    second_store.complete_claimed_read_event(
        event_id=event.event_id,
        claim_token=new_claim.claim_token,
        twin_person_id="target",
        platform="kakao",
        conversation_id="c",
        sender_person_id="self",
        read_at=BASE,
    )
    message = second_store.simulation_messages(
        twin_person_id="target", platform="kakao", conversation_id="c"
    )[0]
    assert message.metadata["read_at"] == BASE.isoformat()
