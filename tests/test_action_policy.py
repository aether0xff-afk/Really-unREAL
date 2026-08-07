from datetime import datetime, timedelta

from backend.simulation.action_policy import Action, BaselineActionPolicy, ObservableState


def test_wait_is_default_action() -> None:
    now = datetime(2026, 8, 7, 23, 0)
    assert BaselineActionPolicy().choose(ObservableState(now=now)) == Action.WAIT


def test_reply_only_becomes_available_at_scheduled_time() -> None:
    now = datetime(2026, 8, 7, 23, 0)
    due = now + timedelta(minutes=10)
    policy = BaselineActionPolicy()

    assert (
        policy.choose(
            ObservableState(
                now=now,
                user_message_pending=True,
                scheduled_reply_at=due,
            )
        )
        == Action.WAIT
    )
    assert (
        policy.choose(
            ObservableState(
                now=due,
                user_message_pending=True,
                scheduled_reply_at=due,
            )
        )
        == Action.REPLY
    )
