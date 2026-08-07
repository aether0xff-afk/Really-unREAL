from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class Action(StrEnum):
    WAIT = "WAIT"
    REPLY = "REPLY"
    INITIATE = "INITIATE"


@dataclass(frozen=True, slots=True)
class ObservableState:
    now: datetime
    user_message_pending: bool = False
    scheduled_reply_at: datetime | None = None
    scheduled_initiation_at: datetime | None = None


class BaselineActionPolicy:
    """Deterministic scaffold for the future learned temporal policy.

    The important invariant is that no text generator gets to decide whether a
    message should exist. Timing/scheduling decides first; generation happens
    only after REPLY or INITIATE is selected.
    """

    def choose(self, state: ObservableState) -> Action:
        if (
            state.user_message_pending
            and state.scheduled_reply_at is not None
            and state.now >= state.scheduled_reply_at
        ):
            return Action.REPLY
        if (
            not state.user_message_pending
            and state.scheduled_initiation_at is not None
            and state.now >= state.scheduled_initiation_at
        ):
            return Action.INITIATE
        return Action.WAIT
