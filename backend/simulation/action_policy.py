from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class Action(StrEnum):
    """Observable/live behavior actions.

    READ is a simulation-only latent event. FOLLOW_UP means another message inside
    an already-active session after the target's own previous burst. INITIATE is
    reserved for a new-session start after a long idle gap.
    """

    WAIT = "WAIT"
    READ = "READ"
    REPLY = "REPLY"
    FOLLOW_UP = "FOLLOW_UP"
    INITIATE = "INITIATE"


MESSAGE_ACTIONS = frozenset({Action.REPLY, Action.FOLLOW_UP, Action.INITIATE})


@dataclass(frozen=True, slots=True)
class ObservableState:
    now: datetime
    user_message_pending: bool = False
    scheduled_read_at: datetime | None = None
    scheduled_reply_at: datetime | None = None
    scheduled_follow_up_at: datetime | None = None
    scheduled_initiation_at: datetime | None = None


class BaselineActionPolicy:
    """Deterministic dispatcher over already-scheduled observable behavior.

    Whether a behavior should exist is decided before this policy is called.
    Generation is never allowed to invent a message action on its own.
    """

    def choose(self, state: ObservableState) -> Action:
        if (
            state.user_message_pending
            and state.scheduled_read_at is not None
            and state.now >= state.scheduled_read_at
        ):
            return Action.READ
        if (
            state.user_message_pending
            and state.scheduled_reply_at is not None
            and state.now >= state.scheduled_reply_at
        ):
            return Action.REPLY
        if (
            not state.user_message_pending
            and state.scheduled_follow_up_at is not None
            and state.now >= state.scheduled_follow_up_at
        ):
            return Action.FOLLOW_UP
        if (
            not state.user_message_pending
            and state.scheduled_initiation_at is not None
            and state.now >= state.scheduled_initiation_at
        ):
            return Action.INITIATE
        return Action.WAIT
