from __future__ import annotations

import hashlib
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Iterable, Sequence

from backend.fusion import EvidenceContext, EvidenceConversation, EvidenceMessage, PersonEvidence
from backend.simulation.action_policy import Action


_DIRECT_CONTEXTS = {
    EvidenceContext.KAKAO_DIRECT,
    EvidenceContext.INSTAGRAM_DIRECT,
}


@dataclass(frozen=True, slots=True)
class ReplayCase:
    """One hidden-future behavioral event for Historical Replay.

    ``context`` contains only observations available strictly before the target
    burst. ``target_burst`` is the held-out real continuation. Timing is kept as
    an interval because KakaoTalk exports are usually minute-precision while
    Instagram timestamps are much finer.

    ``action`` is the best observable REPLY/INITIATE proxy from adjacent sender
    order. When a session restarts after a long gap, sender order alone cannot
    distinguish a late reply from a genuinely new initiation, so
    ``action_is_ambiguous`` is set and the positive action label is excluded from
    action-class evaluation. WAIT timing before the event remains usable.
    """

    case_id: str
    person_id: str
    platform: str
    conversation_id: str
    evidence_context: EvidenceContext
    evidence_weight: float
    action: Action
    observation_end: datetime
    action_at: datetime
    observed_delay_seconds: float
    delay_lower_seconds: float
    delay_upper_seconds: float
    context: tuple[EvidenceMessage, ...]
    target_burst: tuple[EvidenceMessage, ...]
    burst_size: int
    session_restart: bool
    action_is_ambiguous: bool = False


@dataclass(frozen=True, slots=True)
class ActionSnapshot:
    """A point-in-time action label derived from a ReplayCase.

    WAIT snapshots are sampled before the real event. The final snapshot occurs
    at the observed event timestamp only when REPLY/INITIATE is not ambiguous.
    """

    case_id: str
    observed_at: datetime
    expected_action: Action
    elapsed_seconds: float
    remaining_observed_seconds: float


@dataclass(frozen=True, slots=True)
class ReplaySplit:
    train: tuple[ReplayCase, ...]
    validation: tuple[ReplayCase, ...]
    test: tuple[ReplayCase, ...]


@dataclass(frozen=True, slots=True)
class ReplayAudit:
    event_count: int
    reply_count: int
    initiate_count: int
    confident_reply_count: int
    confident_initiate_count: int
    ambiguous_action_count: int
    median_observed_delay_seconds: float | None
    median_burst_size: float | None
    snapshot_count: int
    wait_snapshot_count: int
    action_snapshot_count: int
    context_counts: dict[str, int]
    platform_counts: dict[str, int]
    first_event_at: str | None
    last_event_at: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _timestamp_precision_seconds(message: EvidenceMessage) -> float:
    value = message.message.metadata.get("timestamp_precision_seconds", 0.0)
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


def _delay_interval(
    previous: EvidenceMessage,
    current: EvidenceMessage,
) -> tuple[float, float, float]:
    observed = max(
        0.0,
        (current.message.timestamp - previous.message.timestamp).total_seconds(),
    )
    previous_precision = _timestamp_precision_seconds(previous)
    current_precision = _timestamp_precision_seconds(current)
    lower = max(0.0, observed - previous_precision)
    upper = max(lower, observed + current_precision)
    return observed, lower, upper


def _case_id(
    person_id: str,
    conversation: EvidenceConversation,
    start_index: int,
    action_at: datetime,
) -> str:
    key = "|".join(
        (
            person_id,
            conversation.platform,
            conversation.conversation_id,
            str(start_index),
            action_at.isoformat(),
        )
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _target_burst_end(
    messages: Sequence[EvidenceMessage],
    start_index: int,
    person_id: str,
    burst_gap_seconds: float,
) -> int:
    end = start_index + 1
    while end < len(messages):
        previous = messages[end - 1]
        current = messages[end]
        if current.sender_person_id != person_id:
            break
        gap = (current.message.timestamp - previous.message.timestamp).total_seconds()
        if gap > burst_gap_seconds:
            break
        end += 1
    return end


def _adjacent_action(
    *,
    conversation: EvidenceConversation,
    target_person_id: str,
    previous_sender_person_id: str,
    self_person_id: str,
) -> Action | None:
    """Infer the coarse action role without assuming target != self.

    In a direct conversation, there is only one counterpart to the target. A
    resolved message from that counterpart is therefore a REPLY cue regardless
    of whether the target is another person or the user's own SELF_TWIN. A
    second target burst after the target's own previous burst is the observable
    INITIATE/follow-up proxy.

    Group replay remains conservative: only an explicit reply to ``self`` can be
    labelled for an other-person twin. SELF_TWIN group action labels are skipped
    because an adjacent third-party message is not enough to identify whom the
    user's message was addressing.
    """

    if previous_sender_person_id == target_person_id:
        return Action.INITIATE
    if conversation.context in _DIRECT_CONTEXTS:
        return Action.REPLY
    if target_person_id != self_person_id and previous_sender_person_id == self_person_id:
        return Action.REPLY
    return None


def build_replay_cases(
    evidence: PersonEvidence,
    *,
    self_person_id: str,
    context_size: int = 30,
    burst_gap_seconds: float = 120.0,
    session_gap_hours: float = 6.0,
    include_group: bool = False,
) -> list[ReplayCase]:
    """Build leakage-safe Historical Replay events from source-aware evidence.

    By default only direct conversations are used for action/timing evaluation.
    Group conversations remain useful persona evidence but are not reliable
    labels for whether the target was responding to *the user*.

    Direct-conversation labels are target-relative rather than user-relative, so
    the exact same replay builder supports both another-person twin and a
    SELF_TWIN. Inside one active session, a message after the counterpart is a
    REPLY; a new burst after the target's own previous burst is the coarse
    INITIATE/follow-up proxy. After a long session gap, adjacent sender order is
    not enough to distinguish late reply from genuinely new initiation. Those
    cases are retained for timing/content replay but marked
    ``action_is_ambiguous``.
    """

    if context_size < 1:
        raise ValueError("context_size must be >= 1")
    if burst_gap_seconds < 0:
        raise ValueError("burst_gap_seconds must be >= 0")
    if session_gap_hours <= 0:
        raise ValueError("session_gap_hours must be > 0")

    session_gap_seconds = session_gap_hours * 3600.0
    cases: list[ReplayCase] = []

    for conversation in evidence.conversations:
        if not include_group and conversation.context not in _DIRECT_CONTEXTS:
            continue
        messages = conversation.messages
        index = 0
        while index < len(messages):
            current = messages[index]
            if current.sender_person_id != evidence.person_id:
                index += 1
                continue

            is_continuation = False
            if index > 0 and messages[index - 1].sender_person_id == evidence.person_id:
                gap = (
                    current.message.timestamp
                    - messages[index - 1].message.timestamp
                ).total_seconds()
                is_continuation = gap <= burst_gap_seconds
            if is_continuation:
                index += 1
                continue

            burst_end = _target_burst_end(
                messages,
                index,
                evidence.person_id,
                burst_gap_seconds,
            )
            if index == 0:
                # Left-censored: the export does not tell us how long the person
                # had already been silent before the first observed message.
                index = burst_end
                continue

            previous = messages[index - 1]
            if previous.sender_person_id is None:
                index = burst_end
                continue

            action = _adjacent_action(
                conversation=conversation,
                target_person_id=evidence.person_id,
                previous_sender_person_id=previous.sender_person_id,
                self_person_id=self_person_id,
            )
            if action is None:
                index = burst_end
                continue

            observed, lower, upper = _delay_interval(previous, current)
            session_restart = observed > session_gap_seconds
            context_start = max(0, index - context_size)
            burst = tuple(messages[index:burst_end])
            cases.append(
                ReplayCase(
                    case_id=_case_id(
                        evidence.person_id,
                        conversation,
                        index,
                        current.message.timestamp,
                    ),
                    person_id=evidence.person_id,
                    platform=conversation.platform,
                    conversation_id=conversation.conversation_id,
                    evidence_context=conversation.context,
                    evidence_weight=current.evidence_weight,
                    action=action,
                    observation_end=previous.message.timestamp,
                    action_at=current.message.timestamp,
                    observed_delay_seconds=observed,
                    delay_lower_seconds=lower,
                    delay_upper_seconds=upper,
                    context=tuple(messages[context_start:index]),
                    target_burst=burst,
                    burst_size=len(burst),
                    session_restart=session_restart,
                    action_is_ambiguous=session_restart,
                )
            )
            index = burst_end

    cases.sort(key=lambda case: (case.action_at, case.case_id))
    return cases


def _evenly_select(values: Sequence[float], limit: int) -> list[float]:
    if limit <= 0 or not values:
        return []
    if len(values) <= limit:
        return list(values)
    if limit == 1:
        return [values[-1]]
    indexes = {
        round(index * (len(values) - 1) / (limit - 1))
        for index in range(limit)
    }
    return [values[index] for index in sorted(indexes)]


def build_action_snapshots(
    cases: Iterable[ReplayCase],
    *,
    wait_offsets_seconds: Sequence[float] = (
        60.0,
        300.0,
        1800.0,
        7200.0,
        21600.0,
        86400.0,
    ),
    max_wait_snapshots_per_case: int = 4,
) -> list[ActionSnapshot]:
    """Materialize safe WAIT negatives plus trustworthy positive labels.

    Long-gap cases still contribute WAIT observations before the event, but the
    final REPLY/INITIATE snapshot is omitted when the action role is ambiguous.
    """

    normalized_offsets = sorted(
        {float(value) for value in wait_offsets_seconds if float(value) > 0}
    )
    snapshots: list[ActionSnapshot] = []
    for case in cases:
        valid_waits = [
            offset
            for offset in normalized_offsets
            if offset < case.delay_lower_seconds
        ]
        for offset in _evenly_select(valid_waits, max_wait_snapshots_per_case):
            snapshots.append(
                ActionSnapshot(
                    case_id=case.case_id,
                    observed_at=case.observation_end + timedelta(seconds=offset),
                    expected_action=Action.WAIT,
                    elapsed_seconds=offset,
                    remaining_observed_seconds=max(
                        0.0,
                        case.observed_delay_seconds - offset,
                    ),
                )
            )
        if not case.action_is_ambiguous:
            snapshots.append(
                ActionSnapshot(
                    case_id=case.case_id,
                    observed_at=case.action_at,
                    expected_action=case.action,
                    elapsed_seconds=case.observed_delay_seconds,
                    remaining_observed_seconds=0.0,
                )
            )

    snapshots.sort(key=lambda item: (item.observed_at, item.case_id))
    return snapshots


def chronological_split(
    cases: Iterable[ReplayCase],
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> ReplaySplit:
    """Split by time, never randomly, to avoid future-style leakage."""

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train + validation fractions must be < 1")

    ordered = sorted(cases, key=lambda case: (case.action_at, case.case_id))
    n = len(ordered)
    train_end = int(n * train_fraction)
    validation_end = train_end + int(n * validation_fraction)
    return ReplaySplit(
        train=tuple(ordered[:train_end]),
        validation=tuple(ordered[train_end:validation_end]),
        test=tuple(ordered[validation_end:]),
    )


def audit_replay(
    cases: Sequence[ReplayCase],
    snapshots: Sequence[ActionSnapshot] | None = None,
) -> ReplayAudit:
    snapshots = list(snapshots or build_action_snapshots(cases))
    delays = [case.observed_delay_seconds for case in cases]
    burst_sizes = [case.burst_size for case in cases]
    context_counts = Counter(case.evidence_context.value for case in cases)
    platform_counts = Counter(case.platform for case in cases)
    wait_count = sum(item.expected_action == Action.WAIT for item in snapshots)

    return ReplayAudit(
        event_count=len(cases),
        reply_count=sum(case.action == Action.REPLY for case in cases),
        initiate_count=sum(case.action == Action.INITIATE for case in cases),
        confident_reply_count=sum(
            case.action == Action.REPLY and not case.action_is_ambiguous
            for case in cases
        ),
        confident_initiate_count=sum(
            case.action == Action.INITIATE and not case.action_is_ambiguous
            for case in cases
        ),
        ambiguous_action_count=sum(case.action_is_ambiguous for case in cases),
        median_observed_delay_seconds=(
            float(statistics.median(delays)) if delays else None
        ),
        median_burst_size=(
            float(statistics.median(burst_sizes)) if burst_sizes else None
        ),
        snapshot_count=len(snapshots),
        wait_snapshot_count=wait_count,
        action_snapshot_count=len(snapshots) - wait_count,
        context_counts=dict(context_counts),
        platform_counts=dict(platform_counts),
        first_event_at=cases[0].action_at.isoformat() if cases else None,
        last_event_at=cases[-1].action_at.isoformat() if cases else None,
    )