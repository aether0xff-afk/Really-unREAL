from datetime import datetime, timedelta

from backend.fusion import (
    EvidenceContext,
    EvidenceConversation,
    EvidenceMessage,
    PersonEvidence,
)
from backend.models import ChatMessage
from backend.replay import (
    audit_replay,
    build_action_snapshots,
    build_replay_cases,
    chronological_split,
)
from backend.simulation.action_policy import Action


BASE = datetime(2026, 8, 7, 20, 0)


def _evidence(
    minute: int,
    sender: str,
    person_id: str | None,
    text: str,
    *,
    platform: str = "kakao",
    precision: float = 60.0,
) -> EvidenceMessage:
    message = ChatMessage(
        BASE + timedelta(minutes=minute),
        sender,
        text,
        metadata={
            "platform": platform,
            "timestamp_precision_seconds": precision,
        },
    )
    return EvidenceMessage(
        message=message,
        platform=platform,
        conversation_id="direct-1",
        context=(
            EvidenceContext.KAKAO_DIRECT
            if platform == "kakao"
            else EvidenceContext.INSTAGRAM_DIRECT
        ),
        sender_person_id=person_id,
        evidence_weight=1.0,
    )


def _direct_evidence() -> PersonEvidence:
    messages = (
        _evidence(0, "나", "self", "야"),
        _evidence(2, "상대", "target", "왜"),
        _evidence(2, "상대", "target", "ㅋㅋ"),
        _evidence(10, "나", "self", "뭐해"),
        _evidence(20, "상대", "target", "집"),
        # Same sender after a gap larger than burst_gap => follow-up proxy.
        _evidence(30, "상대", "target", "근데"),
    )
    return PersonEvidence(
        person_id="target",
        conversations=(
            EvidenceConversation(
                platform="kakao",
                conversation_id="direct-1",
                context=EvidenceContext.KAKAO_DIRECT,
                messages=messages,
            ),
        ),
    )


def test_builds_reply_and_initiate_proxies_without_future_leakage() -> None:
    cases = build_replay_cases(
        _direct_evidence(),
        self_person_id="self",
        burst_gap_seconds=120,
    )

    assert [case.action for case in cases] == [
        Action.REPLY,
        Action.REPLY,
        Action.INITIATE,
    ]
    assert [case.action_is_ambiguous for case in cases] == [False, False, False]
    assert cases[0].burst_size == 2
    assert [item.message.text for item in cases[0].target_burst] == ["왜", "ㅋㅋ"]
    assert [item.message.text for item in cases[0].context] == ["야"]
    assert all(
        item.message.text not in {"왜", "ㅋㅋ"}
        for item in cases[0].context
    )


def test_kakao_delay_is_interval_censored_instead_of_fake_second_precision() -> None:
    case = build_replay_cases(
        _direct_evidence(),
        self_person_id="self",
    )[0]

    assert case.observed_delay_seconds == 120.0
    assert case.delay_lower_seconds == 60.0
    assert case.delay_upper_seconds == 180.0


def test_wait_snapshots_only_exist_when_event_cannot_already_have_happened() -> None:
    cases = build_replay_cases(
        _direct_evidence(),
        self_person_id="self",
    )
    second = cases[1]  # observed 10-minute reply, lower bound 9 minutes
    snapshots = build_action_snapshots(
        [second],
        wait_offsets_seconds=(60, 300, 600),
    )

    assert [snapshot.expected_action for snapshot in snapshots] == [
        Action.WAIT,
        Action.WAIT,
        Action.REPLY,
    ]
    assert [snapshot.elapsed_seconds for snapshot in snapshots] == [60.0, 300.0, 600.0]


def test_long_gap_keeps_timing_but_does_not_claim_reply_or_initiate() -> None:
    messages = (
        _evidence(0, "나", "self", "나중에 알려줘"),
        _evidence(24 * 60, "상대", "target", "ㅇㅇ"),
    )
    evidence = PersonEvidence(
        person_id="target",
        conversations=(
            EvidenceConversation(
                platform="kakao",
                conversation_id="direct-1",
                context=EvidenceContext.KAKAO_DIRECT,
                messages=messages,
            ),
        ),
    )
    case = build_replay_cases(evidence, self_person_id="self")[0]
    snapshots = build_action_snapshots(
        [case],
        wait_offsets_seconds=(60, 300, 7200),
    )

    assert case.action == Action.REPLY  # observable sender-order proxy only
    assert case.session_restart is True
    assert case.action_is_ambiguous is True
    assert all(snapshot.expected_action == Action.WAIT for snapshot in snapshots)

    audit = audit_replay([case], snapshots)
    assert audit.reply_count == 1
    assert audit.confident_reply_count == 0
    assert audit.ambiguous_action_count == 1
    assert audit.action_snapshot_count == 0


def test_group_conversations_are_excluded_from_action_replay_by_default() -> None:
    direct = _direct_evidence().conversations[0]
    group = EvidenceConversation(
        platform="kakao",
        conversation_id="group-1",
        context=EvidenceContext.KAKAO_GROUP,
        messages=direct.messages,
    )
    evidence = PersonEvidence("target", (group,))

    assert build_replay_cases(evidence, self_person_id="self") == []
    assert len(
        build_replay_cases(evidence, self_person_id="self", include_group=True)
    ) == 3


def test_chronological_split_and_audit_keep_future_out_of_training() -> None:
    cases = build_replay_cases(_direct_evidence(), self_person_id="self")
    split = chronological_split(
        cases,
        train_fraction=0.34,
        validation_fraction=0.33,
    )

    assert len(split.train) == 1
    assert len(split.validation) == 0
    assert len(split.test) == 2
    assert split.train[-1].action_at <= split.test[0].action_at

    audit = audit_replay(cases)
    assert audit.event_count == 3
    assert audit.reply_count == 2
    assert audit.initiate_count == 1
    assert audit.confident_reply_count == 2
    assert audit.confident_initiate_count == 1
    assert audit.ambiguous_action_count == 0
    assert audit.action_snapshot_count == 3
