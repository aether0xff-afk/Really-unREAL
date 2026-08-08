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


def test_builds_reply_and_follow_up_without_future_leakage() -> None:
    cases = build_replay_cases(
        _direct_evidence(),
        self_person_id="self",
        burst_gap_seconds=120,
    )

    assert [case.action for case in cases] == [
        Action.REPLY,
        Action.REPLY,
        Action.FOLLOW_UP,
    ]
    assert [case.action_is_ambiguous for case in cases] == [False, False, False]
    assert cases[0].burst_size == 2
    assert [item.message.text for item in cases[0].target_burst] == ["왜", "ㅋㅋ"]
    assert [item.message.text for item in cases[0].context] == ["야"]
    assert all(item.message.text not in {"왜", "ㅋㅋ"} for item in cases[0].context)


def test_direct_replay_can_target_self_for_self_twin() -> None:
    messages = (
        _evidence(0, "친구", "friend", "뭐함"),
        _evidence(2, "나", "self", "집"),
        _evidence(10, "친구", "friend", "과제 함?"),
        _evidence(12, "나", "self", "ㄴㄴ"),
        _evidence(20, "나", "self", "근데 낼 몇시감"),
    )
    evidence = PersonEvidence(
        person_id="self",
        conversations=(
            EvidenceConversation(
                platform="kakao",
                conversation_id="direct-1",
                context=EvidenceContext.KAKAO_DIRECT,
                messages=messages,
            ),
        ),
    )

    cases = build_replay_cases(evidence, self_person_id="self")
    assert [case.action for case in cases] == [
        Action.REPLY,
        Action.REPLY,
        Action.FOLLOW_UP,
    ]
    assert all(case.person_id == "self" for case in cases)


def test_long_gap_after_target_is_confident_new_session_initiate() -> None:
    messages = (
        _evidence(0, "상대", "target", "일단 감"),
        _evidence(24 * 60, "상대", "target", "야"),
    )
    evidence = PersonEvidence(
        "target",
        (
            EvidenceConversation(
                platform="kakao",
                conversation_id="direct-1",
                context=EvidenceContext.KAKAO_DIRECT,
                messages=messages,
            ),
        ),
    )
    case = build_replay_cases(evidence, self_person_id="self")[0]
    assert case.action == Action.INITIATE
    assert case.session_restart is True
    assert case.action_is_ambiguous is False


def test_kakao_delay_is_interval_censored_instead_of_fake_second_precision() -> None:
    case = build_replay_cases(_direct_evidence(), self_person_id="self")[0]
    assert case.observed_delay_seconds == 120.0
    assert case.delay_lower_seconds == 60.0
    assert case.delay_upper_seconds == 180.0


def test_wait_snapshots_only_exist_when_event_cannot_already_have_happened() -> None:
    second = build_replay_cases(_direct_evidence(), self_person_id="self")[1]
    snapshots = build_action_snapshots(
        [second],
        wait_offsets_seconds=(60, 300, 600),
    )
    assert [snapshot.expected_action for snapshot in snapshots] == [
        Action.WAIT,
        Action.WAIT,
        Action.REPLY,
    ]


def test_long_gap_after_counterpart_stays_ambiguous() -> None:
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
    snapshots = build_action_snapshots([case], wait_offsets_seconds=(60, 300, 7200))

    assert case.action == Action.REPLY
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
    assert len(build_replay_cases(evidence, self_person_id="self", include_group=True)) == 3


def test_chronological_split_and_audit_keep_future_out_of_training() -> None:
    cases = build_replay_cases(_direct_evidence(), self_person_id="self")
    split = chronological_split(cases, train_fraction=0.34, validation_fraction=0.33)
    assert len(split.train) == 1
    assert len(split.validation) == 0
    assert len(split.test) == 2
    assert split.train[-1].action_at <= split.test[0].action_at

    audit = audit_replay(cases)
    assert audit.event_count == 3
    assert audit.reply_count == 2
    assert audit.follow_up_count == 1
    assert audit.initiate_count == 0
    assert audit.confident_reply_count == 2
    assert audit.confident_follow_up_count == 1
    assert audit.ambiguous_action_count == 0
    assert audit.action_snapshot_count == 3
