from datetime import datetime, timedelta
import urllib.error

from backend.fusion import EvidenceContext, EvidenceMessage
from backend.generation import GeneratedBurst
from backend.gui_live import LiveChatSession
from backend.gui_support import LOCAL_BASE_URL
from backend.ingest.archive import ConversationExport
from backend.models import ChatMessage
from backend.providers.errors import TransientGenerationError
from backend.providers.nvidia import NvidiaNIMLanguageModel
from backend.simulation.action_policy import Action


BASE = datetime(2026, 8, 8, 12, 0)


def _conversation() -> ConversationExport:
    messages = (
        ChatMessage(BASE, "나", "안녕"),
        ChatMessage(BASE + timedelta(minutes=1), "친구", "ㅎㅇ"),
        ChatMessage(BASE + timedelta(minutes=2), "나", "뭐함"),
        ChatMessage(BASE + timedelta(minutes=3), "친구", "집"),
        ChatMessage(BASE + timedelta(minutes=4), "나", "낼 감?"),
        ChatMessage(BASE + timedelta(minutes=5), "친구", "ㅇㅇ"),
        ChatMessage(BASE + timedelta(minutes=6), "나", "몇시"),
        ChatMessage(BASE + timedelta(minutes=8), "친구", "10시"),
    )
    return ConversationExport(
        chat_name="친구",
        source_archive="friend.zip",
        source_text="Talk.txt",
        messages=messages,
    )


class AlwaysReplyPolicy:
    def choose_after_counterpart_message(self, *, observed_at, visible_context):
        return Action.REPLY


class FixedTiming:
    model_name = "fixed"

    def sample_delay_seconds(
        self,
        *,
        platform,
        conversation_id,
        action,
        observed_at=None,
        visible_context=(),
    ):
        if action == Action.REPLY:
            return 30.0
        return None


class FailingThenWorkingModel:
    model = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def generate_burst(self, packet):
        self.calls += 1
        if self.calls == 1:
            raise TransientGenerationError("temporary 503")
        return GeneratedBurst(("살아남은 답장",))


def _session(tmp_path, name="live.db") -> LiveChatSession:
    session = LiveChatSession(
        [_conversation()],
        self_alias="나",
        target_alias="친구",
        provider="local",
        model="fake-local",
        base_url=LOCAL_BASE_URL,
        api_key=None,
        allow_remote_private_context=False,
        store_path=tmp_path / name,
    )
    session.engine.response_policy = AlwaysReplyPolicy()
    session.engine.timing_sampler = FixedTiming()
    return session


def test_nvidia_503_is_classified_as_transient() -> None:
    def transport(url, headers, payload, timeout):
        raise urllib.error.HTTPError(url, 503, "unavailable", {}, None)

    model = NvidiaNIMLanguageModel(
        api_key="test-key",
        transport=transport,
        max_attempts=1,
    )

    from tests.test_nvidia_provider import _packet

    try:
        model.generate_burst(_packet())
    except TransientGenerationError as exc:
        assert "503" in str(exc)
    else:
        raise AssertionError("HTTP 503 must be transient")


def test_transient_failure_is_persisted_by_runtime_and_preserves_behavior_time(tmp_path) -> None:
    session = _session(tmp_path)
    model = FailingThenWorkingModel()
    session.engine.language_model = model

    sent_at = BASE + timedelta(days=1)
    event = session.send_user_message("오랜만", now=sent_at)
    assert event is not None
    original_due_at = event.due_at
    assert event.action == Action.REPLY

    failed_at = event.due_at + timedelta(seconds=1)
    # Runtime handles the provider failure itself; GUI code must not guess which
    # pending event failed after the fact.
    assert session.process_due(now=failed_at) == []

    deferred = session.store.event(event.event_id)
    assert deferred is not None
    assert deferred.status == "RETRY"
    assert deferred.generation_attempts == 1
    assert deferred.due_at == original_due_at
    assert deferred.ready_at > original_due_at
    assert not any(message.text == "살아남은 답장" for message in session.chat_messages())

    assert session.process_due(now=deferred.ready_at - timedelta(milliseconds=1)) == []
    emissions = session.process_due(now=deferred.ready_at + timedelta(seconds=1))
    assert len(emissions) == 1
    assert emissions[0].event_id == event.event_id
    assert emissions[0].due_at == original_due_at
    assert emissions[0].burst.messages == ("살아남은 답장",)
    assert session.store.event(event.event_id).status == "PROCESSED"


def test_provider_retry_does_not_see_context_after_original_behavior_time(tmp_path) -> None:
    session = _session(tmp_path, "causal-retry.db")

    class InspectRetryModel:
        model = "inspect"

        def __init__(self) -> None:
            self.calls = 0

        def generate_burst(self, packet):
            self.calls += 1
            if self.calls == 1:
                raise TransientGenerationError("temporary")
            assert "AFTER_ORIGINAL_DUE" not in str(packet.to_dict())
            return GeneratedBurst(("ok",))

    model = InspectRetryModel()
    session.engine.language_model = model
    event = session.send_user_message("질문", now=BASE + timedelta(days=1))
    assert event is not None
    failed_at = event.due_at + timedelta(seconds=1)
    session.process_due(now=failed_at)
    deferred = session.store.event(event.event_id)
    assert deferred is not None and deferred.status == "RETRY"

    late = EvidenceMessage(
        message=ChatMessage(
            event.due_at + timedelta(seconds=2),
            "self",
            "AFTER_ORIGINAL_DUE",
        ),
        platform="kakao",
        conversation_id=session.conversation_id,
        context=EvidenceContext.KAKAO_DIRECT,
        sender_person_id="self",
        evidence_weight=1.0,
    )
    visible = session._visible_context() + (late,)
    emissions = session.engine.process_due(
        now=deferred.ready_at + timedelta(seconds=1),
        visible_context=visible,
    )
    assert emissions[0].due_at == event.due_at
    assert emissions[0].burst.messages == ("ok",)


def test_new_user_message_during_claim_does_not_cancel_or_block_wrong_reply(tmp_path) -> None:
    session = _session(tmp_path, "race.db")
    first_at = BASE + timedelta(days=1)
    first = session.send_user_message("첫 질문", now=first_at)
    assert first is not None

    claimed = session.store.claim_due_events(
        now=first.due_at + timedelta(seconds=1),
        twin_person_id=session.target_person_id,
        platform="kakao",
        conversation_id=session.conversation_id,
    )
    claimed_reply = next(item for item in claimed if item.action == Action.REPLY)
    assert claimed_reply.status == "CLAIMED"

    second = session.send_user_message("추가 질문", now=first.due_at + timedelta(seconds=2))
    assert second is not None
    assert second.event_id != first.event_id
    assert session.store.event(first.event_id).status == "CLAIMED"
    assert session.store.event(second.event_id).status == "PENDING"

    # A failure belongs to the exact claimed event, not whichever event happens
    # to be first in pending-event ordering now.
    session.store.block_event(first.event_id, error="old generation failed")
    assert session.store.event(first.event_id).status == "BLOCKED"
    assert session.store.event(second.event_id).status == "PENDING"


def test_existing_pre_1_1_database_is_migrated_through_v1_2_claim_columns(tmp_path) -> None:
    import sqlite3

    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as db:
        db.executescript(
            """
            CREATE TABLE scheduled_events (
                event_id TEXT PRIMARY KEY,
                twin_person_id TEXT NOT NULL,
                platform TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                action TEXT NOT NULL,
                due_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'PENDING'
            );
            """
        )

    from backend.simulation.store import SQLiteSimulationStore

    SQLiteSimulationStore(path)
    with sqlite3.connect(path) as db:
        columns = {row[1] for row in db.execute("PRAGMA table_info(scheduled_events)")}
    assert {
        "generation_attempts",
        "last_error",
        "next_attempt_at",
        "claim_token",
        "claimed_at",
    } <= columns
