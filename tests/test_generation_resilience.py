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


class FailingThenWorkingModel:
    model = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def generate_burst(self, packet):
        self.calls += 1
        if self.calls == 1:
            raise TransientGenerationError("temporary 503")
        return GeneratedBurst(("살아남은 답장",))


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


def test_transient_failure_preserves_reply_and_retries_later(tmp_path) -> None:
    session = LiveChatSession(
        [_conversation()],
        self_alias="나",
        target_alias="친구",
        provider="local",
        model="fake-local",
        base_url=LOCAL_BASE_URL,
        api_key=None,
        allow_remote_private_context=False,
        store_path=tmp_path / "live.db",
    )
    model = FailingThenWorkingModel()
    session.engine.language_model = model

    sent_at = BASE + timedelta(days=1)
    event = session.send_user_message("오랜만", now=sent_at)
    original_due_at = event.due_at
    assert event.action == Action.REPLY

    failed_at = event.due_at + timedelta(seconds=1)
    try:
        session.process_due(now=failed_at)
    except TransientGenerationError as exc:
        deferred = session.defer_generation_failure(exc, now=failed_at)
    else:
        raise AssertionError("first generation should fail")

    assert deferred is not None
    assert deferred.event_id == event.event_id
    assert deferred.status == "RETRY"
    assert deferred.generation_attempts == 1
    assert deferred.due_at == original_due_at
    assert deferred.ready_at > original_due_at
    assert session.pending_event() is not None
    assert session.pending_event().event_id == event.event_id
    assert not any(message.text == "살아남은 답장" for message in session.chat_messages())

    # Before the provider retry clock is ready, the event is not due even though
    # the person's original behavior time has already passed.
    assert session.process_due(now=deferred.ready_at - timedelta(milliseconds=1)) == []

    emissions = session.process_due(now=deferred.ready_at + timedelta(seconds=1))
    assert len(emissions) == 1
    assert emissions[0].event_id == event.event_id
    assert emissions[0].due_at == original_due_at
    assert emissions[0].burst.messages == ("살아남은 답장",)
    assert any(message.text == "살아남은 답장" for message in session.chat_messages())


def test_provider_retry_does_not_see_context_after_original_behavior_time(tmp_path) -> None:
    session = LiveChatSession(
        [_conversation()],
        self_alias="나",
        target_alias="친구",
        provider="local",
        model="fake-local",
        base_url=LOCAL_BASE_URL,
        api_key=None,
        allow_remote_private_context=False,
        store_path=tmp_path / "causal-retry.db",
    )

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
    failed_at = event.due_at + timedelta(seconds=1)
    try:
        session.process_due(now=failed_at)
    except TransientGenerationError as exc:
        deferred = session.defer_generation_failure(exc, now=failed_at)
    else:
        raise AssertionError("first generation should fail")

    assert deferred is not None
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


def test_existing_pre_1_1_database_is_migrated(tmp_path) -> None:
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
    assert "generation_attempts" in columns
    assert "last_error" in columns
    assert "next_attempt_at" in columns
