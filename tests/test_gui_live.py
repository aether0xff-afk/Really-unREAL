from datetime import datetime, timedelta

from backend.generation import GeneratedBurst
from backend.gui_live import LiveChatSession
from backend.gui_support import LOCAL_BASE_URL
from backend.ingest.archive import ConversationExport
from backend.live_timing import LiveTimingSample
from backend.models import ChatMessage, MemorySource
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
    )
    return ConversationExport(
        chat_name="친구",
        source_archive="friend.zip",
        source_text="Talk.txt",
        messages=messages,
    )


class FakeModel:
    model = "fake"

    def generate_burst(self, packet):
        assert packet.chosen_action == "REPLY"
        return GeneratedBurst(("시뮬답장",))


class AlwaysReplyPolicy:
    global_reply_probability = 1.0

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
        return 60.0 if action == Action.REPLY else None


class FixedRead:
    def sample_delay_seconds(self, reply_delay_seconds):
        return min(10.0, float(reply_delay_seconds))


class InvalidStructuredTiming:
    model_name = "invalid-structured"

    def sample_timing(
        self,
        *,
        platform,
        conversation_id,
        action,
        observed_at=None,
        visible_context=(),
    ):
        return LiveTimingSample.invalid()

    def sample_delay_seconds(self, **kwargs):
        raise AssertionError("structured runtime must not fall through to scalar API")


class NoEvidenceStructuredTiming:
    model_name = "no-evidence-structured"

    def sample_timing(
        self,
        *,
        platform,
        conversation_id,
        action,
        observed_at=None,
        visible_context=(),
    ):
        return LiveTimingSample.no_evidence()

    def sample_delay_seconds(self, **kwargs):
        raise AssertionError("structured runtime must not fall through to scalar API")


def _session(tmp_path) -> LiveChatSession:
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
    session.engine.response_policy = AlwaysReplyPolicy()
    session.response_policy = session.engine.response_policy
    session.engine.timing_sampler = FixedTiming()
    session.timing_sampler = session.engine.timing_sampler
    session.engine.read_timing_model = FixedRead()
    return session


def test_live_chat_schedules_read_before_reply_and_stores_simulation_only(tmp_path) -> None:
    session = _session(tmp_path)
    session.engine.language_model = FakeModel()

    now = BASE + timedelta(days=1)
    reply = session.send_user_message("지금 뭐함", now=now)
    assert reply is not None
    assert reply.action == Action.REPLY
    assert reply.due_at == now + timedelta(seconds=60)

    read = next(event for event in session.pending_events() if event.action == Action.READ)
    assert read.due_at == now + timedelta(seconds=10)
    assert read.due_at < reply.due_at

    user = session.chat_messages()[-1]
    assert user.sender_person_id == "self"
    assert user.text == "지금 뭐함"
    assert user.read_at is None
    stored = session.store.simulation_messages(
        twin_person_id=session.target_person_id,
        platform="kakao",
        conversation_id=session.conversation_id,
    )
    assert stored[-1].source == MemorySource.SIMULATION
    assert stored[-1].metadata["read_status"] == "UNREAD"

    assert session.process_due(now=read.due_at + timedelta(milliseconds=1)) == []
    assert session.chat_messages()[0].read_at == read.due_at
    assert not any(message.text == "시뮬답장" for message in session.chat_messages())

    emissions = session.process_due(now=reply.due_at + timedelta(milliseconds=1))
    assert len(emissions) == 1
    assert emissions[0].burst.messages == ("시뮬답장",)
    assert session.chat_messages()[-1].text == "시뮬답장"


def test_read_receipt_is_latent_behavior_time_not_generation_time(tmp_path) -> None:
    session = _session(tmp_path)
    session.engine.language_model = FakeModel()
    sent_at = BASE + timedelta(days=2)
    reply = session.send_user_message("읽었냐", now=sent_at)
    assert reply is not None
    read = next(event for event in session.pending_events() if event.action == Action.READ)

    much_later = reply.due_at + timedelta(minutes=10)
    session.process_due(now=much_later)
    user_message = next(
        message for message in session.chat_messages() if message.sender_person_id == "self"
    )
    assert user_message.read_at == read.due_at
    assert user_message.read_at != reply.due_at
    assert user_message.read_at != much_later


def test_rapid_user_bubbles_keep_one_reply_clock_instead_of_resampling(tmp_path) -> None:
    session = _session(tmp_path)
    first_at = BASE + timedelta(days=3)
    first = session.send_user_message("야", now=first_at)
    assert first is not None
    second = session.send_user_message("뭐함", now=first_at + timedelta(seconds=2))
    assert second is not None
    third = session.send_user_message("ㅋㅋ", now=first_at + timedelta(seconds=4))
    assert third is not None

    assert first.event_id == second.event_id == third.event_id
    # Only input-settle can extend the original clock; timing is not re-sampled.
    assert third.due_at == first.due_at
    reply_events = [event for event in session.pending_events() if event.action == Action.REPLY]
    assert len(reply_events) == 1


def test_new_bubble_after_first_read_gets_its_own_read_before_existing_reply(tmp_path) -> None:
    session = _session(tmp_path)
    first_at = BASE + timedelta(days=4)
    reply = session.send_user_message("첫 말", now=first_at)
    assert reply is not None
    first_read = next(event for event in session.pending_events() if event.action == Action.READ)
    session.process_due(now=first_read.due_at + timedelta(milliseconds=1))
    assert session.chat_messages()[0].read_at == first_read.due_at

    second_at = first_read.due_at + timedelta(seconds=5)
    same_reply = session.send_user_message("추가 말", now=second_at)
    assert same_reply is not None and same_reply.event_id == reply.event_id
    second_message = session.chat_messages()[-1]
    assert second_message.read_at is None

    pending_reads = [event for event in session.pending_events() if event.action == Action.READ]
    assert len(pending_reads) == 1
    assert second_at <= pending_reads[0].due_at <= reply.due_at
    session.process_due(now=pending_reads[0].due_at + timedelta(milliseconds=1))
    second_message = next(message for message in session.chat_messages() if message.text == "추가 말")
    assert second_message.read_at == pending_reads[0].due_at


def test_old_simulation_replies_backfill_read_receipts(tmp_path) -> None:
    session = _session(tmp_path)
    sent_at = BASE + timedelta(days=5)
    reply_at = sent_at + timedelta(minutes=4)
    session.store.append_simulation_messages(
        twin_person_id=session.target_person_id,
        platform="kakao",
        conversation_id=session.conversation_id,
        sender_person_id="self",
        messages=((sent_at, "옛날 메시지"),),
        metadata={"role": "user", "ui_live": True},
    )
    session.store.append_simulation_messages(
        twin_person_id=session.target_person_id,
        platform="kakao",
        conversation_id=session.conversation_id,
        sender_person_id=session.target_person_id,
        messages=((reply_at, "옛날 답장"),),
        metadata={"action": "REPLY"},
    )

    reopened = _session(tmp_path)
    user_message = next(message for message in reopened.chat_messages() if message.text == "옛날 메시지")
    assert user_message.read_at == reply_at


def test_live_chat_reset_never_touches_real_evidence(tmp_path) -> None:
    original = _conversation()
    session = _session(tmp_path)
    session.send_user_message("테스트", now=BASE + timedelta(days=6))
    assert session.chat_messages()
    session.reset()
    assert session.chat_messages() == []
    assert original.messages[-1].text == "ㅇㅇ"


def test_structured_invalid_timing_cannot_be_resurrected_by_baseline(tmp_path) -> None:
    session = _session(tmp_path)
    session.engine.timing_sampler = InvalidStructuredTiming()
    session.engine.read_timing_model = None

    now = BASE + timedelta(days=7)
    reply = session.send_user_message("이 행동은 timing gate에서 불가능", now=now)
    assert reply is None
    assert not [event for event in session.pending_events() if event.action == Action.REPLY]


def test_only_no_evidence_structured_timing_uses_empirical_baseline(tmp_path) -> None:
    session = _session(tmp_path)
    session.engine.timing_sampler = NoEvidenceStructuredTiming()
    session.engine.read_timing_model = None

    now = BASE + timedelta(days=8)
    reply = session.send_user_message("근거 부족이면 baseline", now=now)
    assert reply is not None
    assert reply.action == Action.REPLY
    assert reply.due_at > now
