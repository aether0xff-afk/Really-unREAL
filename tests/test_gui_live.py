from datetime import datetime, timedelta

from backend.generation import GeneratedBurst
from backend.gui_live import LiveChatSession
from backend.gui_support import LOCAL_BASE_URL
from backend.ingest.archive import ConversationExport
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
        assert packet.chosen_action.value == "REPLY"
        return GeneratedBurst(("시뮬답장",))


def test_live_chat_schedules_before_generation_and_stores_simulation_only(tmp_path) -> None:
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
    session.engine.language_model = FakeModel()

    now = BASE + timedelta(days=1)
    scheduled = session.send_user_message("지금 뭐함", now=now)

    assert scheduled.action == Action.REPLY
    assert scheduled.due_at >= now
    assert session.chat_messages()[-1].sender_person_id == "self"
    assert session.chat_messages()[-1].text == "지금 뭐함"
    stored = session.store.simulation_messages(
        twin_person_id=session.target_person_id,
        platform="kakao",
        conversation_id=session.conversation_id,
    )
    assert stored[-1].source == MemorySource.SIMULATION

    emissions = session.process_due(now=scheduled.due_at + timedelta(seconds=1))
    assert len(emissions) == 1
    assert emissions[0].burst.messages == ("시뮬답장",)
    assert session.chat_messages()[-1].text == "시뮬답장"


def test_live_chat_reset_never_touches_real_evidence(tmp_path) -> None:
    original = _conversation()
    session = LiveChatSession(
        [original],
        self_alias="나",
        target_alias="친구",
        provider="local",
        model="fake-local",
        base_url=LOCAL_BASE_URL,
        api_key=None,
        allow_remote_private_context=False,
        store_path=tmp_path / "live.db",
    )
    session.send_user_message("테스트", now=BASE + timedelta(days=1))
    assert session.chat_messages()

    session.reset()

    assert session.chat_messages() == []
    assert original.messages[-1].text == "ㅇㅇ"
