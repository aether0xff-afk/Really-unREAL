from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from backend.fusion import EvidenceContext, EvidenceMessage
from backend.gui_runtime import (
    NVIDIA_GUI_FORMAT_ATTEMPTS,
    NVIDIA_GUI_MAX_ATTEMPTS,
    NVIDIA_GUI_TIMEOUT_SECONDS,
)
from backend.gui_support import (
    LOCAL_BASE_URL,
    NVIDIA_BASE_URL,
    NVIDIA_MODEL,
    _target_evidence,
)
from backend.ingest.archive import ConversationExport
from backend.privacy import require_private_context_route
from backend.providers.nvidia import NvidiaNIMLanguageModel
from backend.providers.openai_compatible import OpenAICompatibleLanguageModel
from backend.replay import build_replay_cases
from backend.replay_baseline import EmpiricalTimingBaseline
from backend.replay_sampling import EmpiricalTimingSampler
from backend.retrieval import CutoffExampleIndex
from backend.simulation.action_policy import Action
from backend.simulation.runtime import LiveSimulationEngine, SimulationEmission
from backend.simulation.store import SQLiteSimulationStore, ScheduledEvent


@dataclass(frozen=True, slots=True)
class LiveChatMessage:
    timestamp: datetime
    sender_person_id: str
    text: str


def default_live_store_path() -> Path:
    root = os.environ.get("LOCALAPPDATA")
    base = Path(root) / "Really-unREAL" if root else Path.home() / ".really-unreal"
    return base / "live-simulation.db"


def _language_model(
    *,
    provider: str,
    model: str,
    base_url: str,
    api_key: str | None,
):
    if provider == "nvidia":
        return NvidiaNIMLanguageModel(
            api_key=api_key or None,
            model=model or NVIDIA_MODEL,
            base_url=base_url or NVIDIA_BASE_URL,
            timeout_seconds=NVIDIA_GUI_TIMEOUT_SECONDS,
            max_attempts=NVIDIA_GUI_MAX_ATTEMPTS,
            max_format_attempts=NVIDIA_GUI_FORMAT_ATTEMPTS,
        )
    if provider == "local":
        if not model.strip():
            raise ValueError("로컬 모델 이름을 입력하세요.")
        return OpenAICompatibleLanguageModel(
            model=model.strip(),
            base_url=base_url or LOCAL_BASE_URL,
            api_key=api_key or None,
            timeout_seconds=60.0,
            max_attempts=1,
            format_attempts=1,
        )
    raise ValueError("대화 시작에는 Local LLM 또는 NVIDIA NIM을 선택하세요.")


class LiveChatSession:
    """Desktop adapter around the persistent LiveSimulationEngine."""

    def __init__(
        self,
        conversations: list[ConversationExport],
        *,
        self_alias: str,
        target_alias: str,
        provider: str,
        model: str,
        base_url: str,
        api_key: str | None,
        allow_remote_private_context: bool,
        store_path: str | Path | None = None,
    ) -> None:
        require_private_context_route(
            base_url,
            allow_remote_private_context=allow_remote_private_context,
        )
        _, target_id, evidence = _target_evidence(conversations, self_alias, target_alias)
        direct = [
            conversation
            for conversation in evidence.conversations
            if conversation.platform == "kakao"
            and conversation.context == EvidenceContext.KAKAO_DIRECT
        ]
        if not direct:
            raise ValueError("선택한 상대와의 1:1 카카오톡 대화를 찾지 못했습니다.")
        selected = max(
            direct,
            key=lambda conversation: (
                conversation.messages[-1].message.timestamp
                if conversation.messages
                else datetime.min
            ),
        )

        cases = build_replay_cases(evidence, self_person_id="self")
        if not cases:
            raise ValueError("답장 시간을 학습할 수 있는 과거 대화가 부족합니다.")
        timing = EmpiricalTimingBaseline.fit(cases)
        timing_sampler = EmpiricalTimingSampler(cases)
        index = CutoffExampleIndex.from_replay_cases(cases)
        language_model = _language_model(
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
        )
        store = SQLiteSimulationStore(store_path or default_live_store_path())

        self.self_alias = self_alias
        self.target_alias = target_alias
        self.target_person_id = target_id
        self.provider = provider
        self.evidence = evidence
        self.historical_conversation = selected
        self.store = store
        self.engine = LiveSimulationEngine(
            twin_person_id=target_id,
            platform="kakao",
            conversation_id=selected.conversation_id,
            evidence=evidence,
            retrieval_index=index,
            timing=timing,
            timing_sampler=timing_sampler,
            language_model=language_model,
            store=store,
        )

    @property
    def conversation_id(self) -> str:
        return self.historical_conversation.conversation_id

    def _simulation_messages(self):
        return self.store.simulation_messages(
            twin_person_id=self.target_person_id,
            platform="kakao",
            conversation_id=self.conversation_id,
        )

    def chat_messages(self) -> list[LiveChatMessage]:
        return [
            LiveChatMessage(message.timestamp, message.sender, message.text)
            for message in self._simulation_messages()
        ]

    def _visible_context(self) -> tuple[EvidenceMessage, ...]:
        historical = tuple(self.historical_conversation.messages[-40:])
        simulated = tuple(
            EvidenceMessage(
                message=message,
                platform="kakao",
                conversation_id=self.conversation_id,
                context=EvidenceContext.KAKAO_DIRECT,
                sender_person_id=message.sender,
                evidence_weight=1.0,
            )
            for message in self._simulation_messages()[-40:]
        )
        return historical + simulated

    def send_user_message(self, text: str, *, now: datetime | None = None) -> ScheduledEvent:
        text = text.strip()
        if not text:
            raise ValueError("메시지를 입력하세요.")
        now = now or datetime.now()
        self.store.append_simulation_messages(
            twin_person_id=self.target_person_id,
            platform="kakao",
            conversation_id=self.conversation_id,
            sender_person_id="self",
            messages=((now, text),),
            metadata={"role": "user", "ui_live": True},
        )
        return self.engine.observe_counterpart_message(observed_at=now)

    def process_due(self, *, now: datetime | None = None) -> list[SimulationEmission]:
        now = now or datetime.now()
        return self.engine.process_due(now=now, visible_context=self._visible_context())

    def recover(self, *, now: datetime | None = None) -> list[SimulationEmission]:
        now = now or datetime.now()
        return self.engine.recover(now=now, visible_context=self._visible_context())

    def pending_event(self) -> ScheduledEvent | None:
        matching = [
            event
            for event in self.store.pending_events()
            if event.twin_person_id == self.target_person_id
            and event.platform == "kakao"
            and event.conversation_id == self.conversation_id
        ]
        return min(matching, key=lambda event: event.due_at) if matching else None

    def ensure_idle_initiation(self, *, now: datetime | None = None) -> ScheduledEvent | None:
        if self.pending_event() is not None:
            return self.pending_event()
        now = now or datetime.now()
        messages = self.chat_messages()
        after = messages[-1].timestamp if messages else now
        return self.engine.schedule_idle_initiation(after=after)

    def defer_generation_failure(
        self,
        error: Exception,
        *,
        now: datetime | None = None,
    ) -> ScheduledEvent | None:
        event = self.pending_event()
        if event is None:
            return None
        now = now or datetime.now()
        delays = (5, 15, 30, 60, 120, 300)
        retry_seconds = delays[min(event.generation_attempts, len(delays) - 1)]
        return self.store.defer_event(
            event.event_id,
            retry_at=now + timedelta(seconds=retry_seconds),
            error=str(error),
        )

    def block_generation_failure(self, error: Exception) -> ScheduledEvent | None:
        event = self.pending_event()
        if event is None:
            return None
        return self.store.block_event(event.event_id, error=str(error))

    def retry_blocked(self, *, now: datetime | None = None) -> ScheduledEvent | None:
        event = self.pending_event()
        if event is None or event.status != "BLOCKED":
            return event
        return self.store.retry_blocked_event(
            event.event_id,
            retry_at=now or datetime.now(),
        )

    def cancel_pending(self) -> None:
        self.store.cancel_pending(
            twin_person_id=self.target_person_id,
            platform="kakao",
            conversation_id=self.conversation_id,
        )

    def reset(self) -> None:
        self.store.clear_conversation(
            twin_person_id=self.target_person_id,
            platform="kakao",
            conversation_id=self.conversation_id,
        )

    def pending_label(self, *, now: datetime | None = None) -> str:
        event = self.pending_event()
        if event is None:
            return "대기 중"
        if event.status == "BLOCKED":
            return "답장 행동 보존됨 · 모델 설정 확인 후 재시도"
        now = now or datetime.now()
        remaining = max(0, int((event.due_at - now).total_seconds()))
        if event.action == Action.REPLY:
            prefix = "답장 예정"
        else:
            prefix = "먼저 메시지 가능성"
        if event.status == "RETRY":
            prefix += " · 생성 재시도"
        if remaining < 60:
            return f"{prefix} · 약 {remaining}초 후"
        minutes = remaining // 60
        if minutes < 60:
            return f"{prefix} · 약 {minutes}분 후"
        hours = minutes // 60
        return f"{prefix} · 약 {hours}시간 후"
