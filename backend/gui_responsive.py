from __future__ import annotations

import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

from backend.gui import ReallyUnrealApp
from backend.gui_chat_window import LiveChatWindow
from backend.gui_live import LiveChatSession
from backend.gui_runtime import run_quick_generation_interactive
from backend.gui_support import run_quick_audit


def _percent(value: object) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "-"


def _ci_text(value: object) -> str:
    if not isinstance(value, dict):
        return ""
    lower = value.get("lower")
    upper = value.get("upper")
    if lower is None or upper is None:
        return ""
    return f" (95% 구간 {_percent(lower)}~{_percent(upper)})"


def _friendly_result(result: dict[str, object]) -> str:
    if "audit" in result:
        lines = [
            "빠른 진단 완료",
            "",
            f"대화 상대        {result.get('target', '-')}",
            f"분석된 1:1 대화  {result.get('evidence_conversations', 0)}개",
            f"상대 메시지      {result.get('target_messages', 0)}개",
            f"재현 테스트 후보 {result.get('replay_cases', 0)}개",
        ]
        split = result.get("split")
        if isinstance(split, dict):
            lines.extend(
                [
                    "",
                    "시간순 데이터 분리",
                    f"학습 {split.get('train', 0)} · 검증 {split.get('validation', 0)} · 테스트 {split.get('test', 0)}",
                ]
            )
        selection = result.get("temporal_selection")
        if isinstance(selection, dict):
            selected = selection.get("selected_model", "-")
            friendly = "상황 기반" if selected == "hazard" else "관계별 경험 분포"
            lines.extend(["", f"선택된 답장 시간 모델  {friendly}"])
        warning = result.get("warning")
        if warning:
            lines.extend(["", f"주의: {warning}"])
        lines.extend(
            [
                "",
                "이 진단은 메시지를 생성하지 않습니다.",
                "데이터/시간 모델이 정상적으로 만들어지는지만 확인합니다.",
            ]
        )
        return "\n".join(lines)

    generated = int(result.get("generated_cases", 0) or 0)
    requested = int(result.get("requested_cases", generated) or generated)
    failed = int(result.get("failed_cases", 0) or 0)
    elapsed = float(result.get("elapsed_seconds", 0.0) or 0.0)
    burst_error = result.get("mean_burst_size_absolute_error")
    char_f1 = result.get("mean_char_bigram_f1")
    token_f1 = result.get("mean_token_f1")
    ending_f1 = result.get("mean_ending_f1")
    laugh = result.get("laugh_presence_match_rate")
    cry = result.get("cry_presence_match_rate")
    question = result.get("question_presence_match_rate")
    timing_rate = result.get("timing_inside_rate")
    sample_note = result.get("evaluation_sample_note")

    lines = [
        "모델 성능 테스트 완료",
        "",
        f"생성 성공        {generated} / {requested}",
        f"실패             {failed}",
        f"소요 시간        {elapsed:.1f}초",
        "",
        "내용/문장 재현",
        f"글자 패턴 유사도 {_percent(char_f1)}{_ci_text(result.get('mean_char_bigram_f1_ci95'))}",
        f"단어 겹침        {_percent(token_f1)}{_ci_text(result.get('mean_token_f1_ci95'))}",
        f"말끝 형태        {_percent(ending_f1)}{_ci_text(result.get('mean_ending_f1_ci95'))}",
        "",
        "표현 행동",
        f"웃음 표현 일치   {_percent(laugh)}",
        f"울음 표현 일치   {_percent(cry)}",
        f"질문 여부 일치   {_percent(question)}",
        f"답장 타이밍 범위 {_percent(timing_rate)}{_ci_text(result.get('timing_inside_rate_ci95'))}",
    ]
    if burst_error is not None:
        lines.append(f"메시지 분할 오차 평균 {float(burst_error):.2f}개")
    if sample_note:
        lines.extend(["", f"표본 해석: {sample_note}"])
    lines.extend(
        [
            "",
            "이 화면은 '대화'가 아니라 과거 실제 답을 숨긴 재현 시험입니다.",
            "NVIDIA 기본 3개는 빠른 작동 확인용입니다. 비교하려면 테스트 수를 10~20 이상으로 올리세요.",
            "실제로 시뮬레이션과 대화하려면 '대화 시작'을 누르세요.",
            "세부 수치는 '결과 저장'으로 JSON에 저장할 수 있습니다.",
        ]
    )
    return "\n".join(lines)


class ResponsiveReallyUnrealApp(ReallyUnrealApp):
    """Desktop shell with cancellable evaluation and persistent live chat."""

    def __init__(self) -> None:
        self._cancel_event = threading.Event()
        self._run_started_at: float | None = None
        self._progress_stage = ""
        self._progress_completed = 0
        self._progress_total = 0
        self._progress_failed = 0
        self._responsive_last_mode: str | None = None
        super().__init__()

        self._rename_labels(self)
        controls = self.run_button.master
        self.run_button.configure(text="빠른 진단")
        self.cancel_button = ttk.Button(
            controls,
            text="취소",
            command=self._cancel_run,
            state="disabled",
        )
        self.cancel_button.pack(side="left", padx=(8, 0))

        self.chat_button = ttk.Button(
            controls,
            text="대화 시작",
            command=self._open_chat,
            state="disabled",
        )
        self.chat_button.pack(side="left", padx=(8, 0))

        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(
            controls,
            variable=self.progress_var,
            maximum=100.0,
            length=150,
        )
        self.progress_bar.pack(side="left", padx=(14, 0))

        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert(
            "1.0",
            "1) 빠른 진단: 데이터와 답장 시간 모델 확인\n"
            "2) 모델 성능 테스트: 과거 실제 답을 숨기고 재현 점수 측정\n"
            "   · NVIDIA 기본 3개 = 빠른 작동 확인, 비교용은 10~20개 이상 권장\n"
            "3) 대화 시작: 현재 시점에서 실제 시간 흐름으로 시뮬레이션 대화\n",
        )
        self.output.configure(state="disabled")
        self._sync_provider_defaults()

    def _rename_labels(self, widget: tk.Misc) -> None:
        for child in widget.winfo_children():
            if isinstance(child, ttk.Radiobutton):
                text = str(child.cget("text"))
                if text == "빠른 Audit (LLM 없음)":
                    child.configure(text="빠른 진단 (LLM 없음)")
                elif text == "Local LLM":
                    child.configure(text="모델 테스트 · Local LLM")
                elif text == "NVIDIA NIM":
                    child.configure(text="모델 테스트 · NVIDIA NIM")
            elif isinstance(child, ttk.Label) and str(child.cget("text")) == "Cases":
                child.configure(text="테스트 수")
            self._rename_labels(child)

    def _set_busy(self, busy: bool, status: str) -> None:
        super()._set_busy(busy, status)
        if hasattr(self, "chat_button"):
            enabled = (not busy) and self.mode_var.get() in {"local", "nvidia"}
            self.chat_button.configure(state="normal" if enabled else "disabled")

    def _sync_provider_defaults(self) -> None:
        previous = getattr(self, "_responsive_last_mode", None)
        super()._sync_provider_defaults()
        current = self.mode_var.get()
        if current == "nvidia" and previous != "nvidia" and self.limit_var.get() == 10:
            self.limit_var.set(3)
        if hasattr(self, "run_button"):
            self.run_button.configure(
                text="빠른 진단" if current == "audit" else "모델 테스트"
            )
        if hasattr(self, "chat_button"):
            self.chat_button.configure(
                state="normal" if current in {"local", "nvidia"} else "disabled"
            )
        self._responsive_last_mode = current

    def _run(self) -> None:
        if not self._conversations:
            messagebox.showinfo("Really-unREAL", "먼저 카카오톡 ZIP을 불러오세요.")
            return

        self_alias = self.self_var.get().strip()
        target_alias = self.target_var.get().strip()
        if not self_alias or not target_alias:
            messagebox.showinfo("Really-unREAL", "내 이름과 대화 상대를 선택하세요.")
            return

        mode = self.mode_var.get()
        model = self.model_var.get().strip()
        base_url = self.base_url_var.get().strip()
        api_key = self.api_key_var.get().strip() or None
        allow_remote = self.remote_consent_var.get()
        limit = max(1, int(self.limit_var.get()))

        self._cancel_event.clear()
        self._run_started_at = time.monotonic()
        self._progress_stage = "준비 중"
        self._progress_completed = 0
        self._progress_total = limit if mode != "audit" else 1
        self._progress_failed = 0
        self.progress_var.set(0.0)
        self.cancel_button.configure(state="normal" if mode != "audit" else "disabled")
        self._set_busy(True, "준비 중…")
        self.after(250, self._tick_elapsed)

        def progress(completed: int, total: int, failed: int, stage: str) -> None:
            self.after(
                0,
                lambda c=completed, t=total, f=failed, s=stage: self._on_progress(c, t, f, s),
            )

        def work() -> None:
            try:
                if mode == "audit":
                    result = run_quick_audit(
                        self._conversations,
                        self_alias=self_alias,
                        target_alias=target_alias,
                    )
                else:
                    result = run_quick_generation_interactive(
                        self._conversations,
                        self_alias=self_alias,
                        target_alias=target_alias,
                        provider=mode,
                        model=model,
                        base_url=base_url,
                        api_key=api_key,
                        allow_remote_private_context=allow_remote,
                        limit=limit,
                        progress=progress,
                        is_cancelled=self._cancel_event.is_set,
                    )
                self.after(0, lambda: self._finish_run(result))
            except Exception as exc:
                self.after(0, lambda: self._show_error(exc))

        threading.Thread(target=work, daemon=True).start()

    def _open_chat(self) -> None:
        if not self._conversations:
            messagebox.showinfo("Really-unREAL", "먼저 카카오톡 ZIP을 불러오세요.")
            return
        mode = self.mode_var.get()
        if mode not in {"local", "nvidia"}:
            messagebox.showinfo(
                "Really-unREAL",
                "대화에 사용할 Local LLM 또는 NVIDIA NIM을 먼저 선택하세요.",
            )
            return
        self_alias = self.self_var.get().strip()
        target_alias = self.target_var.get().strip()
        if not self_alias or not target_alias:
            messagebox.showinfo("Really-unREAL", "내 이름과 대화 상대를 선택하세요.")
            return

        self._set_busy(True, "대화 준비 중…")
        model = self.model_var.get().strip()
        base_url = self.base_url_var.get().strip()
        api_key = self.api_key_var.get().strip() or None
        allow_remote = self.remote_consent_var.get()

        def work() -> None:
            try:
                session = LiveChatSession(
                    self._conversations,
                    self_alias=self_alias,
                    target_alias=target_alias,
                    provider=mode,
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                    allow_remote_private_context=allow_remote,
                )
            except Exception as exc:
                self.after(0, lambda: self._show_error(exc))
                return
            self.after(0, lambda: self._chat_ready(session))

        threading.Thread(target=work, daemon=True).start()

    def _chat_ready(self, session: LiveChatSession) -> None:
        self._set_busy(False, f"{session.target_alias} 시뮬레이션 대화 준비됨")
        LiveChatWindow(self, session)

    def _on_progress(self, completed: int, total: int, failed: int, stage: str) -> None:
        self._progress_completed = completed
        self._progress_total = total
        self._progress_failed = failed
        self._progress_stage = stage
        percent = 0.0 if total <= 0 else min(100.0, completed * 100.0 / total)
        self.progress_var.set(percent)
        self._render_progress_status()

    def _render_progress_status(self) -> None:
        if self._run_started_at is None:
            return
        elapsed = int(time.monotonic() - self._run_started_at)
        minutes, seconds = divmod(elapsed, 60)
        elapsed_text = f"{minutes}:{seconds:02d}"
        failed_text = f" · 실패 {self._progress_failed}" if self._progress_failed else ""
        self.status_var.set(f"{self._progress_stage}{failed_text} · {elapsed_text}")

    def _tick_elapsed(self) -> None:
        if self._run_started_at is None:
            return
        self._render_progress_status()
        self.after(500, self._tick_elapsed)

    def _cancel_run(self) -> None:
        if self._run_started_at is None:
            return
        self._cancel_event.set()
        self.cancel_button.configure(state="disabled")
        self._progress_stage = "취소 요청됨 · 현재 모델 요청 종료 대기 (최대 약 45초)"
        self._render_progress_status()

    def _finish_run(self, result: dict[str, object]) -> None:
        cancelled = bool(result.get("cancelled"))
        failed = int(result.get("failed_cases", 0) or 0)
        generated = int(result.get("generated_cases", 0) or 0)
        requested = int(result.get("requested_cases", generated) or generated)
        self._last_result = result
        text = _friendly_result(result)
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("1.0", text)
        self.output.configure(state="disabled")
        self._run_started_at = None
        self.cancel_button.configure(state="disabled")
        self.progress_var.set(
            100.0 if not cancelled else (generated * 100.0 / requested if requested else 0.0)
        )
        self._set_busy(False, "완료")
        if cancelled:
            self.status_var.set(f"취소됨 · {generated}/{requested} 완료 · 실패 {failed}")
        elif "audit" in result:
            self.status_var.set("빠른 진단 완료")
        elif failed:
            self.status_var.set(f"모델 테스트 완료 · {generated}/{requested} 성공 · 실패 {failed}")
        else:
            self.status_var.set(f"모델 테스트 완료 · {generated}/{requested}")

    def _show_error(self, exc: Exception) -> None:
        self._run_started_at = None
        if hasattr(self, "cancel_button"):
            self.cancel_button.configure(state="disabled")
        super()._show_error(exc)
