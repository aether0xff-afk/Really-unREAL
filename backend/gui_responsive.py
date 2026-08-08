from __future__ import annotations

import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

from backend.gui import ReallyUnrealApp
from backend.gui_runtime import run_quick_generation_interactive
from backend.gui_support import run_quick_audit


class ResponsiveReallyUnrealApp(ReallyUnrealApp):
    """v1.0.3 desktop shell with observable, cancellable generation runs."""

    def __init__(self) -> None:
        self._cancel_event = threading.Event()
        self._run_started_at: float | None = None
        self._progress_stage = ""
        self._progress_completed = 0
        self._progress_total = 0
        self._progress_failed = 0
        self._responsive_last_mode: str | None = None
        super().__init__()

        controls = self.run_button.master
        self.cancel_button = ttk.Button(
            controls,
            text="취소",
            command=self._cancel_run,
            state="disabled",
        )
        self.cancel_button.pack(side="left", padx=(8, 0))

        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(
            controls,
            variable=self.progress_var,
            maximum=100.0,
            length=180,
        )
        self.progress_bar.pack(side="left", padx=(14, 0))

    def _sync_provider_defaults(self) -> None:
        previous = getattr(self, "_responsive_last_mode", None)
        super()._sync_provider_defaults()
        current = self.mode_var.get()
        # Hosted Ultra evaluation is intentionally conservative by default.
        # Users can still raise Cases manually after switching providers.
        if current == "nvidia" and previous != "nvidia" and self.limit_var.get() == 10:
            self.limit_var.set(3)
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
        super()._finish_run(result)
        self._run_started_at = None
        self.cancel_button.configure(state="disabled")
        self.progress_var.set(100.0 if not cancelled else (generated * 100.0 / requested if requested else 0.0))
        if cancelled:
            self.status_var.set(f"취소됨 · {generated}/{requested} 완료 · 실패 {failed}")
        elif failed:
            self.status_var.set(f"완료 · {generated}/{requested} 성공 · 실패 {failed}")
        else:
            self.status_var.set(f"완료 · {generated}/{requested}")

    def _show_error(self, exc: Exception) -> None:
        self._run_started_at = None
        if hasattr(self, "cancel_button"):
            self.cancel_button.configure(state="disabled")
        super()._show_error(exc)
