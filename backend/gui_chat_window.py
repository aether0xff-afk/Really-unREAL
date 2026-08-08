from __future__ import annotations

import threading
import tkinter as tk
from datetime import datetime
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from backend.gui_live import LiveChatSession
from backend.providers.errors import PermanentGenerationError, TransientGenerationError


class LiveChatWindow(tk.Toplevel):
    """A small messenger-like view over a persistent LiveChatSession."""

    def __init__(self, master: tk.Misc, session: LiveChatSession) -> None:
        super().__init__(master)
        self.session = session
        self._closed = False
        self._generating = False

        self.title(f"{session.target_alias} · Really-unREAL 시뮬레이션")
        self.geometry("620x720")
        self.minsize(480, 520)
        self.protocol("WM_DELETE_WINDOW", self._close)

        root = ttk.Frame(self, padding=14)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(2, weight=1)

        header = ttk.Frame(root)
        header.grid(row=0, column=0, sticky="ew")
        ttk.Label(header, text=session.target_alias, font=("Segoe UI", 16, "bold")).pack(side="left")
        ttk.Label(header, text="SIMULATION · 실제 카카오톡으로 전송되지 않음").pack(
            side="left", padx=(10, 0), pady=(5, 0)
        )
        ttk.Button(header, text="새 대화", command=self._reset).pack(side="right")
        self.retry_button = ttk.Button(
            header,
            text="생성 재시도",
            command=self._retry_blocked,
            state="disabled",
        )
        self.retry_button.pack(side="right", padx=(0, 8))

        self.status_var = tk.StringVar(value="준비 중…")
        ttk.Label(root, textvariable=self.status_var).grid(
            row=1, column=0, sticky="ew", pady=(8, 8)
        )

        self.transcript = ScrolledText(
            root,
            wrap="word",
            font=("Segoe UI", 11),
            state="disabled",
        )
        self.transcript.grid(row=2, column=0, sticky="nsew")

        composer = ttk.Frame(root)
        composer.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        composer.columnconfigure(0, weight=1)
        self.message_var = tk.StringVar()
        self.entry = ttk.Entry(composer, textvariable=self.message_var)
        self.entry.grid(row=0, column=0, sticky="ew")
        self.entry.bind("<Return>", lambda _event: self._send())
        ttk.Button(composer, text="보내기", command=self._send).grid(
            row=0, column=1, padx=(8, 0)
        )

        ttk.Label(
            root,
            text=(
                "답장은 과거 행동 모델이 먼저 예약합니다. 모델/API가 일시적으로 실패해도 "
                "예약된 행동은 사라지지 않고 생성만 다시 시도합니다."
            ),
            wraplength=580,
        ).grid(row=4, column=0, sticky="ew", pady=(8, 0))

        self._refresh_transcript()
        try:
            self.session.ensure_idle_initiation()
        except Exception:
            pass
        self.entry.focus_set()
        self.after(300, self._poll)

    def _close(self) -> None:
        self._closed = True
        self.destroy()

    def _refresh_transcript(self) -> None:
        messages = self.session.chat_messages()
        self.transcript.configure(state="normal")
        self.transcript.delete("1.0", "end")
        if not messages:
            self.transcript.insert(
                "1.0",
                "아직 시뮬레이션 메시지가 없습니다. 아래에서 먼저 메시지를 보내 보세요.\n",
            )
        else:
            for message in messages:
                speaker = (
                    self.session.target_alias
                    if message.sender_person_id == self.session.target_person_id
                    else "나"
                )
                stamp = message.timestamp.strftime("%m/%d %H:%M")
                self.transcript.insert("end", f"{speaker}  {stamp}\n{message.text}\n\n")
        self.transcript.see("end")
        self.transcript.configure(state="disabled")

    def _send(self) -> None:
        text = self.message_var.get().strip()
        if not text:
            return
        try:
            self.session.send_user_message(text)
        except Exception as exc:
            messagebox.showerror("Really-unREAL", str(exc), parent=self)
            return
        self.message_var.set("")
        self._refresh_transcript()
        self.status_var.set(self.session.pending_label())
        self.retry_button.configure(state="disabled")

    def _poll(self) -> None:
        if self._closed:
            return
        now = datetime.now()
        event = self.session.pending_event()
        if event is None:
            self.status_var.set("대기 중 · 지금은 보낼 행동이 예약되어 있지 않음")
            self.retry_button.configure(state="disabled")
        elif event.status == "BLOCKED":
            self.status_var.set(self.session.pending_label(now=now))
            self.retry_button.configure(state="normal")
        elif event.due_at <= now and not self._generating:
            self.retry_button.configure(state="disabled")
            self._start_generation()
        elif not self._generating:
            self.retry_button.configure(state="disabled")
            self.status_var.set(self.session.pending_label(now=now))
        self.after(500, self._poll)

    def _start_generation(self) -> None:
        self._generating = True
        self.status_var.set(f"{self.session.target_alias} 답장 생성 중…")

        def work() -> None:
            try:
                self.session.process_due(now=datetime.now())
            except TransientGenerationError as exc:
                event = self.session.defer_generation_failure(exc, now=datetime.now())
                if not self._closed:
                    self.after(0, lambda: self._generation_deferred(event))
                return
            except PermanentGenerationError as exc:
                self.session.block_generation_failure(exc)
                if not self._closed:
                    self.after(0, lambda: self._generation_blocked(exc))
                return
            except Exception as exc:
                # Unknown failures must not create a hot retry loop. Preserve the
                # scheduled behavior as BLOCKED until the user explicitly retries.
                self.session.block_generation_failure(exc)
                if not self._closed:
                    self.after(0, lambda: self._generation_blocked(exc))
                return
            if not self._closed:
                self.after(0, self._generation_finished)

        threading.Thread(target=work, daemon=True).start()

    def _generation_finished(self) -> None:
        self._generating = False
        self._refresh_transcript()
        self.retry_button.configure(state="disabled")
        self.status_var.set(self.session.pending_label())

    def _generation_deferred(self, event) -> None:
        self._generating = False
        self.retry_button.configure(state="disabled")
        if event is None:
            self.status_var.set("모델 일시 오류 · 예약된 행동 확인 중")
        else:
            self.status_var.set(self.session.pending_label())

    def _generation_blocked(self, exc: Exception) -> None:
        self._generating = False
        self.retry_button.configure(state="normal")
        self.status_var.set("답장 행동 보존됨 · 모델 설정 확인 후 생성 재시도")
        messagebox.showerror(
            "Really-unREAL",
            "모델 호출을 계속할 수 없습니다. 예약된 답장 행동은 지우지 않았습니다.\n\n"
            f"{exc}\n\n설정/API key를 확인한 뒤 창을 다시 열거나 '생성 재시도'를 누르세요.",
            parent=self,
        )

    def _retry_blocked(self) -> None:
        try:
            event = self.session.retry_blocked(now=datetime.now())
        except Exception as exc:
            messagebox.showerror("Really-unREAL", str(exc), parent=self)
            return
        if event is not None:
            self.retry_button.configure(state="disabled")
            self.status_var.set("생성 재시도 준비 중…")

    def _reset(self) -> None:
        if not messagebox.askyesno(
            "새 대화",
            "현재 SIMULATION 대화와 예약된 답장을 지울까요?\n실제 카카오톡 원본은 지워지지 않습니다.",
            parent=self,
        ):
            return
        self.session.reset()
        try:
            self.session.ensure_idle_initiation()
        except Exception:
            pass
        self.retry_button.configure(state="disabled")
        self._refresh_transcript()
        self.status_var.set("새 시뮬레이션 대화")
