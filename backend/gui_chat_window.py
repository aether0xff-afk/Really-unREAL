from __future__ import annotations

import threading
import tkinter as tk
from datetime import datetime
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from backend.gui_live import LiveChatSession


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
        ttk.Label(
            header,
            text=session.target_alias,
            font=("Segoe UI", 16, "bold"),
        ).pack(side="left")
        ttk.Label(
            header,
            text="SIMULATION · 실제 카카오톡으로 전송되지 않음",
        ).pack(side="left", padx=(10, 0), pady=(5, 0))
        ttk.Button(header, text="새 대화", command=self._reset).pack(side="right")

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
                "답장은 즉시 생성하지 않습니다. 과거 대화에서 학습한 답장 시간에 맞춰 "
                "예약되고, 그 시간이 되었을 때만 모델을 호출합니다."
            ),
            wraplength=580,
        ).grid(row=4, column=0, sticky="ew", pady=(8, 0))

        self._refresh_transcript()
        try:
            self.session.ensure_idle_initiation()
        except Exception:
            # INITIATE timing may be unavailable for sparse relationships.
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
            event = self.session.send_user_message(text)
        except Exception as exc:
            messagebox.showerror("Really-unREAL", str(exc), parent=self)
            return
        self.message_var.set("")
        self._refresh_transcript()
        self.status_var.set(self.session.pending_label())
        # event is intentionally only a scheduled behavior; generation happens
        # later in _poll when event.due_at becomes current.
        _ = event

    def _poll(self) -> None:
        if self._closed:
            return
        now = datetime.now()
        event = self.session.pending_event()
        if event is None:
            self.status_var.set("대기 중 · 지금은 보낼 행동이 예약되어 있지 않음")
        elif event.due_at <= now and not self._generating:
            self._start_generation()
        elif not self._generating:
            self.status_var.set(self.session.pending_label(now=now))
        self.after(500, self._poll)

    def _start_generation(self) -> None:
        self._generating = True
        self.status_var.set(f"{self.session.target_alias} 답장 생성 중…")

        def work() -> None:
            try:
                self.session.process_due(now=datetime.now())
            except Exception as exc:
                self.session.cancel_pending()
                if not self._closed:
                    self.after(0, lambda: self._generation_failed(exc))
                return
            if not self._closed:
                self.after(0, self._generation_finished)

        threading.Thread(target=work, daemon=True).start()

    def _generation_finished(self) -> None:
        self._generating = False
        self._refresh_transcript()
        self.status_var.set(self.session.pending_label())

    def _generation_failed(self, exc: Exception) -> None:
        self._generating = False
        self.status_var.set("생성 실패 · 예약 취소됨")
        messagebox.showerror(
            "Really-unREAL",
            f"모델 응답 생성에 실패했습니다.\n\n{exc}",
            parent=self,
        )

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
        self._refresh_transcript()
        self.status_var.set("새 시뮬레이션 대화")
