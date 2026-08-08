from __future__ import annotations

import json
import os
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from backend import __version__
from backend.gui_support import (
    LOCAL_BASE_URL,
    NVIDIA_BASE_URL,
    NVIDIA_MODEL,
    direct_targets_for_self,
    load_quick_kakao,
    rank_self_aliases,
    run_quick_audit,
    run_quick_generation,
)


class ReallyUnrealApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(f"Really-unREAL {__version__}")
        self.geometry("900x720")
        self.minsize(780, 620)
        self._conversations = []
        self._last_result: dict[str, object] | None = None

        self.archive_var = tk.StringVar()
        self.self_var = tk.StringVar()
        self.target_var = tk.StringVar()
        self.mode_var = tk.StringVar(value="audit")
        self.model_var = tk.StringVar()
        self.base_url_var = tk.StringVar(value=LOCAL_BASE_URL)
        self.api_key_var = tk.StringVar()
        self.limit_var = tk.IntVar(value=10)
        self.remote_consent_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="카카오톡 ZIP을 선택하세요.")

        self._build_ui()
        self.mode_var.trace_add("write", lambda *_: self._sync_provider_defaults())

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=18)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(5, weight=1)

        header = ttk.Frame(root)
        header.grid(row=0, column=0, sticky="ew")
        ttk.Label(header, text="Really-unREAL", font=("Segoe UI", 20, "bold")).pack(side="left")
        ttk.Label(header, text=f"v{__version__}  ·  local-first replay lab").pack(side="left", padx=(10, 0), pady=(7, 0))
        ttk.Button(header, text="사용법", command=self._show_help).pack(side="right")

        intro = ttk.Label(
            root,
            text=(
                "카카오톡 내보내기 ZIP만 넣으면 내 이름 후보와 대화 상대를 자동으로 찾습니다. "
                "원본 대화는 GitHub나 결과 창에 업로드/출력하지 않습니다."
            ),
            wraplength=820,
        )
        intro.grid(row=1, column=0, sticky="ew", pady=(8, 14))

        data_box = ttk.LabelFrame(root, text="1. 데이터", padding=12)
        data_box.grid(row=2, column=0, sticky="ew")
        data_box.columnconfigure(1, weight=1)
        ttk.Label(data_box, text="카카오톡 ZIP").grid(row=0, column=0, sticky="w")
        ttk.Entry(data_box, textvariable=self.archive_var).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(data_box, text="찾아보기", command=self._browse_archive).grid(row=0, column=2)
        self.scan_button = ttk.Button(data_box, text="불러오기", command=self._scan_archive)
        self.scan_button.grid(row=0, column=3, padx=(8, 0))

        ttk.Label(data_box, text="내 이름").grid(row=1, column=0, sticky="w", pady=(10, 0))
        self.self_combo = ttk.Combobox(data_box, textvariable=self.self_var, state="readonly")
        self.self_combo.grid(row=1, column=1, sticky="ew", padx=8, pady=(10, 0))
        self.self_combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh_targets())
        ttk.Label(data_box, text="자동 추정 · 직접 확인 필요").grid(row=1, column=2, columnspan=2, sticky="w", pady=(10, 0))

        ttk.Label(data_box, text="대화 상대").grid(row=2, column=0, sticky="w", pady=(10, 0))
        self.target_combo = ttk.Combobox(data_box, textvariable=self.target_var, state="readonly")
        self.target_combo.grid(row=2, column=1, sticky="ew", padx=8, pady=(10, 0))
        ttk.Label(data_box, text="1:1 대화가 많은 순").grid(row=2, column=2, columnspan=2, sticky="w", pady=(10, 0))

        run_box = ttk.LabelFrame(root, text="2. 실행", padding=12)
        run_box.grid(row=3, column=0, sticky="ew", pady=(14, 0))
        run_box.columnconfigure(1, weight=1)

        modes = ttk.Frame(run_box)
        modes.grid(row=0, column=0, columnspan=4, sticky="w")
        ttk.Radiobutton(modes, text="빠른 Audit (LLM 없음)", variable=self.mode_var, value="audit").pack(side="left")
        ttk.Radiobutton(modes, text="Local LLM", variable=self.mode_var, value="local").pack(side="left", padx=(18, 0))
        ttk.Radiobutton(modes, text="NVIDIA NIM", variable=self.mode_var, value="nvidia").pack(side="left", padx=(18, 0))

        ttk.Label(run_box, text="Model").grid(row=1, column=0, sticky="w", pady=(10, 0))
        self.model_entry = ttk.Entry(run_box, textvariable=self.model_var)
        self.model_entry.grid(row=1, column=1, sticky="ew", padx=8, pady=(10, 0))
        ttk.Label(run_box, text="Cases").grid(row=1, column=2, sticky="e", pady=(10, 0))
        ttk.Spinbox(run_box, from_=1, to=100, textvariable=self.limit_var, width=7).grid(row=1, column=3, pady=(10, 0))

        ttk.Label(run_box, text="Base URL").grid(row=2, column=0, sticky="w", pady=(10, 0))
        self.url_entry = ttk.Entry(run_box, textvariable=self.base_url_var)
        self.url_entry.grid(row=2, column=1, sticky="ew", padx=8, pady=(10, 0))
        ttk.Label(run_box, text="API key").grid(row=2, column=2, sticky="e", pady=(10, 0))
        self.key_entry = ttk.Entry(run_box, textvariable=self.api_key_var, show="•", width=22)
        self.key_entry.grid(row=2, column=3, pady=(10, 0))

        self.consent_check = ttk.Checkbutton(
            run_box,
            text="원격 서버로 대화 문맥이 전송되는 것을 이해하고 허용합니다.",
            variable=self.remote_consent_var,
        )
        self.consent_check.grid(row=3, column=0, columnspan=4, sticky="w", pady=(10, 0))

        buttons = ttk.Frame(root)
        buttons.grid(row=4, column=0, sticky="ew", pady=(14, 8))
        self.run_button = ttk.Button(buttons, text="실행", command=self._run)
        self.run_button.pack(side="left")
        ttk.Button(buttons, text="결과 저장", command=self._save_result).pack(side="left", padx=(8, 0))
        ttk.Label(buttons, textvariable=self.status_var).pack(side="right")

        result_box = ttk.LabelFrame(root, text="3. 결과 · 원문 메시지는 표시하지 않음", padding=8)
        result_box.grid(row=5, column=0, sticky="nsew")
        result_box.rowconfigure(0, weight=1)
        result_box.columnconfigure(0, weight=1)
        self.output = ScrolledText(result_box, wrap="word", font=("Consolas", 10))
        self.output.grid(row=0, column=0, sticky="nsew")
        self.output.insert("1.0", "ZIP을 불러온 뒤 상대를 선택하고 Audit부터 실행해 보세요.\n")
        self.output.configure(state="disabled")
        self._sync_provider_defaults()

    def _browse_archive(self) -> None:
        path = filedialog.askopenfilename(
            title="카카오톡 내보내기 ZIP 선택",
            filetypes=[("ZIP archive", "*.zip"), ("All files", "*.*")],
        )
        if path:
            self.archive_var.set(path)
            self._scan_archive()

    def _scan_archive(self) -> None:
        path = self.archive_var.get().strip()
        if not path:
            messagebox.showinfo("Really-unREAL", "카카오톡 ZIP을 먼저 선택하세요.")
            return
        self._set_busy(True, "ZIP 분석 중…")

        def work() -> None:
            try:
                conversations = load_quick_kakao(path)
                candidates = rank_self_aliases(conversations)
                if not candidates:
                    raise ValueError("대화 참여자 이름을 찾지 못했습니다.")
                self.after(0, lambda: self._finish_scan(conversations, candidates))
            except Exception as exc:  # GUI boundary: show a friendly error.
                self.after(0, lambda: self._show_error(exc))

        threading.Thread(target=work, daemon=True).start()

    def _finish_scan(self, conversations, candidates: list[str]) -> None:
        self._conversations = conversations
        self.self_combo["values"] = candidates
        self.self_var.set(candidates[0])
        self._refresh_targets()
        self._set_busy(False, f"{len(conversations)}개 대화를 읽었습니다. 내 이름과 상대를 확인하세요.")

    def _refresh_targets(self) -> None:
        if not self._conversations or not self.self_var.get():
            return
        targets = direct_targets_for_self(self._conversations, self.self_var.get())
        self.target_combo["values"] = targets
        self.target_var.set(targets[0] if targets else "")

    def _sync_provider_defaults(self) -> None:
        mode = self.mode_var.get()
        if mode == "audit":
            self.model_var.set("")
            self.base_url_var.set(LOCAL_BASE_URL)
            self.remote_consent_var.set(False)
        elif mode == "nvidia":
            self.model_var.set(NVIDIA_MODEL)
            self.base_url_var.set(NVIDIA_BASE_URL)
        elif mode == "local":
            if self.base_url_var.get() == NVIDIA_BASE_URL:
                self.base_url_var.set(LOCAL_BASE_URL)
            if self.model_var.get() == NVIDIA_MODEL:
                self.model_var.set("")

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
        self._set_busy(True, "실행 중…")

        def work() -> None:
            try:
                if mode == "audit":
                    result = run_quick_audit(
                        self._conversations,
                        self_alias=self_alias,
                        target_alias=target_alias,
                    )
                else:
                    api_key = self.api_key_var.get().strip() or None
                    if mode == "nvidia" and api_key:
                        # Never persist the key; it lives only in this process call.
                        os.environ.pop("NVIDIA_API_KEY", None)
                    result = run_quick_generation(
                        self._conversations,
                        self_alias=self_alias,
                        target_alias=target_alias,
                        provider=mode,
                        model=self.model_var.get().strip(),
                        base_url=self.base_url_var.get().strip(),
                        api_key=api_key,
                        allow_remote_private_context=self.remote_consent_var.get(),
                        limit=self.limit_var.get(),
                    )
                self.after(0, lambda: self._finish_run(result))
            except Exception as exc:  # GUI boundary: surface failures without raw context.
                self.after(0, lambda: self._show_error(exc))

        threading.Thread(target=work, daemon=True).start()

    def _finish_run(self, result: dict[str, object]) -> None:
        self._last_result = result
        text = json.dumps(result, ensure_ascii=False, indent=2)
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("1.0", text)
        self.output.configure(state="disabled")
        self._set_busy(False, "완료")

    def _save_result(self) -> None:
        if self._last_result is None:
            messagebox.showinfo("Really-unREAL", "저장할 결과가 아직 없습니다.")
            return
        path = filedialog.asksaveasfilename(
            title="분석 결과 저장",
            defaultextension=".json",
            initialfile="really-unreal-result.json",
            filetypes=[("JSON", "*.json")],
        )
        if path:
            Path(path).write_text(
                json.dumps(self._last_result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            self.status_var.set(f"저장됨: {Path(path).name}")

    def _set_busy(self, busy: bool, status: str) -> None:
        state = "disabled" if busy else "normal"
        self.scan_button.configure(state=state)
        self.run_button.configure(state=state)
        self.status_var.set(status)

    def _show_error(self, exc: Exception) -> None:
        self._set_busy(False, "실패")
        messagebox.showerror("Really-unREAL", str(exc))

    def _show_help(self) -> None:
        messagebox.showinfo(
            "Really-unREAL 사용법",
            "1) 카카오톡 대화 내보내기 ZIP을 선택합니다.\n"
            "2) 자동 추정된 '내 이름'이 맞는지 확인합니다.\n"
            "3) 재현할 대화 상대를 선택합니다.\n"
            "4) 처음에는 '빠른 Audit'을 실행합니다.\n"
            "5) 생성 평가가 필요하면 Local LLM 또는 NVIDIA NIM을 선택합니다.\n\n"
            "Local LLM 기본 주소는 LM Studio 호환 http://127.0.0.1:1234/v1 입니다.\n"
            "NVIDIA NIM은 private context가 외부로 전송되므로 동의 체크가 필수입니다.\n"
            "API key와 원문 메시지는 결과 JSON에 저장되지 않습니다.",
        )


def main() -> None:
    app = ReallyUnrealApp()
    app.mainloop()


if __name__ == "__main__":
    main()
