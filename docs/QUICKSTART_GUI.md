# Really-unREAL GUI Quick Start

The desktop UI provides a simple first-run path without editing `identity.local.json` or typing long CLI commands.

## Windows portable build

1. Download the latest `Really-unREAL-*-Windows-x64.zip` release asset.
2. Extract it. It contains `Really-unREAL.exe` and this quick-start guide; no Python installation is required.
3. Run `Really-unREAL.exe`.
4. Windows SmartScreen may warn about an unsigned community build. Check that the file came from this repository's GitHub Release.

The app does not install a service and never sends messages to KakaoTalk, Instagram, or any other real platform.

## 1. Export and load KakaoTalk data

Choose **여러 ZIP 선택**. You can select one or many individual KakaoTalk chat ZIPs, and each selected ZIP may itself be an outer bundle containing several chat exports.

On Windows, use **Ctrl** to pick individual files or **Shift** to select a range. Exact duplicate conversations are removed so accidental duplicate selection does not double-count evidence.

After loading, confirm the suggested **내 이름** and choose **대화 상대**.

## 2. 빠른 진단 — no LLM

**빠른 진단 (LLM 없음)** does not generate a reply. It checks that the selected data can build the replay dataset and reply-timing model.

Use it when you want to answer questions such as:

- Did the Kakao exports parse correctly?
- Is my display name / target mapping usable?
- Is there enough history for replay and timing analysis?
- Can the relationship-specific timing model be built?

No model API is called in this mode.

## 3. 모델 테스트 — Historical Replay benchmark

**모델 테스트 · Local LLM** and **모델 테스트 · NVIDIA NIM** are evaluation modes, not chat modes.

Really-unREAL hides real historical continuations, asks the model to reproduce them using only older evidence, and then compares the generated burst with the held-out real continuation. The result panel shows a human-readable summary such as generation success count, elapsed time, writing-form similarity, message-splitting error, expression matches, and timing-range matches.

Use **결과 저장** when you want the full numeric JSON.

### Local model

Default OpenAI-compatible endpoint:

```text
http://127.0.0.1:1234/v1
```

This works with local runtimes such as LM Studio when their OpenAI-compatible server is enabled.

### NVIDIA NIM

Default model:

```text
nvidia/nemotron-3-ultra-550b-a55b
```

Hosted inference sends cutoff-safe private conversation context to NVIDIA. Paste the API key only for the current run and check the explicit remote-context consent box. The GUI does not write the API key into result files.

## 4. 대화 시작 — live SIMULATION

Select **Local LLM** or **NVIDIA NIM**, then press **대화 시작**.

A separate messenger-like window opens for the selected person. Messages typed there are simulation input only; they are not sent to a real service.

The live mode is intentionally not an instant chatbot:

1. You type a message.
2. Really-unREAL records it as `SIMULATION`.
3. The learned relationship timing model schedules a `REPLY` time.
4. The chat window shows a countdown such as `답장 예정 · 약 28초 후`.
5. Only when the scheduled time arrives does the language model generate the reply.
6. After a reply, an optional future `INITIATE` event may be scheduled from historical behavior.

Pending events and simulated messages are persisted in a local SQLite database, so closing and reopening the app does not deliberately collapse elapsed time. **새 대화** clears only SIMULATION state; imported real evidence is never deleted by that button.

## Privacy boundaries

- raw archives stay on the local machine;
- imported evidence remains `REAL`;
- user-entered live messages and generated live messages are `SIMULATION`;
- simulation output is never promoted into asserted real history;
- hosted/private-context routes require explicit consent;
- no real messaging-platform sending API is called;
- historical raw message text is not printed in benchmark results by default.

## Advanced CLI

Use the existing CLI for Instagram/Kakao identity fusion, SELF twin experiments, dense retrieval ablations, closed-loop Shadow Simulation, custom identity maps, and full benchmark configuration.

## Build from source

Python 3.11+ is required.

```bash
python -m pip install -e '.[dev,build]'
pytest -q
pyinstaller --noconfirm --clean --onefile --windowed --name Really-unREAL backend/gui_entry.py
./dist/Really-unREAL.exe --smoke
```
