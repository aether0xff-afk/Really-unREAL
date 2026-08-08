# Really-unREAL 1.1 GUI Quick Start

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

## 2. 빠른 진단 — LLM 없음

**빠른 진단 (LLM 없음)** does not generate a reply. It checks that the selected data can build the replay dataset and reply-timing model.

Use it to check whether the exports parsed correctly, the self/target mapping is usable, enough history exists, and a relationship-specific timing model can be built. No model API is called.

## 3. 모델 테스트 — Historical Replay benchmark

**모델 테스트 · Local LLM** and **모델 테스트 · NVIDIA NIM** are evaluation modes, not chat modes.

Really-unREAL hides real historical continuations, asks the model to reproduce them using only older evidence, and then evaluates separate observable dimensions:

- character-pattern and token overlap;
- ending/style overlap;
- message-burst splitting;
- laughter/cry/question behavior;
- reply timing falling inside the feasible historical interval.

When enough cases exist the GUI shows descriptive 95% intervals. A point estimate from a tiny sample must not be treated as an overall fidelity score.

**NVIDIA defaults to 3 cases only as a quick smoke check.** For actual model comparison, raise **테스트 수** to at least 10–20 when time/API budget allows. Use **결과 저장** for the full numeric JSON.

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

A separate messenger-like window opens for the selected person. Messages typed there are simulation input only; they are not sent to a real messaging service.

### What happens after you send a message

1. Your input is recorded as `SIMULATION`.
2. The live behavior model looks at observable context: relationship, clock time, weekday/weekend, recent activity, previous gap, and—when enough evidence exists—whether the current message is a question, very short message, or statement.
3. A stochastic `REPLY` time is scheduled from historical behavior; it is not a fixed 30-second timer.
4. The chat window displays the countdown.
5. Only when the behavior time arrives is the language model called.
6. Generation uses the current visible conversation, relationship-focused style/burst profiles, cutoff-safe memory/retrieval, and at most two older REAL replies as private style exemplars.
7. A guard retries generation if a long output is suspiciously close to one of those historical exemplars.
8. After a reply, an optional `INITIATE` event may be scheduled from historical behavior.

### What if NVIDIA/local inference fails?

Provider availability does **not** decide whether the simulated person wanted to reply.

- Temporary HTTP 429/5xx, timeout or network failure: the original reply behavior time is preserved; generation delivery retries separately with backoff.
- Configuration/credential/invalid-format failure: the action is kept as blocked instead of silently deleted. Fix the provider setting and choose **생성 재시도**.
- A retry never gets to read real/simulated messages that arrived after the original behavior time merely because the provider was down.

Pending events and simulated messages are persisted in a local SQLite database. **새 대화** clears only SIMULATION state; imported REAL evidence is never deleted by that button.

## Privacy boundaries

- raw archives stay on the local machine;
- imported evidence remains `REAL`;
- user-entered live messages and generated live messages are `SIMULATION`;
- simulation output is never promoted into asserted real history;
- hosted/private-context routes require explicit consent;
- no real messaging-platform sending API is called;
- historical raw message text is not printed in benchmark results;
- live style exemplars are prompt-time private context only and are not written to public result JSON or release artifacts.

## Important interpretation limit

Really-unREAL 1.1 uses a general language model conditioned by person-specific observable evidence. It does **not** claim that the base LLM itself has been neurally fine-tuned into that person, nor that replay similarity reconstructs hidden identity or mental state.

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
