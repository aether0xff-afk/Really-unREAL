# Really-unREAL v1.0.1 GUI Quick Start

v1.0.1 adds a small local desktop UI so the first useful run does not require editing `identity.local.json` or typing long CLI commands.

## Windows portable build

1. Download the `Really-unREAL-v1.0.1-Windows-x64` build artifact or release ZIP.
2. Extract the ZIP.
3. Run `Really-unREAL.exe`.
4. Windows SmartScreen may warn about an unsigned community build. Check that the file came from this repository's GitHub Actions/Release before choosing to run it.

The app does not install a service and does not automatically send messages to KakaoTalk, Instagram, or any other platform.

## 1. Export KakaoTalk data

The easiest input is either:

- one KakaoTalk chat ZIP containing `Talk_*.txt`; or
- an outer ZIP containing several KakaoTalk per-chat ZIP exports.

Attachments are ignored by the current text pipeline.

## 2. Load the ZIP

Open the app and choose **카카오톡 ZIP → 찾아보기**.

The app scans the archive locally and suggests **내 이름** by looking for the display name that appears across many chats. This is only a convenience heuristic. **Always confirm the suggested name yourself.**

After confirming your name, choose **대화 상대**. Direct-chat targets are shown roughly in order of how much target-authored evidence is available.

## 3. Start with Quick Audit

Choose **빠른 Audit (LLM 없음)** and press **실행**.

This is the recommended first run because it needs no API key and sends no conversation context to a model. It builds the same leakage-safe Historical Replay core used by the CLI and reports aggregate counts/timing metrics only.

The result panel intentionally does not print raw private chat messages.

## 4. Optional Local LLM generation

Choose **Local LLM** when an OpenAI-compatible local server is already running.

Default endpoint:

```text
http://127.0.0.1:1234/v1
```

This works with local runtimes such as LM Studio when their OpenAI-compatible server is enabled. Enter the model ID exposed by that server. An API key is usually unnecessary for a loopback-only local server.

Loopback (`127.0.0.1` / `localhost`) traffic is treated as local. A non-loopback URL requires the explicit remote-context consent checkbox.

## 5. Optional NVIDIA NIM

Choose **NVIDIA NIM** to use the hosted NVIDIA model adapter.

Default model:

```text
nvidia/nemotron-3-ultra-550b-a55b
```

Hosted inference sends a cutoff-safe private conversation context to NVIDIA. Therefore:

- paste the API key only for the current run;
- check the explicit remote-context consent box;
- do not share screenshots containing the key.

The GUI does not write the API key to result JSON or an identity file.

## Privacy boundaries

The GUI keeps the v1 core rules:

- raw archives stay on the local machine;
- generated results are `SIMULATION`, never promoted to real history;
- raw historical responses are not exposed as RAG examples by default;
- hosted/private-context routes require explicit consent;
- no real messaging-platform sending API is called;
- the results pane is aggregate-only by default.

## Advanced CLI

The GUI is deliberately a small front door, not a replacement for research workflows. Use the existing CLI for:

- Instagram/Kakao cross-platform identity fusion;
- SELF twin experiments;
- dense retrieval ablations;
- closed-loop Shadow Simulation;
- custom identity maps and full benchmark configuration.

See the root `README.md` for those commands.

## Build from source

Python 3.11+ is required.

```bash
python -m pip install -e '.[dev,build]'
pytest -q
pyinstaller --noconfirm --clean --onefile --windowed --name Really-unREAL backend/gui.py
```

The executable is written to `dist/Really-unREAL.exe` on Windows.
