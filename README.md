# Really-unREAL 1.1.0

A local-first conversation-behavior digital-twin framework grounded in real message history.

Really-unREAL does **not** claim to reconstruct hidden thoughts, feelings, attraction, intent, or a person's mind. It models narrower observable behavior: **when someone stays silent, when they reply or follow up, how timing changes with visible context, how they split messages, and how they write.**

> Make the most plausible behavior from the evidence, not the most entertaining behavior.

## Quick start — Windows GUI

1. Download `Really-unREAL-v1.1.0-Windows-x64.zip` from the GitHub Release.
2. Extract it and run `Really-unREAL.exe`.
3. Choose one or many KakaoTalk export ZIP files.
4. Confirm the suggested **내 이름**.
5. Choose a direct-chat **대화 상대**.
6. Run **빠른 진단** to check the imported data without an LLM.
7. Optionally run **모델 테스트** on held-out historical replies.
8. Choose Local LLM or NVIDIA NIM and press **대화 시작** for a persistent simulation conversation.

The app does not install a background service and never sends a message to KakaoTalk, Instagram, or another real messaging platform.

See `docs/QUICKSTART_GUI.md` for the full desktop walkthrough.

## Desktop modes

```text
Kakao ZIP(s)
   |
   v
confirm self -> choose target
   |
   +--> 빠른 진단
   |      parsing / identity / timing health check
   |      no LLM
   |
   +--> 모델 테스트
   |      hide real historical continuation
   |      reproduce and measure observable behavior
   |
   +--> 대화 시작
          persistent SIMULATION chat
          actual clock + WAIT/REPLY/INITIATE
```

The NVIDIA default of **3 test cases is only a quick smoke check**. For model comparisons, use 10–20+ held-out cases when practical. Really-unREAL reports content overlap, endings, expression behavior, message splitting, and timing separately instead of inventing one identity-fidelity score.

## What 1.1.0 changes

### Provider-independent behavior

A language-model outage no longer changes the simulated person's behavioral decision.

```text
behavior model schedules REPLY at T
             |
             +--> provider success -> generated message
             |
             +--> 429 / 5xx / timeout
                    behavior time T stays unchanged
                    generation retries on a separate delivery clock
```

Permanent provider/configuration failures preserve the behavior as blocked until generation is explicitly retried. A provider retry cannot read conversation context that arrived after the original modeled behavior time merely because the provider was unavailable.

### Context-conditioned stochastic timing

Live timing can condition on observable evidence including:

- current direct relationship;
- REPLY vs INITIATE;
- time of day;
- weekday/weekend;
- recent 15-minute conversation activity;
- previous visible-message gap;
- question / very-short / statement message type when enough relationship evidence exists.

The richer discrete hazard model is deployed only when it improves held-out validation. Sparse histories use contextual empirical backoff. Kakao same-minute observations remain timing intervals rather than fake exact seconds.

### Stronger person-specific generation

```text
current visible conversation       -> WHAT must be answered
relationship style fingerprint     -> HOW this person tends to write
burst profile                      -> bubble count / amount of text
cutoff-safe topic/event memory     -> continuity
historical situation retrieval     -> similar observed contexts
<=2 older REAL style exemplars     -> phrasing/rhythm evidence in Live mode
anti-copy guard                    -> regenerate suspicious long reuse
```

Short common expressions such as `ㅇㅇ`, `ㄴㄴ`, or `ㅋㅋ` may naturally recur. Long near-verbatim reuse of a historical style exemplar triggers another independently worded generation attempt.

Historical Replay/model-test paths keep raw historical response exemplars disabled by default.

## Core architecture

A normal persona chatbot is roughly:

```text
history -> RAG -> LLM -> message
```

Really-unREAL separates behavior from prose:

```text
real clock
   |
   v
WAIT / REPLY / INITIATE
   |
   +---- WAIT ----------------------> nothing happens
   |
   v
cutoff-safe persona + memory + retrieval
   |
   v
language model
   |
   v
SIMULATION memory
```

The LLM never gets to talk merely because the app is running.

## Privacy model

```text
REAL        observations imported from real records
SIMULATION  generated events and user-entered simulation messages
```

- Raw archives and `identity.local.json` are not release content.
- SIMULATION output is never promoted into asserted REAL history.
- Loopback model endpoints are local by default.
- Non-loopback private-context transmission requires explicit consent.
- API keys are not written to evaluation JSON or release artifacts.
- Live style exemplars are private prompt-time evidence and are not written into public result JSON, CI status, or release assets.
- No real messaging-platform send API exists in the core.

## Historical Replay

Replay hides a real continuation and asks the system to reproduce behavior using only older evidence.

```text
visible past
    |
    +---- cutoff
            |
           WAIT / REPLY / INITIATE timing
            |
            +---- held-out real event
                    timing interval
                    burst shape
                    hidden content used only after generation for evaluation
```

Train/validation/test are chronological, not random. Persona, retrieval, style fingerprints, topics, events, and optional Live style exemplars all obey temporal cutoff barriers.

## Main capabilities

- KakaoTalk text/ZIP ingestion and multi-ZIP GUI selection.
- Meta/Instagram ingestion and source-aware fusion for advanced workflows.
- Conservative identity mapping.
- PERSON and SELF twin replay datasets.
- Leakage-safe chronological Historical Replay.
- Interval-aware Kakao timing.
- Relationship empirical timing + validation-gated discrete hazard model.
- Context-conditioned stochastic live timing.
- Relationship-focused language/style and burst profiles.
- Topic/event memory and historical-situation retrieval.
- Anti-copy guarded Live style exemplars.
- NVIDIA NIM and OpenAI-compatible local generation adapters.
- Closed-loop Shadow Simulation baseline.
- Persistent SQLite live runtime with provider retry recovery.
- Strict REAL/SIMULATION separation.

## Important limits

- Human communication is stochastic; replay similarity is not proof of identity reconstruction.
- The base LLM remains a **general language model conditioned by person-specific observable evidence**. v1.1.0 does not claim a neural fine-tune into a real person.
- INITIATE is intrinsically harder than REPLY because unseen future events cannot be inferred from nowhere.
- Sparse relationships require broader statistical backoff.
- Kakao exports cannot reveal exact second-level timing.
- Hosted inference still depends on provider availability even though outages no longer alter modeled behavior decisions.
- Windows portable binaries are unsigned community builds, so SmartScreen may warn.

## Advanced workflows

The CLI remains available for cross-platform identity fusion, SELF twin experiments, dense retrieval ablations, custom identity maps, Historical Replay audits, replay generation benchmarks, and Shadow Simulation.

Relevant docs:

- `docs/QUICKSTART_GUI.md`
- `docs/HISTORICAL_REPLAY.md`
- `docs/TEMPORAL_HAZARD.md`
- `docs/CUTOFF_RAG.md`
- `docs/SELF_TWIN.md`
- `docs/V1_0.md`

## Build from source

Python 3.11+:

```bash
python -m pip install -e '.[dev,build]'
pytest -q
pyinstaller --noconfirm --clean --onefile --windowed --name Really-unREAL backend/gui_entry.py
./dist/Really-unREAL.exe --smoke
```

Official release workflows run tests, integrated runtime imports, PyInstaller, the packaged EXE smoke path, and release packaging before publication.

## 1.1 development train

- `v1.1.0-dev.1` — provider failure resilience
- `v1.1.0-dev.2` — context-conditioned timing
- `v1.1.0-dev.3` — stronger person-specific generation + anti-copy
- `v1.1.0-rc.1` — integration / behavior-vs-delivery clock separation / evaluation uncertainty
- **`v1.1.0` — final release**
