# Really-unREAL 1.1

A local-first conversation-behavior digital-twin framework grounded in real message history.

Really-unREAL does **not** claim to reconstruct hidden thoughts, feelings, attraction, or intent. It models narrower observable behavior: **when someone stays silent, when they reply or follow up, how timing changes with the current situation, how they split messages, how they write, and which previously observed topics/events plausibly continue.**

> Make the most plausible behavior from the evidence, not the most entertaining behavior.

## Start here — Windows desktop GUI

The 1.1 desktop flow is designed so a user can load KakaoTalk exports and start without manually creating an identity JSON file.

### Portable Windows build

1. Download the latest `Really-unREAL-*-Windows-x64.zip` from GitHub Releases.
2. Extract it and run `Really-unREAL.exe`.
3. Choose one or many KakaoTalk export ZIP files.
4. Confirm the automatically suggested **내 이름**.
5. Pick a direct-chat **대화 상대**.
6. Use **빠른 진단** to check parsing and timing data without an LLM.
7. Optionally run **모델 테스트** on held-out historical replies.
8. Choose Local LLM or NVIDIA NIM and press **대화 시작** for a persistent simulation conversation.

The portable executable does not require a separate Python installation. It does not install a background service and it never sends a message to KakaoTalk, Instagram, or another real messaging platform.

For step-by-step details, see `docs/QUICKSTART_GUI.md`.

### Run the GUI from source

Python 3.11+:

```bash
python -m pip install -e '.[dev]'
python -m backend.gui_entry
```

Installing the package also exposes the `really-unreal` GUI entry point.

## Three desktop modes

```text
Kakao ZIP(s)
   |
   v
confirm self name -> choose direct-chat target
   |
   +--> 빠른 진단
   |      no LLM, no API key
   |      checks replay/timing data
   |
   +--> 모델 테스트
   |      hides real historical continuations
   |      measures timing/content/style reproduction
   |
   +--> 대화 시작
          persistent SIMULATION chat
          actual clock + WAIT/REPLY/INITIATE timing
```

The NVIDIA model-test default of 3 cases is a **quick smoke check**, not a statistically strong fidelity estimate. Use 10–20+ held-out cases for model comparisons when practical. The GUI reports uncertainty and keeps content, endings, expression behavior, message splitting, and timing as separate metrics instead of inventing one overall identity score.

## What changed in 1.1

### Provider-independent behavior

The behavior model and the language-model provider are separate systems.

```text
behavior says REPLY at 21:07:30
           |
           v
provider call
   | success -> message appears at modeled behavior time
   |
   +-- HTTP 503 / timeout -> behavior remains scheduled
                             generation retries separately
```

A temporary NVIDIA/local-server outage no longer deletes a reply the behavior model already decided should happen. The original simulated behavior time is immutable; provider retry time is separate bookkeeping. Permanent configuration errors preserve the action as blocked until the user retries generation.

### Context-conditioned live timing

Live timing no longer reuses one median or blindly samples one global delay distribution. It can condition on observable context including:

- current direct relationship;
- REPLY vs INITIATE action;
- time of day;
- weekday/weekend;
- recent 15-minute conversation activity;
- gap between recent visible messages;
- question / very-short / statement message type when enough relationship evidence exists.

The existing discrete hazard model is used only when held-out validation shows that it beats the empirical baseline. Sparse relationships use conservative contextual empirical backoff.

### Stronger person-specific generation

Generation now combines:

```text
current visible conversation       -> what must be answered
relationship-focused style         -> how the person writes
burst behavior profile             -> how many bubbles / how much text
cutoff-safe topics/events           -> continuity only
historical situation retrieval      -> similar observed situations
<= 2 older REAL style exemplars     -> phrasing/rhythm evidence in Live mode
anti-copy guard                     -> reject long near-verbatim reuse
```

Short common expressions such as `ㅇㅇ`, `ㄴㄴ`, `ㅋㅋ` may naturally recur. Long historical wording is treated differently: near-verbatim exemplar reuse causes a regeneration attempt.

## What makes it different

A normal persona chatbot is roughly:

```text
history -> RAG -> LLM -> message
```

Really-unREAL separates behavior from prose:

```text
real clock
   |
   v
WAIT / REPLY / INITIATE timing
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

The LLM never gets to talk merely because the application is running.

## Core capabilities

- KakaoTalk text/ZIP ingestion, including selecting many ZIPs at once.
- Meta/Instagram export ingestion and source-aware evidence fusion for advanced workflows.
- Conservative cross-platform identity mapping.
- PERSON and SELF twin replay datasets.
- Chronological train/validation/test Historical Replay with future-leakage barriers.
- `WAIT / REPLY / INITIATE` behavior modeling.
- Interval-aware timing for coarse Kakao minute timestamps.
- Relationship-aware empirical timing baseline.
- Validation-gated discrete hazard timing model.
- Context-conditioned stochastic live timing.
- Relationship-conditioned language/style profiles and burst profiles.
- Observable topic memory and explicit event/date memory.
- Cutoff-safe historical-situation retrieval.
- Live-only limited style exemplars with anti-copy protection.
- Provider-agnostic generation through the `BurstLanguageModel` contract.
- NVIDIA NIM and generic OpenAI-compatible local adapters.
- Closed-loop Shadow Simulation baseline.
- Persistent SQLite-backed live discrete-event runtime.
- Strict separation of `REAL` and `SIMULATION` memories.
- Explicit consent required before non-loopback private-context transmission.
- No real messaging-platform auto-send API in the core.

## Privacy model

Private data is local by default.

```text
raw archives              gitignored / not release content
identity.local.json       gitignored
SQLite simulation state   local only
API keys                   process memory only
```

Loopback generation/embedding endpoints are permitted by default. Any non-loopback endpoint that receives private conversation context requires explicit remote-context consent.

The system distinguishes:

```text
REAL        observations imported from real records
SIMULATION  generated events/messages
```

Simulation output is never promoted into asserted real history. Private style exemplars are prompt-time evidence only and are not written to public evaluation JSON, CI status, release notes, or release assets.

## Historical Replay

Replay hides a real continuation and asks the system to reproduce behavior using only older evidence.

```text
visible past
    |
    +---- cutoff
    |       |
    |      WAIT?
    |       |
    +-------+--------- real held-out event
                        |
                        +-- timing interval
                        +-- REPLY/INITIATE proxy or ambiguity
                        +-- burst shape
                        +-- hidden content used only for evaluation
```

Train/validation/test are chronological, not random. Kakao minute timestamps are represented as feasible intervals rather than fake second-level truth.

## Recommended research workflow

```text
raw exports
   |
identity + source fusion
   |
Historical Replay
   |
replay generation benchmark
   |
closed-loop Shadow Simulation
   |
LiveSimulationEngine
```

Do not treat a convincing live chat as proof of fidelity. Replay and shadow evaluation exist specifically to catch convincing-looking but behaviorally wrong models.

## Advanced CLI quick start

Create and review a gitignored `identity.local.json` first for advanced cross-platform/custom identity workflows.

### Audit behavior

PERSON twin:

```bash
python -m backend.replay_audit \
  ./data/raw/kakao_bundle.zip \
  ./data/raw/instagram_export.zip \
  ./identity.local.json \
  person-001
```

SELF twin:

```bash
python -m backend.replay_audit \
  ./data/raw/kakao_bundle.zip \
  ./data/raw/instagram_export.zip \
  ./identity.local.json \
  --self-twin
```

### Generate/evaluate with a local model

```bash
python -m backend.replay_generate \
  ./data/raw/kakao_bundle.zip \
  ./data/raw/instagram_export.zip \
  ./identity.local.json \
  --self-twin \
  --provider local \
  --model YOUR_LOCAL_MODEL \
  --sources both
```

The default local endpoint is `http://127.0.0.1:1234/v1`, suitable for an OpenAI-compatible local server such as LM Studio.

### Optional hosted NVIDIA generation

Hosted generation sends cutoff-safe private context to a remote endpoint and therefore requires explicit consent:

```bash
python -m backend.replay_generate \
  ./data/raw/kakao_bundle.zip \
  ./data/raw/instagram_export.zip \
  ./identity.local.json \
  person-001 \
  --provider nvidia \
  --allow-remote-private-context
```

## Known limits

- Human communication is stochastic. Replay similarity is not proof of identity reconstruction.
- The base LLM remains a general language model conditioned by person-specific evidence; 1.1 does not claim to be a neural fine-tune of the person.
- INITIATE is harder than REPLY because unseen future events cannot be inferred from nowhere.
- Sparse relationships still require broader fallbacks.
- Kakao exports cannot reveal true second-level reply timing; same-minute observations remain intervals.
- Hosted inference depends on the remote provider's availability even though provider outages no longer alter modeled behavior decisions.
- PERSON twins raise consent and impersonation concerns; the core intentionally does not send to real platforms.
- Windows portable binaries are unsigned community builds, so SmartScreen may warn.

## Build from source

```bash
python -m pip install -e '.[dev,build]'
pytest -q
pyinstaller --noconfirm --clean --onefile --windowed --name Really-unREAL backend/gui_entry.py
./dist/Really-unREAL.exe --smoke
```

Official release workflows run the test suite, import the desktop runtime, build the one-file Windows executable, execute the packaged `--smoke` path, and only then publish the portable ZIP.

## Status

- Phase 1 — ingest/profiles: implemented
- Phase 1.5 — identity/source fusion: implemented
- Phase 2 — leakage-safe Historical Replay + generation evaluation: implemented
- Phase 3 — closed-loop Shadow Simulation: baseline implemented
- Phase 4 — persistent real-time discrete-event runtime: implemented
- 1.1 — provider-resilient behavior, context-conditioned timing, stronger person-specific generation: release-candidate stage

See `docs/QUICKSTART_GUI.md`, `docs/V1_0.md`, `docs/HISTORICAL_REPLAY.md`, `docs/TEMPORAL_HAZARD.md`, `docs/CUTOFF_RAG.md`, and `docs/SELF_TWIN.md`.

## Tests

```bash
python -m pytest
```
