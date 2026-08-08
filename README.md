# Really-unREAL 1.0.1

A local-first conversation-behavior digital-twin framework grounded in real message history.

Really-unREAL does **not** claim to reconstruct hidden thoughts, feelings, attraction, or intent. It models narrower, observable behavior: **when someone stays silent, when they reply or follow up, how they split messages, how they write, and which previously observed topics/events plausibly continue.**

> Make the most plausible behavior from the evidence, not the most entertaining behavior.

## Start here — desktop GUI

v1.0.1 adds a small Windows-friendly desktop UI around the v1.0 research core.

### Portable Windows build

1. Download `Really-unREAL-v1.0.1-Windows-x64.zip` from the GitHub Release or Actions artifact.
2. Extract it.
3. Run `Really-unREAL.exe`.
4. Choose a KakaoTalk export ZIP.
5. Confirm the automatically suggested **내 이름**.
6. Pick a direct-chat **대화 상대**.
7. Run **빠른 Audit (LLM 없음)** first.

The portable executable does not require a separate Python installation. It does not install a background service and it does not send messages to KakaoTalk, Instagram, or any other real platform.

For screenshots/step-by-step details, see `docs/QUICKSTART_GUI.md`.

### Run the GUI from source

Python 3.11+:

```bash
python -m pip install -e '.[dev]'
python -m backend.gui
```

Installing the package also exposes the `really-unreal` GUI entry point.

## What the GUI does

The v1.0.1 GUI intentionally focuses on the safest simple path:

```text
Kakao ZIP
   |
   v
suggest my display name -> user confirms
   |
   v
choose direct-chat target
   |
   +--> Quick Audit (no LLM, no API key)
   |
   +--> Local OpenAI-compatible model
   |
   +--> NVIDIA NIM (explicit remote-context consent required)
```

Results are aggregate metrics by default. Raw private chat text and API keys are not written to the result JSON.

The GUI does **not** replace advanced research workflows. Instagram fusion, SELF twin, dense retrieval, source ablation, Shadow Simulation, custom identity maps, and the persistent live runtime remain available through the CLI.

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
cutoff-safe persona + memory + RAG
   |
   v
language model
   |
   v
SIMULATION memory
```

The LLM never gets to talk merely because the application is running.

## Twin modes

```text
PERSON twin  -> reproduce another explicitly mapped person's observable behavior
SELF twin    -> reproduce the user's own observable behavior across relationships
```

SELF mode is relationship-aware: if enough evidence exists, behavior in the current relationship overrides broader fallbacks rather than averaging how one person talks to everyone.

## Core capabilities

- KakaoTalk text/ZIP ingestion with minute-level timestamp precision metadata.
- Meta/Instagram export ingestion and source-aware evidence fusion.
- Conservative cross-platform identity mapping.
- PERSON and SELF twin replay datasets.
- Chronological train/validation/test Historical Replay with future-leakage barriers.
- `WAIT / REPLY / INITIATE` behavior modeling.
- Interval-aware timing evaluation for coarse Kakao timestamps.
- Relationship-aware empirical timing baseline.
- Context-conditioned discrete hazard timing model with validation-gated selection.
- Relationship-conditioned language/style profiles.
- Observable topic memory and explicit event/date memory.
- Cutoff-safe historical-situation retrieval.
- Raw historical responses withheld from RAG by default to reduce nearest-neighbor copying.
- Optional dense+lexical retrieval through an embedding-provider interface.
- Provider-agnostic generation through the `BurstLanguageModel` contract.
- NVIDIA NIM and generic OpenAI-compatible local model adapters.
- Same-case Kakao-only vs Kakao+Instagram ablation.
- Closed-loop Shadow Simulation.
- Persistent SQLite-backed live discrete-event runtime.
- Strict separation of `REAL` and `SIMULATION` memories.
- Remote private-context transmission blocked unless explicitly enabled.
- No real messaging-platform auto-send API in the core.

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

Do not jump straight from an imported chat log to live simulation. Replay and shadow evaluation exist specifically to catch convincing-looking but behaviorally wrong models.

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

### Closed-loop shadow test

```bash
python -m backend.shadow_audit \
  ./data/raw/kakao_bundle.zip \
  ./data/raw/instagram_export.zip \
  ./identity.local.json \
  CONVERSATION_ID \
  --self-twin \
  --provider local \
  --model YOUR_LOCAL_MODEL
```

## Privacy model

Private data is local by default.

```text
raw archives              gitignored
identity.local.json       gitignored
SQLite simulation state   gitignored
artifacts                  gitignored
```

Loopback generation/embedding endpoints are permitted by default. Any non-loopback endpoint that receives private conversation context requires explicit remote-context consent.

The system distinguishes:

```text
REAL        observations imported from real records
SIMULATION  generated events/messages
```

Simulation output is never promoted into asserted real history.

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

## Retrieval and memory

Generation can see only cutoff-safe evidence:

```text
visible context
relationship-conditioned style profile
observable topic cues
explicit event/date cues
retrieved historical situations
response-shape statistics
```

Raw old responses are hidden by default. Topic memory means "this relationship actually discussed this", not "the person secretly likes this".

## Live runtime

`backend.simulation.runtime.LiveSimulationEngine` is the persistent local discrete-event core.

- incoming counterpart messages can replace idle initiation with a scheduled reply;
- future actions are persisted before generation;
- `WAIT` does not invoke the LLM;
- a due action invokes generation at the due time;
- overdue pending events can recover after restart;
- generated messages are stored as `SIMULATION` only;
- no real-platform sending API exists in the core.

The v1.0.1 GUI is currently a replay/audit front door; it is not yet a full messenger UI for the live runtime.

## Known limits

- Human communication is stochastic. High replay accuracy is not proof of identity reconstruction.
- INITIATE remains harder than REPLY because unobserved future events cannot be inferred from nowhere.
- Event memory is conservative, not a full calendar/NLU system.
- Dense retrieval quality depends on the embedding model and must beat the lexical floor empirically.
- Sparse relationships fall back to broader behavior statistics.
- Kakao exports cannot reveal true second-level reply timing.
- The v1.0.1 GUI currently focuses on Kakao direct-chat PERSON replay; advanced modes remain CLI-first.
- PERSON twins raise consent and impersonation concerns; SELF twin is the safer default product direction.
- Windows portable binaries are currently unsigned community builds, so SmartScreen may warn.

## Build from source

```bash
python -m pip install -e '.[dev,build]'
pytest -q
pyinstaller --noconfirm --clean --onefile --windowed --name Really-unREAL backend/gui.py
```

The Windows workflow packages `Really-unREAL.exe` and `QUICKSTART.md` into `Really-unREAL-v1.0.1-Windows-x64.zip`.

## Status

**v1.0 simulation core:** implemented.  
**v1.0.1 desktop usability layer:** in release validation.

- Phase 1 — ingest/profiles: implemented
- Phase 1.5 — identity/source fusion: implemented
- Phase 2 — leakage-safe Historical Replay + generation evaluation: implemented
- Phase 3 — closed-loop Shadow Simulation: implemented baseline
- Phase 4 — persistent real-time discrete-event runtime: implemented core
- v1.0.1 — desktop quick-start GUI + portable Windows build: validation branch

See `docs/QUICKSTART_GUI.md`, `docs/V1_0.md`, `docs/HISTORICAL_REPLAY.md`, `docs/TEMPORAL_HAZARD.md`, `docs/CUTOFF_RAG.md`, and `docs/SELF_TWIN.md`.

## Tests

```bash
python -m pytest
```
