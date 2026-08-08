# Really-unREAL 1.0

A local-first conversation-behavior digital-twin framework grounded in real message history.

Really-unREAL does **not** claim to reconstruct hidden thoughts, feelings, attraction, or intent. Its target is narrower and testable: observable communication behavior — **when someone stays silent, when they reply or follow up, how they split messages, what language style they use, and which previously observed topics/events plausibly continue.**

> Make the most plausible behavior from the evidence, not the most entertaining behavior.

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

SELF mode is relationship-aware: if enough evidence exists, the user's style and reply timing for the current conversation override broader fallbacks. This avoids averaging together how one person talks to every friend.

## v1.0 capabilities

- KakaoTalk text/ZIP ingestion with minute-level timestamp precision metadata.
- Meta/Instagram export ingestion and source-aware evidence fusion.
- Conservative cross-platform identity mapping.
- PERSON and SELF twin replay datasets.
- Chronological train/validation/test Historical Replay with future-leakage barriers.
- `WAIT / REPLY / INITIATE` behavior modeling; long-gap action roles may remain ambiguous rather than being falsely labeled.
- Interval-aware timing evaluation for coarse Kakao timestamps.
- Relationship-aware empirical timing baseline.
- Context-conditioned discrete hazard model, selected only when held-out validation beats the simpler baseline.
- Relationship-conditioned language/style profiles.
- Observable topic memory.
- Observable event/date memory from explicit past mentions.
- Cutoff-safe historical-situation retrieval.
- Raw historical responses withheld from RAG by default to reduce nearest-neighbor copying.
- Optional dense+lexical retrieval through an embedding-provider interface.
- Provider-agnostic generation replay through the `BurstLanguageModel` contract.
- NVIDIA NIM and generic OpenAI-compatible local model adapters.
- Same-case Kakao-only vs Kakao+Instagram ablation.
- **Closed-loop Shadow Simulation:** hidden target messages are replaced by simulated messages and the resulting drift is scored.
- **Persistent live runtime:** future actions are stored in SQLite, survive application restarts, and generate text only when due.
- `REAL` and `SIMULATION` memories remain separate.
- Remote private-context transmission is blocked unless explicitly enabled.
- The core never automatically sends messages to a real messaging platform.

## Recommended workflow

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

Do not jump straight from an imported chat log to live simulation. Replay and shadow evaluation exist specifically to catch a convincing-looking but behaviorally wrong model.

## Quick start

Create and review a gitignored `identity.local.json` first.

### 1. Audit behavior

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

### 2. Generate/evaluate with a local model

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

The default local endpoint is `http://127.0.0.1:1234/v1`, suitable for an OpenAI-compatible local server such as LM Studio or another compatible runtime.

### 3. Optional hosted NVIDIA generation

Hosted generation contains private context. It therefore requires explicit consent:

```bash
python -m backend.replay_generate \
  ./data/raw/kakao_bundle.zip \
  ./data/raw/instagram_export.zip \
  ./identity.local.json \
  person-001 \
  --provider nvidia \
  --allow-remote-private-context
```

### 4. Closed-loop shadow test

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

Shadow Simulation keeps real counterpart messages as external input, hides the target's future, feeds simulated target messages into later context, and only then compares the resulting trajectory with reality.

### 5. Optional local dense retrieval

```bash
python -m backend.replay_generate \
  ./data/raw/kakao_bundle.zip \
  ./data/raw/instagram_export.zip \
  ./identity.local.json \
  --self-twin \
  --provider local \
  --model YOUR_LOCAL_MODEL \
  --embedding-base-url http://127.0.0.1:1234/v1 \
  --embedding-model YOUR_EMBEDDING_MODEL
```

Only historical **context** is embedded for ranking; held-out or historical response text is not used as a semantic shortcut.

## Privacy model

Private data is local by default.

```text
raw archives              gitignored
identity.local.json       gitignored
SQLite simulation state   gitignored
artifacts                  gitignored
```

Loopback generation/embedding endpoints are permitted by default. Any non-loopback endpoint that receives private conversation context requires `--allow-remote-private-context`.

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

Train/validation/test are chronological, not random. Kakao minute timestamps are represented as feasible timing intervals rather than fake second-level truth.

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

Raw old responses are hidden by default. Topic memory means "this relationship actually discussed this", not "the person secretly likes this". Event memory means "an event/date was mentioned", not "the event definitely happened".

## Live runtime

`backend.simulation.runtime.LiveSimulationEngine` is the v1.0 local discrete-event core.

- incoming counterpart messages replace idle initiation with a scheduled reply;
- future actions are persisted before generation;
- `WAIT` does not invoke the LLM;
- a due action invokes generation at the due time;
- after restart, overdue pending events can be recovered;
- generated messages are stored as `SIMULATION` only;
- no real-platform sending API exists in the core.

The runtime is a framework component, not a finished messenger UI.

## Evaluation philosophy

A plausible screenshot is not a benchmark.

Really-unREAL evaluates timing, event existence, burst size, lexical overlap, token overlap, endings, question/laugh/cry behavior, and closed-loop event drift separately. More complex timing models are selected on validation only and do not earn credit merely for being sophisticated.

A previous private development diagnostic across 13 sufficiently populated direct relationships produced roughly `0.597` macro balanced accuracy for the empirical timing floor, `0.669` for the hazard model alone, and `0.678` with validation-gated per-person model selection. These remain development diagnostics, not universal performance claims, and should be rerun after material model changes.

## Known limits

- Human communication is stochastic. High replay accuracy is not proof of identity reconstruction.
- INITIATE remains harder than REPLY because unobserved future events cannot be inferred from nowhere.
- Event memory is conservative, not a full calendar/NLU system.
- Dense retrieval quality depends on the embedding model and must beat the lexical floor empirically.
- Sparse relationships fall back to broader behavior statistics.
- Kakao exports cannot reveal true second-level reply timing.
- The live core is not yet a polished mobile/desktop product UI.
- PERSON twins raise consent and impersonation concerns; SELF twin is the safer default product direction.

## Status

**v1.0 simulation core:** implemented.

- Phase 1 — ingest/profiles: implemented
- Phase 1.5 — identity/source fusion: implemented
- Phase 2 — leakage-safe Historical Replay + generation evaluation: implemented
- Phase 3 — closed-loop Shadow Simulation: implemented baseline
- Phase 4 — persistent real-time discrete-event runtime: implemented core

See `docs/V1_0.md`, `docs/HISTORICAL_REPLAY.md`, `docs/TEMPORAL_HAZARD.md`, `docs/CUTOFF_RAG.md`, and `docs/SELF_TWIN.md`.

## Tests

```bash
python -m pytest
```
