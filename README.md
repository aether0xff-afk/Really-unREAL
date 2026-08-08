# Really-unREAL

A local-first real-time conversation behavior simulator grounded in real message history.

The project does **not** claim to reproduce a real person's hidden thoughts, feelings, or identity. Its target is narrower and testable: reproduce *observable communication behavior* from supplied records — what is said, how it is said, when replies arrive, when a person follows up or initiates, and when nothing happens.

> Make the most plausible behavior from the evidence, not the most entertaining behavior.

## Core principles

- **Real time matters.** If three real days pass, three simulated days pass.
- **Silence is an action.** `WAIT` is a first-class outcome.
- **Behavior before prose.** A temporal/action model decides whether to wait, reply, or initiate before an LLM writes a message.
- **REAL and SIMULATION memories never mix.** Every stored event carries an explicit source.
- **No mind-reading scores.** Model observable signals such as reply delay, initiation/follow-up rate, topic continuation, and message style rather than fictional affection percentages.
- **Local-first.** Private conversations and identity mappings are gitignored. Remote generation/embedding is optional and must be an explicit privacy choice.
- **Source-aware.** KakaoTalk, Instagram DMs, and social activity remain distinguishable so one context does not silently overwrite another.
- **Kakao-primary.** KakaoTalk is the primary source for persona and temporal behavior. Instagram is supplemental evidence used to fill gaps and add cross-platform context, not to override stable Kakao-derived behavior.
- **Conservative identity resolution.** Fuzzy name similarity may suggest a match, but never silently merges two real people.
- **Replay before vibes.** New modeling choices must improve held-out historical behavior, not merely produce subjectively convincing chat samples.
- **No automatic impersonation boundary.** Simulation output is not silently sent as the real user or another person.

## Twin modes

Really-unREAL now treats two targets as first-class modes:

```text
PERSON twin  -> reproduce another explicitly mapped person's observable behavior
SELF twin    -> reproduce the user's own observable communication behavior
```

`SELF_TWIN` is not just PERSON mode with a different name. A user's export contains many relationships, and the same person can talk and respond differently to different people. The current implementation therefore conditions both persona and timing on the active relationship when enough evidence exists, while preserving global fallbacks.

See `docs/SELF_TWIN.md`.

## Implemented pipeline

The current pipeline can:

1. Parse KakaoTalk text exports and ZIP bundles.
2. Parse Meta/Instagram information-download ZIPs, including DMs and activity counts.
3. Normalize messages into a stable schema with source metadata and timestamp precision.
4. Suggest cross-platform identity candidates without auto-merging ambiguous names.
5. Fuse approved aliases into stable local person IDs while preserving source/context relevance.
6. Build leakage-safe Historical Replay events with `WAIT / REPLY / INITIATE`, timing intervals, ambiguity flags, and message bursts.
7. Build the same direct replay dataset for another-person twins and `SELF_TWIN`.
8. Split replay chronologically into train / validation / test.
9. Fit a weighted empirical timing baseline with relationship -> platform/action -> global backoff.
10. Fit a context-conditioned discrete hazard model and tune it on validation only.
11. Fall back to the empirical model when the richer hazard model lacks enough history or fails to beat validation.
12. Build cutoff-safe language/persona snapshots, with current-relationship evidence boosted instead of averaging every relationship equally.
13. Build cutoff-safe observable topic memory for long-horizon continuity and spontaneous initiation support.
14. Retrieve historical situations while withholding raw historical responses by default to reduce nearest-neighbour copying.
15. Optionally blend dense embedding similarity with the dependency-free lexical retriever while preserving the same temporal cutoff.
16. Generate message bursts through the NVIDIA NIM adapter and evaluate held-out content/style separately from timing.
17. Compare Kakao-only vs Kakao+Instagram on the **same** Kakao held-out cases.
18. Expose local audit/replay tools without committing private source data.

## Quick commands

KakaoTalk archive audit:

```bash
python -m backend.audit ./data/raw/kakao_bundle.zip
```

Cross-platform identity candidates:

```bash
python -m backend.identity_audit \
  ./data/raw/kakao_bundle.zip \
  ./data/raw/instagram_export.zip
```

After reviewing a gitignored `identity.local.json`, Historical Replay for another person:

```bash
python -m backend.replay_audit \
  ./data/raw/kakao_bundle.zip \
  ./data/raw/instagram_export.zip \
  ./identity.local.json \
  person-001
```

NVIDIA generation replay for another person:

```bash
python -m backend.nvidia_replay \
  ./data/raw/kakao_bundle.zip \
  ./data/raw/instagram_export.zip \
  ./identity.local.json \
  person-001 \
  --sources both
```

The same pipeline as a self digital twin:

```bash
python -m backend.nvidia_replay \
  ./data/raw/kakao_bundle.zip \
  ./data/raw/instagram_export.zip \
  ./identity.local.json \
  --self-twin \
  --sources both
```

Optional OpenAI-compatible dense retrieval, preferably through a local embedding endpoint:

```bash
python -m backend.nvidia_replay \
  ./data/raw/kakao_bundle.zip \
  ./data/raw/instagram_export.zip \
  ./identity.local.json \
  --self-twin \
  --embedding-base-url http://127.0.0.1:1234/v1 \
  --embedding-model YOUR_LOCAL_EMBEDDING_MODEL
```

Run tests with:

```bash
python -m pytest
```

## Evidence hierarchy

For a simulated relationship, not all observations have equal behavioral relevance. The default ordering is deliberately Kakao-first:

```text
Kakao 1:1                        1.00   primary
Instagram DM                     0.55   supplemental
Kakao group conversation         0.40   supporting style/context
Instagram group conversation     0.20   weak supporting context
posts / stories / comments       contextual evidence only
likes / saves / follows          weak contextual signals only
```

These are starting relevance weights, not calibrated probabilities and not relationship scores. Historical Replay is responsible for validating or tuning them. The system should never turn follows, likes, or engagement into claims about hidden feelings toward a person.

## Historical Replay

Replay hides a real continuation and asks the simulator to reproduce observable behavior rather than an exact string.

```text
visible past
    |
    +---- observation time
    |          |
    |        WAIT?
    |          |
    +----------+---- real event
                      |
                      +---- REPLY / INITIATE proxy
                      +---- ambiguity flag for long gaps
                      +---- timing interval
                      +---- message burst
                      +---- held-out content
```

KakaoTalk minute timestamps are treated as interval-censored rather than fake second-level truth. Long-gap sender order is not forced into a trustworthy `REPLY` or `INITIATE` label. Direct conversations are the default action/timing benchmark; group messages remain supporting persona/context evidence unless explicitly requested.

The empirical timing floor now supports relationship-specific thresholds when enough confident history exists. The richer temporal model is a discrete survival/hazard model conditioned on visible conversation activity, time of day, weekday/weekend, source, and recent message gaps. The richer model is used only when chronological validation proves that it earns the extra complexity.

A private-data development diagnostic across 13 sufficiently populated direct relationships previously improved macro test balanced accuracy from roughly `0.597` for the empirical baseline to `0.669` for hazard alone and `0.678` with validation-gated per-person selection. These are development diagnostics, not final benchmark claims, and should be rerun after material timing-model changes.

## Retrieval, persona, and long-term continuity

The generation layer is intentionally prevented from receiving the held-out future.

```text
cutoff-safe visible context
        +
relationship-conditioned style profile
        +
observable topic-memory cues
        +
cutoff-safe retrieved situations
        +
response-shape statistics (raw old replies hidden by default)
        |
        v
       LLM
```

The dependency-free retriever uses lexical/character similarity as a floor. An optional embedding provider can add dense context similarity. Only historical **context** is embedded for ranking; target response text is not used as a semantic shortcut.

Topic memory stores compact observable cues — recency, mention count, relationship-specific mentions, and last-seen time — rather than turning conversation topics into inferred interests or motives.

## Architecture

```text
Kakao / Instagram / other records
              |
              v
        source-aware ingest
              |
              v
      identity resolution
              |
              +--> PERSON target
              +--> SELF target
              |
              v
        evidence fusion  ---> immutable REAL memory
              |
              +------> relationship-conditioned language profile
              +------> observable topic memory
              +------> Historical Replay
              |             |
              |             +--> relationship-aware empirical timing
              |             +--> context-conditioned hazard
              |
real clock ---+-----------> action policy: WAIT / REPLY / INITIATE proxy
                                           |
                                           v
                              cutoff-safe hybrid retrieval
                                           |
                                           v
                                    message generator
                                           |
                                           v
                                  SIMULATION memory
```

The temporal/action layer sits **above** the language model. The model should not generate a message merely because the application is running.

## Current limits

The repository now addresses several earlier weaknesses, but these remain open:

- Dense semantic retrieval is supported structurally but still needs real held-out benchmarking against the lexical floor with a chosen embedding model.
- Topic memory is an observable token-level continuity model, not full event understanding. Calendar-like commitments, shared plans, and evolving projects need a richer event-memory layer.
- `INITIATE` remains harder than `REPLY`; the system can constrain initiation toward observed topics but cannot know an unobserved future event.
- The richer hazard model is not yet relationship-conditioned as deeply as the empirical fallback; sparse per-relationship hazard features need careful validation before adding identity-specific cells.
- A fully local LLM/embedding configuration is still needed for the strongest privacy mode; hosted endpoints receive the context intentionally sent to them.
- Phase 3 shadow simulation is still required before any claim of convincing long-running live behavior.

## Roadmap

- **Phase 1:** parsing + observable profiles — implemented
- **Phase 1.5:** source fusion and per-person identity resolution — implemented
- **Phase 2A:** Historical Replay dataset/labels/splits — implemented
- **Phase 2B:** weighted empirical timing baseline — implemented
- **Phase 2B.1:** context-conditioned survival/hazard timing model — implemented
- **Phase 2C:** cutoff-safe RAG + persona + NVIDIA language generation — implemented baseline
- **Phase 2C.1:** relationship-conditioned persona + observable topic memory — implemented baseline
- **Phase 2C.2:** optional dense semantic retrieval backend — implemented, benchmark pending
- **Phase 2D:** same-case Kakao-only vs Kakao+Instagram ablation runner — implemented, larger private benchmark pending
- **Phase 2E:** SELF_TWIN replay/generation mode — implemented baseline
- **Phase 3:** shadow simulation against a past time interval — next major milestone
- **Phase 4:** live real-time simulation with spontaneous initiation and long-term event memory

See `docs/IDENTITY_AND_FUSION.md`, `docs/HISTORICAL_REPLAY.md`, `docs/TEMPORAL_HAZARD.md`, `docs/CUTOFF_RAG.md`, and `docs/SELF_TWIN.md` for the detailed design and evaluation contract.
