# Cutoff-safe RAG and generation context

Phase 2C begins only after the temporal model has decided that an action should exist. The language layer must never be allowed to decide whether to talk.

## Two independent future-leakage barriers

Historical generation can leak the future through more than retrieval.

### 1. Retrieval cutoff

`backend.retrieval.CutoffExampleIndex` may be built from the full local history, but a replay query only returns examples whose real target action occurred **strictly before** the replay observation time.

```text
past examples          replay cutoff             future examples
<---------------------------|--------------------------->
          eligible          |          forbidden
```

Strict `<` is used rather than `<=` because KakaoTalk timestamps are usually minute-precision. Two messages carrying the same displayed minute do not establish a safe chronological order.

### 2. Persona cutoff

A language/style profile calculated from the entire export would also leak future behavior. `backend.persona.cutoff.build_cutoff_language_profile()` therefore uses only target messages strictly older than the replay cutoff.

The profile is source-weighted so Kakao remains primary and Instagram remains supplemental.

## Historical example unit

The retrieval index is built from Historical Replay cases. Each historical example contains:

```text
visible context before a real target action
                +
real target response burst
                +
source / timing metadata
```

At query time only the **visible current context** is used for ranking. The held-out current target burst is never read.

## Retrieval score

The dependency-free baseline ranker combines:

- Korean/ASCII token overlap;
- character-bigram cosine similarity;
- recency;
- source evidence weight;
- a small same-platform preference.

This is intentionally a baseline. A later embedding retriever must beat it under the same cutoff rules rather than silently replacing it.

## Generation packet

`backend.generation_context.build_generation_context()` produces the only information a future language model may see:

```text
chosen action from temporal policy
visible recent conversation
cutoff-safe weighted language profile
cutoff-safe retrieved historical examples
```

The caller must provide `chosen_action`. The packet builder does not inspect the real held-out replay action, preserving the separation:

```text
Temporal model
    |
    +--> WAIT  -> no generation
    |
    +--> REPLY / INITIATE
              |
              v
       GenerationContextPacket
              |
              v
             LLM
```

## Current status

The repository now has the full pre-LLM path:

```text
raw exports
 -> identity/source fusion
 -> Historical Replay
 -> empirical/hazard temporal choice
 -> cutoff-safe retrieval
 -> cutoff-safe persona snapshot
 -> generation context packet
 -> [LLM required here]
```

The next experiment needs an actual language model backend. The model should return only the message burst, not hidden emotional scores or chain-of-thought.
