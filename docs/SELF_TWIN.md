# SELF_TWIN

Really-unREAL can target either another explicitly mapped person or the user themself.

`SELF_TWIN` is not a claim that the system reconstructs a mind, identity, or hidden internal state. It is a narrower behavioral model:

```text
observable message history
        |
        +--> when I usually answer this relationship
        +--> how I write to this relationship
        +--> what observable topics recur
        +--> whether I wait / reply / follow up
        |
        v
plausible future communication behavior
```

## Why SELF_TWIN is a first-class mode

A self twin is not just an other-person twin with a different name. The user's own export contains many relationships, and the same person may communicate differently with each one.

The current implementation therefore adds three relationship-aware fallbacks:

1. **Direct replay is target-relative.** In a direct chat, a message after the counterpart is a `REPLY` whether the target is another person or `self`.
2. **Relationship-conditioned persona.** Global writing history remains a fallback, while messages from the current conversation receive additional evidence weight.
3. **Relationship-conditioned timing.** When one conversation has enough confident events, its empirical timing threshold is used before platform/action/global fallback.

These weights and minimum sample counts are behavioral hyperparameters, not relationship scores. Historical Replay should tune them.

## Observable topic memory

Spontaneous initiation is harder than replying because there may be no immediate user message to answer.

Rather than inventing a hidden motive, `backend.topic_memory` builds cutoff-safe topic cues from actual past conversation text. It stores only compact observable features such as:

- token / topic cue;
- recency-weighted score;
- number of observed mentions;
- number of mentions in the current relationship;
- last observed timestamp.

The generator is instructed to use these cues only for continuity. A topic cue is not evidence that the person secretly likes, wants, or intends anything.

## Semantic retrieval

The default retriever is dependency-free lexical similarity. `CutoffExampleIndex` can now optionally accept an embedding provider and blend dense similarity with the lexical floor.

Only historical **context** is embedded for ranking. Held-out target responses are never embedded into the query ranker, and the strict replay cutoff still applies.

A generic OpenAI-compatible embedding adapter is available in `backend.providers.embeddings`. A local endpoint is preferred when private conversation text must remain on-device.

## Running a self replay

```bash
python -m backend.nvidia_replay \
  ./data/raw/kakao_bundle.zip \
  ./data/raw/instagram_export.zip \
  ./identity.local.json \
  --self-twin \
  --sources both \
  --limit 20
```

The identity map must contain exactly one `is_self=true` person.

Optional dense retrieval:

```bash
python -m backend.nvidia_replay \
  ./data/raw/kakao_bundle.zip \
  ./data/raw/instagram_export.zip \
  ./identity.local.json \
  --self-twin \
  --embedding-base-url http://127.0.0.1:1234/v1 \
  --embedding-model YOUR_LOCAL_EMBEDDING_MODEL
```

If an embedding endpoint is remote rather than local, the retrieval context sent for embedding leaves the device. NVIDIA hosted generation likewise receives the generation packet. Fully local inference remains the strongest privacy configuration.

## Product boundary

The safest commercial interpretation of SELF_TWIN is a private communication assistant or simulation environment, for example:

- predict whether the user would answer now or later;
- draft a response in the user's relationship-specific style;
- replay old periods to measure fidelity before enabling live behavior;
- simulate the user's own communication habits without claiming hidden mental state.

Automatic impersonation or undisclosed autonomous sending is intentionally outside the current design. Generated simulation memory must remain distinct from REAL history.
