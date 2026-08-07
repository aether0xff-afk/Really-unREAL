# Really-unREAL

A local-first relationship conversation simulator grounded in real message history.

The project does **not** claim to reproduce a real person's hidden thoughts or feelings. Its target is narrower and testable: reproduce *observable conversational behavior* from supplied records — what is said, how it is said, when replies arrive, when a person initiates, and when nothing happens.

> Make the most plausible behavior from the evidence, not the most entertaining behavior.

## Core principles

- **Real time matters.** If three real days pass, three simulated days pass.
- **Silence is an action.** `WAIT` is a first-class outcome.
- **Behavior before prose.** A temporal/action model decides whether to wait, reply, or initiate before an LLM writes a message.
- **REAL and SIMULATION memories never mix.** Every stored event carries an explicit source.
- **No mind-reading scores.** Model observable signals such as reply delay, initiation rate, topic continuation, and message style rather than fictional affection percentages.
- **Local-first.** Private conversations stay on the user's device by default. `data/` and private identity mappings are gitignored.
- **Source-aware.** KakaoTalk, Instagram DMs, and social activity remain distinguishable so one context does not silently overwrite another.
- **Kakao-primary.** KakaoTalk is the primary source for persona and temporal behavior. Instagram is supplemental evidence used to fill gaps and add cross-platform context, not to override stable Kakao-derived behavior.
- **Conservative identity resolution.** Fuzzy name similarity may suggest a match, but never silently merges two real people.

## Phase 1 / 1.5 — implemented scaffold

The current pipeline can:

1. Parse KakaoTalk text exports and ZIP bundles.
2. Parse Meta/Instagram information-download ZIPs, including DMs and activity counts.
3. Normalize messages into a stable schema with source metadata.
4. Extract language/style and temporal statistics.
5. Suggest cross-platform identity candidates without auto-merging ambiguous names.
6. Fuse approved aliases into stable local person IDs while preserving source/context relevance.
7. Expose local audit tools without committing private source data.

KakaoTalk text analysis:

```bash
python -m backend.cli ./data/raw/chat.txt --target "상대방 이름"
```

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

Then copy `examples/identity.example.json` to the gitignored `identity.local.json`, review aliases, and inspect one person's fused evidence:

```bash
python -m backend.fusion_audit \
  ./data/raw/kakao_bundle.zip \
  ./data/raw/instagram_export.zip \
  ./identity.local.json \
  person-001
```

Run tests with:

```bash
python -m pytest
```

## Evidence hierarchy

For a simulated relationship, not all observations have equal behavioral relevance. The default ordering is deliberately Kakao-first:

```text
Kakao 1:1 with the user           1.00   primary
Instagram DM with the user        0.55   supplemental
Kakao group conversation          0.40   supporting style/context
Instagram group conversation      0.20   weak supporting context
posts / stories / comments        contextual evidence only
likes / saves / follows           weak preference signals only
```

These are starting relevance weights, not calibrated probabilities and not relationship scores. Historical Replay is responsible for validating or tuning them. Instagram should help when Kakao evidence is sparse, but should not outweigh a stable behavior pattern observed repeatedly in KakaoTalk.

The system should never turn follows, likes, or engagement into claims about hidden feelings toward a person.

## Architecture

```text
Kakao / Instagram / other records
              |
              v
        source-aware ingest
              |
              v
      identity resolution
   (explicit local person IDs)
              |
              v
        evidence fusion  ---> immutable REAL memory
              |
              +------> language profile
              |
              +------> temporal profile
              |
              +------> contextual / interest evidence
                              |
real clock ------------------+----> action policy: WAIT / REPLY / INITIATE
                                           |
                                           v
                                    memory retrieval
                                           |
                                           v
                                    message generator
                                           |
                                           v
                                  SIMULATION memory
```

The temporal/action layer sits **above** the language model. The model should not generate a message merely because the application is running.

## Roadmap

- **Phase 1:** parsing + observable profiles
- **Phase 1.5:** source fusion and per-person identity resolution
- **Phase 2:** historical replay (hide the real continuation and predict action/timing)
- **Phase 3:** shadow simulation against a past time interval
- **Phase 4:** live real-time simulation with spontaneous initiation and long-term memory

See `docs/IDENTITY_AND_FUSION.md` and the other documents in `docs/` for the detailed design and evaluation plan.
