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
- **Local-first.** Private conversations stay on the user's device by default. `data/` is gitignored.
- **Source-aware.** KakaoTalk, Instagram DMs, and public/social activity are kept distinguishable so evidence from one context does not silently overwrite another.

## Phase 1 — implemented scaffold

Phase 1 turns exported messenger/social data into measurable profiles:

1. Parse KakaoTalk text exports and ZIP bundles.
2. Parse Meta/Instagram information-download ZIPs, including DMs and activity counts.
3. Normalize messages into a stable schema with source metadata.
4. Extract language/style statistics.
5. Extract temporal behavior such as active hours, reply-delay distributions, and initiation patterns.
6. Expose local audit/inspection tools without committing private source data.

KakaoTalk text analysis:

```bash
python -m backend.cli ./data/raw/chat.txt --target "상대방 이름"
```

KakaoTalk archive audit:

```bash
python -m backend.audit ./data/raw/kakao_bundle.zip
```

Run tests with:

```bash
python -m pytest
```

## Evidence hierarchy

For a simulated relationship, not all observations should have equal weight:

```text
1:1 conversation with the user   highest behavioral relevance
small-group conversation          supporting evidence
large-group conversation          general style/context only
Instagram DM                      direct cross-platform evidence
posts / stories / comments        self-presentation and interests
likes / saves / follows           weak preference signals only
```

The system should never turn follows, likes, or engagement into claims about hidden feelings toward a person.

## Architecture

```text
Kakao / Instagram / other records
              |
              v
        source-aware ingest  ---> immutable REAL memory
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

See `docs/` for the detailed design and evaluation plan.
