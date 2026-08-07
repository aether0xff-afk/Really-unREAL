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

## Phase 1 — implemented scaffold

Phase 1 turns an exported KakaoTalk text file into measurable profiles:

1. Parse sender, timestamp, and message text.
2. Normalize messages into a stable schema.
3. Extract language/style statistics.
4. Extract temporal behavior such as active hours, reply-delay distributions, and initiation patterns.
5. Expose a small CLI for inspection.

```bash
python -m backend.cli ./data/raw/chat.txt --target "상대방 이름"
```

Run tests with:

```bash
python -m pytest
```

## Architecture

```text
real message history
        |
        v
     ingest  ---> immutable REAL memory
        |
        +------> language profile
        |
        +------> temporal profile
                        |
real clock ------------+----> action policy: WAIT / REPLY / INITIATE
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
- **Phase 2:** historical replay (hide the real continuation and predict action/timing)
- **Phase 3:** shadow simulation against a past time interval
- **Phase 4:** live real-time simulation with spontaneous initiation and long-term memory

See `docs/` for the detailed design and evaluation plan.
