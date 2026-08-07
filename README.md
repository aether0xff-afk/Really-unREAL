# Really-unREAL

A local-first relationship conversation simulator grounded in real message history.

The project does **not** try to claim what a real person secretly thinks or feels. Its target is narrower and testable: reproduce *observable conversational behavior* from supplied records — what is said, how it is said, when replies arrive, when a person initiates, and when nothing happens.

> Make the most plausible behavior from the evidence, not the most entertaining behavior.

## Core ideas

- **Real time matters.** If three real days pass, three simulated days pass.
- **Silence is an action.** `WAIT` is a first-class outcome.
- **Behavior before prose.** A temporal/action model decides whether to wait, reply, or initiate before an LLM writes a message.
- **Real and simulated memories never mix.** Every stored event carries an explicit source.
- **No mind-reading scores.** We model observable signals such as reply delay, initiation rate, topic continuation, and message style rather than fictional affection percentages.
- **Local-first.** Private conversations should stay on the user's device by default.

## Initial milestone

Phase 1 turns an exported KakaoTalk text file into measurable profiles:

1. Parse sender, timestamp, and message text.
2. Normalize the conversation into a stable schema.
3. Extract language/style statistics.
4. Extract temporal behavior such as active hours, reply-delay distributions, and initiation patterns.
5. Evaluate later models with historical replay rather than subjective "feels similar" judgments.

The next milestone will add historical replay: hide a real continuation, predict the next action/message timing, then compare with what actually happened.

## Planned architecture

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

See `docs/` for the detailed design.

## Status

Repository bootstrap in progress.
