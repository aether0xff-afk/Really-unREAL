# Context-conditioned temporal hazard model

Phase 2B models *when* the target acts before any language model is allowed to generate text.

The production question is not simply "what is this person's average reply delay?". It is:

> Given the visible conversation so far and the amount of silence that has already elapsed, how likely is the person to act during the current time interval?

## Why hazard instead of one delay average

The empirical baseline learns one weighted timing threshold for reply/follow-up behavior. That is useful as a floor, but it cannot distinguish an active back-and-forth conversation from a stale thread.

`backend.replay_hazard.DiscreteHazardModel` uses discrete elapsed-time bins:

```text
0-1 min
1-5 min
5-30 min
30 min-2 h
2-6 h
6-24 h
1-3 d
3-7 d
7-30 d
30-90 d
90-365 d
```

For each bin it estimates the conditional probability of an event given that the target has remained silent up to that bin.

## Observable features only

The current model conditions on information that exists before the held-out future:

- whether the visible state implies a reply or a target follow-up;
- Kakao vs Instagram source;
- four-hour local time band;
- weekday vs weekend;
- how many visible messages remain within the most recent 15-minute activity window;
- the gap between the last two visible messages.

It does **not** read the held-out target message or any later memory.

## Sparse-history backoff

A fully specific feature combination may occur only once. Instead of trusting that cell, the model backs off through progressively broader contexts:

```text
action + platform + hour + weekend + activity + previous-gap
                       ↓
action + hour + weekend + activity + previous-gap
                       ↓
action + hour + activity
                       ↓
action + activity
                       ↓
action
                       ↓
global elapsed-bin hazard
```

Cells also receive a small prior toward the global hazard for the same elapsed-time bin.

## Kakao-primary source weighting

Replay cases retain the source evidence weight established in Phase 1.5. The hazard model uses those weights as effective risk/event mass rather than duplicating records.

Default source relevance remains:

```text
Kakao direct       1.00
Instagram direct   0.55
Kakao group        0.40
Instagram group    0.20
```

Action/timing replay still excludes group conversations by default.

## Validation-gated model selection

The richer model is **not automatically preferred**. For each person:

1. fit empirical and hazard models on the chronological training split;
2. tune the hazard action threshold on validation only;
3. require at least 50 training replay events and 10 validation events;
4. select hazard only if validation balanced accuracy beats empirical by more than 0.01;
5. inspect the test split only after the choice is frozen.

This matters because a richer model can overfit a person with a short chat history.

## Current private-data diagnostic

A local diagnostic was run against the supplied KakaoTalk direct-message history without committing raw conversations or real-person mappings.

Among 13 direct relationships with at least 20 replay events, the macro average test balanced accuracy was approximately:

```text
empirical timing baseline       0.597
hazard model                    0.669
validation-gated selection      0.678
```

The validation gate selected hazard for 6 relationships and kept the empirical fallback for 7. This is the intended behavior: complexity must earn its use per person rather than being imposed globally.

These numbers are a development diagnostic, not a final benchmark. They were computed before content/RAG modeling and should be rerun from the repository's canonical CLI after CI and identity-map finalization.

## Simulation use

The hazard model exposes two distinct outputs:

- a deterministic median predicted delay for evaluation;
- seeded sampling from the learned survival distribution for later live simulation.

The second output is important. A real person does not answer at one fixed average delay every time, so live mode should sample plausible timing rather than repeatedly scheduling the same deterministic delay.

## Next step

Phase 2C adds cutoff-safe retrieval. Only after the temporal model decides that an action exists do retrieval and an LLM receive permission to construct the message content.
