# Historical Replay

Historical Replay is the first objective test of Really-unREAL. The system is shown only the past of a real conversation and is evaluated against what actually happened next.

The goal is **not** exact-string prediction. A real person could plausibly phrase the same response several ways. The first evaluation target is observable behavior:

1. whether the person waits or acts;
2. whether the action is a reply or an initiation/follow-up;
3. when the action happens;
4. how many messages are sent as one burst;
5. later, whether the generated content is semantically and stylistically plausible.

## Replay case

`backend.replay.ReplayCase` contains:

```text
visible past context
        |
        +---- observation_end
        |
        |     hidden interval
        |
        +---- action_at
                 |
                 +---- held-out target message burst
```

The target burst is never included in the visible context.

## Message bursts

Consecutive messages from the target are grouped when they are no more than 120 seconds apart and no other speaker intervenes. This preserves behavior such as:

```text
야
근데
그거 됨?
```

as one conversational action instead of pretending it is three independent decisions.

If the same person sends another message after a larger gap without receiving a new user message, that becomes a new `INITIATE`/follow-up event.

## Action labels

For direct conversations:

- previous visible sender is the user -> `REPLY`
- previous visible sender is the target -> `INITIATE`
- no action yet at a sampled earlier checkpoint -> `WAIT`

The first target message in an export is skipped because the interval before the export is left-censored.

Group conversations are excluded from action/timing replay by default. A group utterance is useful style evidence, but the immediately previous group message may have been addressed to somebody other than the user. `--include-group` exists for controlled diagnostics only; labels involving an unrelated third-party sender are still skipped.

## WAIT negatives

Evaluating only moments when a message exists would reward a model that always talks. Replay therefore creates `WAIT` snapshots before the real event.

Default checkpoints are approximately:

```text
1 min, 5 min, 30 min, 2 h, 6 h, 24 h
```

A WAIT point is emitted only when it lies **strictly before the earliest possible event time**. This matters for coarse timestamps.

## Timestamp precision and interval censoring

KakaoTalk exports normally contain minute-level timestamps. A line displayed as `12:03` does not justify pretending the event occurred at `12:03:00.000`.

Really-unREAL stores:

```text
KakaoTalk timestamp precision: 60 s
Instagram timestamp precision: 0.001 s
```

Therefore an observed Kakao delay of 120 seconds is represented approximately as:

```text
observed: 120 s
possible interval: 60 .. 180 s
```

Timing metrics respect that interval. A prediction inside the observed interval is not penalized as if second-level ground truth existed.

## Leakage prevention

Replay has three leakage barriers:

1. **Context barrier:** a case contains only messages before the held-out target burst.
2. **Temporal split:** train / validation / test are split chronologically, never randomly.
3. **Retrieval cutoff:** when RAG is added, retrieval for a replay case must only use memories that existed before that case's observation time. Building one vector index from the entire future conversation and retrieving from it would invalidate the experiment.

The current implementation covers the first two barriers in the replay dataset. The retrieval layer must enforce the third before language-model evaluation is considered valid.

## Empirical timing baseline

`backend.replay_baseline.EmpiricalTimingBaseline` is the current floor model.

It deliberately knows very little:

- it derives the candidate action type from the **visible previous sender**;
- it learns a weighted empirical timing quantile from the training split;
- if enough samples exist, it conditions the threshold on platform and action type;
- Kakao samples retain full evidence weight while Instagram DM samples are supplemental;
- it predicts `WAIT` until the learned threshold, then predicts the observable candidate action.

This is not intended to be a sophisticated human-timing model. Its purpose is to establish a reproducible floor. A later survival/hazard model is useful only if it improves held-out replay rather than merely sounding more complicated.

Timing error is interval-aware: predicting anywhere inside a censored real interval receives zero timing error.

## CLI

After creating the gitignored identity map:

```bash
python -m backend.replay_audit \
  ./data/raw/kakao_bundle.zip \
  ./data/raw/instagram_export.zip \
  ./identity.local.json \
  person-001
```

The command prints:

- replay event counts;
- reply/initiation counts;
- WAIT/action snapshot counts;
- chronological train/validation/test sizes;
- the empirical baseline's learned timing thresholds;
- held-out test action accuracy/recall and interval-aware timing error.

Held-out private message text is not printed by default.

## Evaluation order

Phase 2 is intentionally staged:

### 2A. Dataset validity — implemented

- parsing correctness
- no future leakage
- sensible reply/initiation labels
- burst grouping
- timestamp censoring
- chronological split

### 2B. Temporal baseline — empirical floor implemented

The weighted empirical quantile baseline is implemented. The next temporal model should be a proper survival/hazard model conditioned on observable context such as time of day, elapsed silence, recent session structure, and platform. It must beat the empirical floor on held-out later data.

### 2C. Content retrieval + generation

Only after temporal behavior is valid do we add cutoff-safe source-aware RAG and a language model. KakaoTalk remains primary evidence; Instagram stays supplemental.

### 2D. Historical ablation

Compare at least:

```text
Kakao only
Kakao + supplemental Instagram
Instagram only (diagnostic, not production default)
```

If Instagram hurts held-out Kakao replay, its weight should be reduced rather than assuming more data is always better.
