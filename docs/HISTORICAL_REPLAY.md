# Historical Replay

Historical Replay is the first objective test of Really-unREAL. The system is shown only the past of a real conversation and is evaluated against what actually happened next.

The goal is **not** exact-string prediction. A real person could plausibly phrase the same response several ways. The evaluation target is observable behavior:

1. whether the person waits or acts;
2. when the action happens;
3. when the evidence is strong enough, whether the action is a reply or an initiation/follow-up;
4. how many messages are sent as one burst;
5. whether generated content is semantically and stylistically plausible.

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

## Action labels and long-gap ambiguity

Inside an active direct-message session, adjacent sender order provides a useful coarse label:

- previous visible sender is the user -> `REPLY` proxy
- previous visible sender is the target -> `INITIATE`/follow-up proxy
- no action yet at a sampled earlier checkpoint -> `WAIT`

This proxy is **not trusted after a long session gap**. If days pass between messages, the same observed sender order could represent either a late reply or a genuinely new initiation. Really-unREAL therefore keeps two separate facts:

```text
action             coarse adjacent-sender proxy
session_restart    long temporal gap
```

and marks the REPLY/INITIATE role as ambiguous when a session restarts. Those events still contribute timing evidence and WAIT negatives, but their final positive REPLY/INITIATE label is excluded from action-class evaluation and from action-conditioned timing buckets.

The first target message in an export is skipped because the interval before the export is left-censored.

Group conversations are excluded from action/timing replay by default. A group utterance is useful style evidence, but the immediately previous group message may have been addressed to somebody other than the user. `--include-group` exists for controlled diagnostics only; labels involving an unrelated third-party sender are still skipped.

## WAIT negatives

Evaluating only moments when a message exists would reward a model that always talks. Replay therefore creates `WAIT` snapshots before the real event.

Default checkpoints are approximately:

```text
1 min, 5 min, 30 min, 2 h, 6 h, 24 h
```

A WAIT point is emitted only when it lies **strictly before the earliest possible event time**. This matters for coarse timestamps.

For long-gap ambiguous events, these safe WAIT checkpoints are still usable even though the final REPLY/INITIATE class is not.

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

Timing metrics respect that interval. A prediction inside the observed interval is not penalized as if second-level ground truth existed. The empirical baseline also fits the midpoint of the feasible interval rather than treating same-minute messages as literal zero-second replies.

## Leakage prevention

Replay has three leakage barriers:

1. **Context barrier:** a case contains only messages before the held-out target burst.
2. **Temporal split:** train / validation / test are split chronologically, never randomly.
3. **Retrieval cutoff:** retrieval for a replay case only uses memories that existed before that case's observation time.

Persona snapshots follow the same cutoff rule. Computing style statistics from the full export would also leak future behavior.

## Empirical timing baseline

`backend.replay_baseline.EmpiricalTimingBaseline` is the floor model.

It deliberately knows very little:

- it derives a coarse candidate action type from the visible previous sender;
- it learns a weighted empirical timing quantile from the training split;
- if enough trustworthy samples exist, it conditions the threshold on platform and action type;
- ambiguous long-gap cases remain in the global timing distribution but do not contaminate REPLY/INITIATE-specific buckets;
- Kakao samples retain full evidence weight while Instagram DM samples are supplemental;
- it predicts `WAIT` until the learned threshold, then predicts the observable candidate action.

Timing error is interval-aware: predicting anywhere inside a censored real interval receives zero timing error.

## Hazard model

The context-conditioned discrete hazard model uses observable features such as elapsed silence, time of day, recent activity, previous gap, platform, and the coarse reply/follow-up state.

Long-gap ambiguous cases update only the action-agnostic global survival curve. They do not update action-conditioned feature cells. The richer model is selected only when it beats the empirical baseline on held-out validation and there are enough confident action events.

## CLI

After creating the gitignored identity map:

```bash
python -m backend.replay_audit \
  ./data/raw/kakao_bundle.zip \
  ./data/raw/instagram_export.zip \
  ./identity.local.json \
  person-001
```

The audit reports raw proxy counts together with confident REPLY/INITIATE counts and ambiguous long-gap counts. Held-out private message text is not printed by default.

For NVIDIA generation:

```bash
python -m backend.nvidia_replay \
  ./data/raw/kakao_bundle.zip \
  ./data/raw/instagram_export.zip \
  ./identity.local.json \
  person-001 \
  --sources kakao
```

A fair source ablation is available with:

```bash
--sources both
```

Both runs score the **same Kakao-only chronological held-out cases**. Instagram may enrich persona/RAG evidence, but it does not change the split or the cases being scored.

## Evaluation order

### 2A. Dataset validity

- parsing correctness
- no future leakage
- burst grouping
- timestamp censoring
- chronological split
- explicit long-gap action ambiguity

### 2B. Temporal modeling

- interval-aware empirical floor
- context-conditioned hazard model
- validation-based model selection
- ambiguous action labels excluded from action-specific conditioning

### 2C. Content retrieval + generation

- cutoff-safe source-aware RAG
- cutoff-safe persona snapshot
- raw historical responses withheld from the LLM by default
- content overlap and style-shape metrics reported separately

### 2D. Historical ablation

Compare at least:

```text
Kakao only
Kakao + supplemental Instagram
```

on the exact same held-out Kakao cases. If Instagram hurts replay, its weight should be reduced rather than assuming more data is always better.
