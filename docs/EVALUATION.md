# Evaluation

Subjective similarity is useful for demos but insufficient for development. Really-unREAL uses held-out historical behavior as the primary evaluation target.

## Phase 2: Historical Replay

Choose cut points in real conversations. For each cut point, hide the continuation from the model and provide only the past.

Evaluate separately:

### Action prediction

Did the model predict the correct broad next behavior?

- WAIT
- REPLY
- INITIATE

### Timing calibration

Compare the actual delay with the model's predicted distribution rather than demanding an exact timestamp.

Possible metrics:

- negative log likelihood of observed delay
- interval coverage
- median absolute timing error
- calibration by time bucket

### Text behavior

Exact-text matching is inappropriate. Compare observable features and retrieval consistency:

- length distribution
- bubble splitting
- lexical/style statistics
- semantic similarity to the held-out response
- whether retrieved examples are genuinely similar historical situations

## Phase 3: Shadow Simulation

Start from a past date and replay wall-clock time while hiding all future records. Compare generated session starts, replies, silence intervals, and timing against what really happened.

The main failure mode to watch is **overactivity**: an entertaining model that sends far more messages than the real record.

## Data leakage rule

No held-out continuation may enter retrieval, profile extraction, prompt construction, or temporal statistics for the corresponding evaluation fold. Time-based train/test splits are preferred over random message splits.

## Success criterion

The simulator improves only when it predicts unseen observable behavior better. A message that merely sounds convincing is not enough.
