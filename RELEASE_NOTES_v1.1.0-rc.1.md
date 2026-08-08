# Really-unREAL v1.1.0-rc.1

Integrated release candidate for the 1.1 behavior/generation update.

## Provider-independent behavior

- A scheduled `REPLY` / `INITIATE` survives HTTP 429/5xx, timeouts and temporary network failures.
- Simulated behavior time (`due_at`) is now immutable after scheduling.
- Provider retry time is stored separately, so an API outage cannot shift the person's modeled reply time or expose later messages to a retry.
- Permanent configuration/credential failures preserve the behavior as `BLOCKED` and expose an explicit generation retry action.

## Context-conditioned timing

- Live timing conditions on relationship, action, time of day, weekend, recent 15-minute activity and previous-message gap.
- When enough same-relationship evidence exists, visible message type (question / very short / statement) also conditions timing.
- The richer discrete hazard model is deployed only when it beats the empirical baseline on held-out validation; sparse relationships use contextual empirical backoff.
- The old fixed-median and unconditional-sampling live behavior are no longer the primary runtime path.

## Stronger person-specific generation

- Added relationship-focused style fingerprints and burst behavior profiles.
- Live generation can see at most two strictly older REAL replies as private style exemplars.
- Long near-verbatim exemplar reuse triggers anti-copy regeneration; common short expressions may recur naturally.
- Historical Replay/model-test raw response examples remain disabled by default.
- Future/held-out replies remain outside persona, retrieval and style construction.

## Evaluation clarity

- Model tests report 95% descriptive intervals for content/timing metrics when possible.
- The GUI explicitly marks the default 3-case NVIDIA run as a quick smoke check and recommends 10–20+ held-out cases for model comparison.
- Content overlap, endings, expression behavior, message splitting and timing are shown separately instead of pretending one score is overall fidelity.

## Privacy

- Raw archives, API keys and private style exemplars are not written into public result JSON, CI status, release notes or release assets.
- Hosted inference still requires explicit private-context consent.
- SIMULATION history remains separate from imported REAL evidence.
