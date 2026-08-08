# Really-unREAL v1.1.0-dev.2

Second 1.1 development build: context-conditioned live timing.

## What changed

- Live reply timing now observes the current clock time, weekend state, recent 15-minute conversation activity, and the gap between the last visible messages.
- The current direct relationship remains the strongest evidence source before broader platform/action backoff.
- When held-out validation shows the existing discrete hazard model is better, live timing refits and samples that hazard model on all historical REAL cases.
- Sparse relationships still receive context conditioning through a non-parametric hierarchy instead of falling straight back to one global distribution.
- Timing context is passed through the persistent runtime for both REPLY and idle INITIATE scheduling.
- The deterministic Historical Replay timing baseline remains unchanged for repeatable evaluation.

## Why

v1.0.5 fixed the constant 30-second problem by sampling historical delays, but it still ignored whether the current conversation was happening at 8am or 10pm, during an active exchange or after a long gap. dev.2 makes those observable conditions part of the live timing decision.
