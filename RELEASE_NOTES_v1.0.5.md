# Really-unREAL v1.0.5

v1.0.5 fixes the live-chat timing behavior that could schedule the same reply delay on every turn.

## Fixed

- Live chat no longer reuses one deterministic median reply delay such as 30 seconds for every message.
- Reply and initiation delays are sampled from the selected person's observed Historical Replay timing intervals.
- Relationship-specific conversation timing is preferred, then platform/action evidence, then broader action evidence.
- KakaoTalk same-minute events keep their timestamp uncertainty: an observed `[0, 60]` second reply interval is sampled inside that interval instead of being collapsed to the 30-second midpoint.
- Longer observed reply gaps remain part of the empirical distribution, so minutes or hours can occur when they are present in the person's history.
- Ambiguous REPLY-vs-INITIATE events are not used to teach action-specific timing.

## What did not change

- The deterministic empirical median baseline is still used for benchmark/evaluation paths where repeatability is useful.
- Scheduled live events are still persisted locally before generation.
- Hosted NVIDIA use still requires explicit private-context consent.
- Simulation messages remain `SIMULATION` and are never promoted to imported REAL history.

## Validation

The release workflow runs the full test suite, imports the live GUI and timing sampler, builds the Windows one-file executable, executes its packaged `--smoke` path, packages the portable ZIP, and publishes the release only if every gate succeeds.
