# Really-unREAL v1.1.1

Small desktop usability patch for Live Simulation.

## Read / unread simulation state

- User-sent SIMULATION messages now start as `안읽음`.
- When a modeled `REPLY` behavior reaches its scheduled time, messages visible to that reply become `읽음 · HH:MM`.
- The read timestamp is tied to the simulated behavior time, not to model/API delivery time.
- A provider 503, timeout, or retry therefore cannot move or erase an already-modeled read state.
- Read state is persisted in local SQLite together with the simulation conversation.
- Existing <=1.1.0 simulation histories are backfilled conservatively: when an older simulated target reply already exists, preceding user messages are marked read at that reply timestamp.

## Interpretation

The read indicator is **simulation inference**, not a real KakaoTalk read receipt. Really-unREAL does not query or modify the real messaging service.

## Windows artifact

The release workflow runs the Python regression suite, imports the integrated desktop runtime, builds a one-file Windows executable, runs the packaged `--smoke` path, packages `Really-unREAL-v1.1.1-Windows-x64.zip`, and publishes the GitHub Release. Hosted NVIDIA availability remains an informational external-health signal rather than an artifact-correctness gate.
