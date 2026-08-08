# Really-unREAL v1.1.0

Really-unREAL 1.1 turns the first live-chat desktop prototype into a more coherent behavior simulation runtime.

## Highlights

- **Provider-independent behavior:** an HTTP 503, timeout, or temporary local-server failure can no longer erase a reply the behavior model already scheduled.
- **Separate behavior and delivery clocks:** the simulated reply time stays immutable while provider generation retries use a separate clock, preventing outage-induced timing shifts and later-context leakage.
- **Context-conditioned live timing:** reply timing can depend on relationship, action, time of day, weekday/weekend, recent activity, recent-message gap, and supported question/short/statement context.
- **Validation-gated hazard timing:** the richer hazard model is deployed only when it improves held-out validation; sparse histories use contextual empirical backoff.
- **Stronger person-specific generation:** relationship-focused style fingerprints and burst profiles provide more than simple length/`ㅋㅋ` statistics.
- **Private style exemplars with anti-copy guard:** Live mode may expose at most two strictly older REAL responses for style only; long near-verbatim reuse triggers regeneration.
- **Clearer evaluation:** model tests separate content overlap, endings, expression behavior, message splitting and timing, include descriptive 95% intervals, and mark 3-case NVIDIA runs as smoke tests rather than strong estimates.
- **Persistent recovery:** retry/blocked generation state and simulation messages survive app restarts in local SQLite, including migration from <=1.0.5 stores.

## Development train

1. `v1.1.0-dev.1` — provider failure resilience
2. `v1.1.0-dev.2` — context-conditioned live timing
3. `v1.1.0-dev.3` — stronger person-specific generation + anti-copy
4. `v1.1.0-rc.1` — integrated behavior/delivery clock separation, question-aware timing and evaluation uncertainty
5. `v1.1.0` — final validated release

## Privacy and interpretation

- Raw KakaoTalk/Instagram archives are not release content.
- API keys are not written to result JSON or release artifacts.
- Hosted generation still requires explicit private-context consent.
- REAL and SIMULATION memory remain separate.
- The base LLM remains a general language model conditioned by person-specific observable evidence; this release does not claim neural fine-tuning into a real person or reconstruction of hidden mental state.
- The core never sends messages to a real messaging platform.

## Windows release gate

The required artifact gate covers the full Python regression/provider-contract tests, integrated GUI imports, PyInstaller one-file build, execution of the packaged EXE smoke path, portable ZIP packaging, and GitHub Release creation.

Hosted NVIDIA NIM is also probed with bounded retries and reported as a separate external-health status. A current third-party outage does **not** invalidate or block an otherwise verified Windows artifact; in the live runtime, provider failures likewise preserve the already-scheduled simulated behavior and retry generation separately.
