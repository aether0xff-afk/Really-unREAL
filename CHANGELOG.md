# Changelog

## 1.0.0 — 2026-08-08

First stable simulation-core release.

### Added

- PERSON and SELF digital-twin modes.
- Leakage-safe chronological Historical Replay.
- Interval-aware Kakao timing evaluation and ambiguity handling for long gaps.
- Relationship-aware empirical timing with validation-gated hazard fallback.
- Cutoff-safe persona, topic memory, and explicit event/date memory.
- Response-shape RAG with raw historical response text hidden by default.
- Optional dense+lexical retrieval through an embedding provider interface.
- Provider-agnostic replay generation core.
- NVIDIA NIM and generic OpenAI-compatible language-model adapters.
- Explicit privacy gate for remote private-context transmission.
- Closed-loop historical Shadow Simulation with bounded event matching.
- SQLite discrete-event runtime with restart recovery and SIMULATION/REAL separation.
- Same-case Kakao vs Kakao+Instagram ablation.
- Aggregate-only private replay/shadow reporting by default.

### Safety and privacy

- Private archives, identity maps, SQLite state, and generated artifacts remain
  ignored by Git.
- The core never automatically sends generated content to a real messaging
  service.
- Remote generation/embedding with private conversation context requires an
  explicit opt-in flag.

### Known limits

See `docs/V1_0.md`. Version 1.0 is a stable simulation framework, not a claim of
perfect identity reconstruction or a finished consumer messenger application.
