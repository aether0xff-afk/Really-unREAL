# Really-unREAL v1.0.0

First public release of the local-first conversation-behavior digital-twin simulation framework.

## Highlights

- KakaoTalk and Instagram export ingestion with platform-aware evidence weighting.
- Explicit cross-platform identity fusion without unsafe fuzzy auto-merges.
- Leakage-safe Historical Replay with chronological train/validation/test splits.
- Interval-aware message timing evaluation for coarse KakaoTalk timestamps.
- Context-conditioned hazard timing model with empirical fallback/model selection.
- Cutoff-safe persona snapshots and historical-example retrieval.
- Provider-agnostic burst generation contract and held-out generation evaluator.
- Hosted NVIDIA NIM adapter for Nemotron 3 Ultra with reasoning disabled for terse observable message generation.
- Synthetic NVIDIA NIM smoke test and secret-backed GitHub Actions validation.
- Privacy boundary: raw private chat archives are never required in the repository or release artifacts.

## Evaluation architecture

Really-unREAL separates observable behavior into distinct components:

1. Temporal policy decides WAIT / REPLY / INITIATE.
2. Cutoff-safe retrieval and persona construction expose only information available before the held-out event.
3. The language model generates message bursts only after a non-WAIT action is chosen.
4. Timing quality and language-generation quality are evaluated separately.

## Privacy

The repository contains framework code and synthetic tests only. Real KakaoTalk/Instagram archives, identity maps, private prompts, generated private messages, and held-out target messages are not included in this release.
