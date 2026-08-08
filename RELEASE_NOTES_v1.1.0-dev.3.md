# Really-unREAL v1.1.0-dev.3

Third 1.1 development build: stronger person-specific generation without turning retrieval into copying.

## What changed

- Added a richer observable style fingerprint: length distribution, spacing, jamo runs, ellipses, repeated punctuation, common openings and punctuation shapes.
- Added a relationship/action-aware burst profile for typical bubble count and total response length.
- Live generation may use at most two cutoff-safe older REAL replies as private style exemplars.
- Historical Replay/model-test paths keep raw response exemplars disabled by default.
- Added an anti-copy generation guard: long near-verbatim exemplar reuse triggers a second independently worded generation attempt.
- Very short recurring expressions such as `ㅇㅇ` or `ㄴㄴ` are not treated as memorization.
- All style/exemplar retrieval remains strictly older than the current cutoff; future/held-out responses are never exposed.
- Private exemplar text is not added to public result JSON, logs, or release artifacts.

## Why

Earlier versions mainly gave the base model aggregate style statistics. That could make responses generically casual without strongly reproducing the selected person's observable writing behavior. dev.3 provides stronger person-specific evidence while explicitly defending against retrieval-copy shortcuts.
