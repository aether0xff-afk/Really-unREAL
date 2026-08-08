# Cutoff-safe RAG and generation context

Phase 2C begins only after the temporal layer has decided that an action should exist. The language layer must never be allowed to decide whether to talk.

## Two independent future-leakage barriers

Historical generation can leak the future through more than retrieval.

### 1. Retrieval cutoff

`backend.retrieval.CutoffExampleIndex` may be built from the full local history, but a replay query only returns examples whose real target action occurred **strictly before** the replay observation time.

```text
past examples          replay cutoff             future examples
<---------------------------|--------------------------->
          eligible          |          forbidden
```

Strict `<` is used rather than `<=` because KakaoTalk timestamps are usually minute-precision. Two messages carrying the same displayed minute do not establish a safe chronological order.

### 2. Persona cutoff

A language/style profile calculated from the entire export would also leak future behavior. `backend.persona.cutoff.build_cutoff_language_profile()` therefore uses only target messages strictly older than the replay cutoff.

The profile is source-weighted so Kakao remains primary and Instagram remains supplemental.

## Historical example unit

The retrieval index is built from Historical Replay cases. Internally each historical example contains its real response so it can be evaluated and summarized, but **raw response text is not passed to the language model by default**.

The generation packet exposes:

```text
historical visible context
response burst size
per-bubble lengths
question / laugh / cry counts
short ending fragments
source / recency / retrieval score
```

instead of handing the model a past sentence to reuse. Raw historical response text can only be enabled explicitly for a copy-risk ablation with `raw_response_examples` / `--raw-rag-responses`.

## Retrieval score

The dependency-free baseline ranker combines:

- Korean/ASCII token overlap;
- character-bigram cosine similarity;
- recency;
- source evidence weight;
- a small same-platform preference.

This is a **lexical similarity proxy**, not a learned semantic embedding. The compatibility field is still called `semantic_similarity` in the current result object, but documentation and experiments should not overclaim what it measures.

When the action role is trustworthy, retrieval is action-aware: REPLY generation retrieves historical REPLY examples and INITIATE/follow-up generation retrieves the corresponding bucket. Long-gap ambiguous cases disable this filter rather than trusting a weak sender-order proxy.

## Generation packet

`backend.generation_context.build_generation_context()` produces the information a language model may see:

```text
chosen coarse action from temporal policy
long-gap action ambiguity flag
visible recent conversation
cutoff-safe weighted language profile
cutoff-safe historical contexts + response shapes
```

The language profile now includes additional observable style statistics such as question/exclamation use, terminal-punctuation habits, multiline use, and frequent short endings. These are still surface behavior, not inferred feelings or personality traits.

The caller supplies `chosen_action`. The packet builder does not inspect the real held-out replay target burst.

## Copy control

The default production path uses:

```text
raw_response_examples = 0
```

The prompt explicitly tells the model to treat retrieved examples as situation/shape evidence and to prefer the current visible context. A future copy-rate evaluator can compare generated output with the hidden retrieval corpus, but raw nearest-neighbour responses do not need to be exposed merely to generate text.

## Evaluation

Generation quality is intentionally split into multiple observable metrics rather than one misleading score:

- burst-size absolute error;
- total character-length error;
- character-bigram F1;
- token F1;
- short-ending F1;
- laugh-expression presence match;
- cry-expression presence match;
- question presence match.

These are still not a complete semantic evaluator. In particular, a future embedding or human evaluation layer is needed for paraphrases whose wording differs while meaning stays similar.

## Source ablation

`backend.nvidia_replay --sources both` compares Kakao-only and fused Kakao+Instagram evidence on the **same Kakao chronological held-out cases**. Instagram may change persona/retrieval evidence but not the benchmark split itself.

That makes the current `0.55` Instagram DM weight a testable starting value rather than an assumed truth.
