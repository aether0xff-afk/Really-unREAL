# Identity resolution and source fusion

Really-unREAL may observe the same real person under different aliases across KakaoTalk and Instagram. Mixing those records incorrectly would corrupt both persona and timing models, so identity resolution is intentionally conservative.

## Rule 1: fuzzy similarity never silently merges people

`backend.identity.suggest_identity_matches()` ranks possible Kakao/Instagram alias pairs. Only identical normalized display names are marked `safe_auto_match`. Similar names remain review candidates.

Examples:

```text
Kakao: same-name alias     Instagram: same-name alias     -> safe exact match
Kakao: full display name   Instagram: shortened name      -> review candidate only
Kakao: one person          Instagram: ambiguous nickname   -> review candidate only
```

A reviewed mapping is stored in a local `identity.local.json`. This file is gitignored because it contains real-person identifiers. Explicit user-confirmed aliases may be merged there even when their strings are not similar.

## Stable person IDs

After review, every person gets an internal ID independent of platform names:

```json
{
  "person_id": "person-001",
  "aliases": {
    "kakao": ["..."],
    "instagram": ["..."]
  }
}
```

Messages retain their original sender name, but fused evidence also carries `sender_person_id`.

## Kakao-primary source policy

KakaoTalk is the primary source for reconstructing persona and temporal behavior. Instagram is supplemental evidence: it fills gaps, provides extra direct-message examples, and contributes cross-platform/self-presentation context, but it should not override a stable pattern repeatedly observed in KakaoTalk.

The current default relevance weights are:

```text
kakao_direct       weight 1.00   primary behavioral evidence
instagram_direct   weight 0.55   supplemental direct evidence
kakao_group        weight 0.40   supporting style/context
instagram_group    weight 0.20   weak supporting context
```

These are starting relevance weights, not probabilities and not relationship scores. They must be validated or tuned by Historical Replay. If a person has little or no Kakao data, Instagram can still provide useful evidence, but the model must retain the fact that the evidence came from a supplemental source.

Posts, stories, comments, likes, saves, and follows are kept outside the core conversational-behavior weight hierarchy. They may provide context or weak interest signals, but must not be converted into claims about hidden feelings or relationship status.

## Source-aware evidence

`backend.fusion.collect_person_evidence()` does not flatten all observations into one bag. Every utterance keeps its platform, conversation ID, direct/group context, and evidence weight.

The full surrounding conversation is kept for retrieval, but only messages explicitly resolved to the target person become persona evidence.

## Why keep the full conversation?

A target message such as `ㅋㅋ` is not useful without context. Historical replay and RAG need the preceding user message, surrounding topic, time gap, and conversation source. Therefore fusion stores `EvidenceConversation` objects containing all messages while separately identifying the target's own utterances.

## Local workflow

Generate candidate matches:

```bash
python -m backend.identity_audit \
  ./data/raw/kakao_bundle.zip \
  ./data/raw/instagram_export.zip
```

Copy `examples/identity.example.json` to `identity.local.json`, then approve aliases manually. Real names and account-to-person mappings remain local and are never committed.

Audit evidence for one mapped person:

```bash
python -m backend.fusion_audit \
  ./data/raw/kakao_bundle.zip \
  ./data/raw/instagram_export.zip \
  ./identity.local.json \
  person-001
```

## Next step

Phase 2 will build historical replay datasets from these fused, source-aware conversations. The evaluation target is not exact wording alone. It includes:

- whether the person replies or waits;
- reply-delay distribution;
- initiation / re-initiation behavior;
- message splitting and burst structure;
- semantic/style similarity of the eventual response.
