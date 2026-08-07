# Identity resolution and source fusion

Really-unREAL may observe the same real person under different aliases across KakaoTalk and Instagram. Mixing those records incorrectly would corrupt both persona and timing models, so identity resolution is intentionally conservative.

## Rule 1: fuzzy similarity never silently merges people

`backend.identity.suggest_identity_matches()` ranks possible Kakao/Instagram alias pairs. Only identical normalized display names are marked `safe_auto_match`. Similar names remain review candidates.

Examples:

```text
Kakao: 이은세   Instagram: 이은세   -> safe exact match
Kakao: 임명민   Instagram: 명민     -> review candidate only
Kakao: 감동현   Instagram: 동현     -> review candidate only
```

A reviewed mapping is stored in a local `identity.local.json`. This file is gitignored because it contains real-person identifiers.

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

## Source-aware evidence

`backend.fusion.collect_person_evidence()` does not flatten all observations into one bag. It preserves the context in which each utterance was observed:

```text
kakao_direct       weight 1.00
instagram_direct   weight 1.00
instagram_group    weight 0.45
kakao_group        weight 0.35
```

These are default relevance weights, not probabilities and not relationship scores. They express how directly a sample reflects the target person's behavior in a one-to-one interaction.

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

Copy `examples/identity.example.json` to `identity.local.json`, then approve aliases manually.

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
