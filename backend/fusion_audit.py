from __future__ import annotations

import argparse
import json
from collections import Counter

from backend.fusion import canonical_target_messages, collect_person_evidence
from backend.identity import IdentityMap
from backend.ingest.archive import load_kakao_archive
from backend.ingest.instagram import load_instagram_export


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit source-aware evidence for one explicitly mapped person"
    )
    parser.add_argument("kakao_archive")
    parser.add_argument("instagram_archive")
    parser.add_argument("identity_map", help="Local identity.local.json path")
    parser.add_argument("person_id")
    args = parser.parse_args()

    identities = IdentityMap.from_json(args.identity_map)
    kakao = load_kakao_archive(args.kakao_archive)
    instagram = load_instagram_export(args.instagram_archive)
    evidence = collect_person_evidence(
        args.person_id,
        identities,
        kakao_conversations=kakao,
        instagram_threads=instagram.threads,
    )
    canonical = canonical_target_messages(evidence)

    platform_counts = Counter(
        message.metadata.get("platform", "unknown") for message in canonical
    )
    output = {
        "person_id": args.person_id,
        "conversation_count": len(evidence.conversations),
        "target_message_count": len(canonical),
        "target_messages_by_context": evidence.counts_by_context(),
        "target_messages_by_platform": dict(platform_counts),
        "first_target_message": canonical[0].timestamp.isoformat() if canonical else None,
        "last_target_message": canonical[-1].timestamp.isoformat() if canonical else None,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
