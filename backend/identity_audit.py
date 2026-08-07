from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from backend.identity import suggest_identity_matches
from backend.ingest.archive import load_kakao_archive
from backend.ingest.instagram import load_instagram_export


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Suggest Kakao/Instagram identity matches without merging them"
    )
    parser.add_argument("kakao_archive")
    parser.add_argument("instagram_archive")
    parser.add_argument("--minimum-score", type=float, default=0.55)
    args = parser.parse_args()

    kakao = load_kakao_archive(args.kakao_archive)
    instagram = load_instagram_export(args.instagram_archive)

    kakao_aliases = sorted(
        {
            participant
            for conversation in kakao
            for participant in conversation.participants
        }
    )
    instagram_aliases = sorted(
        {
            participant
            for thread in instagram.threads
            for participant in thread.participants
        }
    )
    candidates = suggest_identity_matches(
        kakao_aliases,
        instagram_aliases,
        minimum_score=args.minimum_score,
    )

    output = {
        "kakao_alias_count": len(kakao_aliases),
        "instagram_alias_count": len(instagram_aliases),
        "safe_exact_matches": [
            asdict(candidate) for candidate in candidates if candidate.safe_auto_match
        ],
        "review_candidates": [
            asdict(candidate) for candidate in candidates if not candidate.safe_auto_match
        ],
        "note": (
            "Only safe_exact_matches may be auto-applied. Review candidates must "
            "be approved in a local identity map before evidence fusion."
        ),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
