from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from backend.identity import build_safe_identity_map, suggest_identity_matches
from backend.ingest.archive import load_kakao_archive
from backend.ingest.instagram import load_instagram_export


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Suggest Kakao/Instagram identity matches without unsafe merging"
    )
    parser.add_argument("kakao_archive")
    parser.add_argument("instagram_archive")
    parser.add_argument("--minimum-score", type=float, default=0.55)
    parser.add_argument(
        "--write-safe-map",
        help="Optional path for an identity map containing exact matches only",
    )
    parser.add_argument("--self-kakao", help="Your Kakao display name")
    parser.add_argument("--self-instagram", help="Your Instagram display name")
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

    written_map = None
    if args.write_safe_map:
        if bool(args.self_kakao) != bool(args.self_instagram):
            raise SystemExit(
                "--self-kakao and --self-instagram must be provided together"
            )
        identity_map = build_safe_identity_map(
            candidates,
            self_kakao_alias=args.self_kakao,
            self_instagram_alias=args.self_instagram,
        )
        identity_map.to_json(args.write_safe_map)
        written_map = args.write_safe_map

    output = {
        "kakao_alias_count": len(kakao_aliases),
        "instagram_alias_count": len(instagram_aliases),
        "safe_exact_matches": [
            asdict(candidate) for candidate in candidates if candidate.safe_auto_match
        ],
        "review_candidates": [
            asdict(candidate) for candidate in candidates if not candidate.safe_auto_match
        ],
        "written_safe_identity_map": written_map,
        "note": (
            "Only safe_exact_matches are written automatically. Review candidates "
            "must be explicitly added to the local identity map before fusion."
        ),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
