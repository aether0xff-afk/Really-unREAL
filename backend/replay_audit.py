from __future__ import annotations

import argparse
import json

from backend.fusion import collect_person_evidence
from backend.identity import IdentityMap
from backend.ingest.archive import load_kakao_archive
from backend.ingest.instagram import load_instagram_export
from backend.replay import (
    audit_replay,
    build_action_snapshots,
    build_replay_cases,
    chronological_split,
)
from backend.replay_baseline import (
    EmpiricalTimingBaseline,
    evaluate_empirical_baseline,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and audit leakage-safe Historical Replay events"
    )
    parser.add_argument("kakao_archive")
    parser.add_argument("instagram_archive")
    parser.add_argument("identity_map")
    parser.add_argument("person_id")
    parser.add_argument("--context-size", type=int, default=30)
    parser.add_argument("--burst-gap-seconds", type=float, default=120.0)
    parser.add_argument("--session-gap-hours", type=float, default=6.0)
    parser.add_argument(
        "--include-group",
        action="store_true",
        help="Include only user-addressable labels from group conversations",
    )
    parser.add_argument(
        "--baseline-quantile",
        type=float,
        default=0.5,
        help="Empirical timing quantile used by the simple baseline",
    )
    args = parser.parse_args()

    identities = IdentityMap.from_json(args.identity_map)
    self_person_id = identities.self_person_id
    if self_person_id is None:
        raise SystemExit(
            "identity map must mark exactly one person with is_self=true before replay"
        )

    kakao = load_kakao_archive(args.kakao_archive)
    instagram = load_instagram_export(args.instagram_archive)
    evidence = collect_person_evidence(
        args.person_id,
        identities,
        kakao_conversations=kakao,
        instagram_threads=instagram.threads,
    )
    cases = build_replay_cases(
        evidence,
        self_person_id=self_person_id,
        context_size=args.context_size,
        burst_gap_seconds=args.burst_gap_seconds,
        session_gap_hours=args.session_gap_hours,
        include_group=args.include_group,
    )
    snapshots = build_action_snapshots(cases)
    split = chronological_split(cases) if cases else None

    baseline_output: dict[str, object] | None = None
    if split is not None and split.train and split.test:
        baseline = EmpiricalTimingBaseline.fit(
            split.train,
            quantile=args.baseline_quantile,
        )
        test_snapshots = build_action_snapshots(split.test)
        baseline_output = {
            "thresholds": baseline.thresholds_dict(),
            "test_metrics": evaluate_empirical_baseline(
                baseline,
                split.test,
                test_snapshots,
            ).to_dict(),
        }

    output: dict[str, object] = {
        "person_id": args.person_id,
        "include_group": args.include_group,
        "audit": audit_replay(cases, snapshots).to_dict(),
        "split": (
            {
                "train": len(split.train),
                "validation": len(split.validation),
                "test": len(split.test),
            }
            if split is not None
            else {"train": 0, "validation": 0, "test": 0}
        ),
        "empirical_timing_baseline": baseline_output,
        "privacy": (
            "This audit reports counts/timing only. Hidden real message text is not "
            "printed by default."
        ),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
