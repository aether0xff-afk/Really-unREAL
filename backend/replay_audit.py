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
from backend.replay_hazard import (
    evaluate_hazard_model,
    select_temporal_model,
)
from backend.twin import TwinMode, resolve_twin_spec


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and audit leakage-safe Historical Replay events"
    )
    parser.add_argument("kakao_archive")
    parser.add_argument("instagram_archive")
    parser.add_argument("identity_map")
    parser.add_argument(
        "person_id",
        nargs="?",
        help="Target person ID for PERSON mode. Omit with --self-twin.",
    )
    parser.add_argument(
        "--self-twin",
        action="store_true",
        help="Audit the identity map's is_self=true person as the replay target",
    )
    parser.add_argument("--context-size", type=int, default=30)
    parser.add_argument("--burst-gap-seconds", type=float, default=120.0)
    parser.add_argument("--session-gap-hours", type=float, default=6.0)
    parser.add_argument(
        "--include-group",
        action="store_true",
        help="Include only trustworthy target-relative labels from group conversations",
    )
    parser.add_argument(
        "--baseline-quantile",
        type=float,
        default=0.5,
        help="Empirical timing quantile used by the simple baseline",
    )
    args = parser.parse_args()

    identities = IdentityMap.from_json(args.identity_map)
    try:
        spec = resolve_twin_spec(
            identities,
            mode=TwinMode.SELF if args.self_twin else TwinMode.PERSON,
            person_id=args.person_id,
        )
    except (KeyError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    kakao = load_kakao_archive(args.kakao_archive)
    instagram = load_instagram_export(args.instagram_archive)
    evidence = collect_person_evidence(
        spec.target_person_id,
        identities,
        kakao_conversations=kakao,
        instagram_threads=instagram.threads,
    )
    cases = build_replay_cases(
        evidence,
        self_person_id=spec.self_person_id,
        context_size=args.context_size,
        burst_gap_seconds=args.burst_gap_seconds,
        session_gap_hours=args.session_gap_hours,
        include_group=args.include_group,
    )
    snapshots = build_action_snapshots(cases)
    split = chronological_split(cases) if cases else None

    baseline_output: dict[str, object] | None = None
    temporal_selection_output: dict[str, object] | None = None

    if split is not None and split.train and split.test:
        baseline = EmpiricalTimingBaseline.fit(
            split.train,
            quantile=args.baseline_quantile,
        )
        test_snapshots = build_action_snapshots(split.test)
        baseline_test_metrics = evaluate_empirical_baseline(
            baseline,
            split.test,
            test_snapshots,
        )
        baseline_output = {
            "thresholds": baseline.thresholds_dict(),
            "test_metrics": baseline_test_metrics.to_dict(),
        }

        if split.validation:
            (
                selection,
                selected_baseline,
                hazard,
                baseline_validation_metrics,
                hazard_validation_metrics,
            ) = select_temporal_model(split.train, split.validation)

            # The selector uses the canonical median empirical baseline. Keep its
            # test comparison next to the hazard result so model selection stays
            # validation-only and the final test set is never used to choose.
            selected_baseline_test = evaluate_empirical_baseline(
                selected_baseline,
                split.test,
                test_snapshots,
            )
            hazard_test = evaluate_hazard_model(
                hazard,
                split.test,
                test_snapshots,
            )
            selected_test_metrics = (
                hazard_test.to_dict()
                if selection.selected_model == "hazard"
                else selected_baseline_test.to_dict()
            )
            temporal_selection_output = {
                "selection": selection.to_dict(),
                "hazard": {
                    "summary": hazard.summary_dict(),
                    "validation_metrics": hazard_validation_metrics.to_dict(),
                    "test_metrics": hazard_test.to_dict(),
                },
                "canonical_empirical": {
                    "validation_metrics": baseline_validation_metrics.to_dict(),
                    "test_metrics": selected_baseline_test.to_dict(),
                },
                "selected_test_metrics": selected_test_metrics,
                "selection_contract": (
                    "Only train+validation choose the model. Test is reported after "
                    "selection and never participates in the choice."
                ),
            }

    output: dict[str, object] = {
        "twin_mode": spec.mode.value,
        "target_person_id": spec.target_person_id,
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
        "temporal_model_selection": temporal_selection_output,
        "privacy": (
            "This audit reports counts/timing only. Hidden real message text is not "
            "printed by default."
        ),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
