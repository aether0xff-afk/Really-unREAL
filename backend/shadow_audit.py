from __future__ import annotations

import argparse
import json
import os

from backend.fusion import collect_person_evidence
from backend.identity import IdentityMap
from backend.ingest.archive import load_kakao_archive
from backend.ingest.instagram import load_instagram_export
from backend.privacy import require_private_context_route
from backend.providers.nvidia import NvidiaNIMLanguageModel
from backend.providers.openai_compatible import OpenAICompatibleLanguageModel
from backend.replay import build_replay_cases
from backend.replay_baseline import EmpiricalTimingBaseline
from backend.simulation.shadow import run_shadow_simulation
from backend.twin import TwinMode, resolve_twin_spec


NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
LOCAL_BASE_URL = "http://127.0.0.1:1234/v1"
NVIDIA_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a closed-loop historical shadow interval. Real target messages in "
            "the holdout are hidden and simulated messages feed the next turn."
        )
    )
    parser.add_argument("kakao_archive")
    parser.add_argument("instagram_archive")
    parser.add_argument("identity_map")
    parser.add_argument("conversation_id")
    parser.add_argument("person_id", nargs="?")
    parser.add_argument("--self-twin", action="store_true")
    parser.add_argument("--provider", choices=("local", "nvidia"), default="local")
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env", default="LOCAL_LLM_API_KEY")
    parser.add_argument("--allow-remote-private-context", action="store_true")
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--match-window-seconds", type=float, default=21600.0)
    args = parser.parse_args()

    if not 0.05 <= args.holdout_fraction <= 0.50:
        raise SystemExit("--holdout-fraction must be between 0.05 and 0.50")

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
    cases = build_replay_cases(evidence, self_person_id=spec.self_person_id)
    conversation_cases = sorted(
        [case for case in cases if case.conversation_id == args.conversation_id],
        key=lambda case: (case.action_at, case.case_id),
    )
    if len(conversation_cases) < 5:
        raise SystemExit("shadow audit requires at least 5 replay events in the conversation")

    holdout_count = max(1, int(round(len(conversation_cases) * args.holdout_fraction)))
    split_index = len(conversation_cases) - holdout_count
    if split_index < 1:
        raise SystemExit("shadow holdout leaves no earlier history")
    start_at = conversation_cases[split_index].observation_end
    end_at = conversation_cases[-1].action_at
    training_cases = [case for case in cases if case.action_at < start_at]
    if not training_cases:
        raise SystemExit("no replay history exists before shadow start")
    timing = EmpiricalTimingBaseline.fit(training_cases)

    if args.provider == "nvidia":
        base_url = args.base_url or NVIDIA_BASE_URL
        model_name = args.model or NVIDIA_MODEL
        require_private_context_route(
            base_url,
            allow_remote_private_context=args.allow_remote_private_context,
        )
        model = NvidiaNIMLanguageModel(model=model_name, base_url=base_url)
    else:
        base_url = args.base_url or LOCAL_BASE_URL
        model_name = args.model
        if not model_name:
            raise SystemExit("--model is required for --provider local")
        require_private_context_route(
            base_url,
            allow_remote_private_context=args.allow_remote_private_context,
        )
        model = OpenAICompatibleLanguageModel(
            model=model_name,
            base_url=base_url,
            api_key=os.environ.get(args.api_key_env),
        )

    report, _ = run_shadow_simulation(
        evidence=evidence,
        replay_cases=cases,
        timing=timing,
        language_model=model,
        conversation_id=args.conversation_id,
        start_at=start_at,
        end_at=end_at,
        match_window_seconds=args.match_window_seconds,
    )
    output = {
        "twin_mode": spec.mode.value,
        "provider": args.provider,
        "model": model_name,
        "shadow": report.to_dict(),
        "privacy": (
            "The report contains aggregate metrics only. Held-out real target text and "
            "simulated messages are not printed. Remote model use requires explicit consent."
        ),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
