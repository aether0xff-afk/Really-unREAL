from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass

from backend.fusion import PersonEvidence, collect_person_evidence
from backend.generation import evaluate_generated_burst
from backend.generation_context import build_generation_context
from backend.identity import IdentityMap
from backend.ingest.archive import load_kakao_archive
from backend.ingest.instagram import load_instagram_export
from backend.providers.nvidia import NvidiaNIMLanguageModel
from backend.replay import ReplayCase, build_replay_cases, chronological_split
from backend.replay_hazard import select_temporal_model
from backend.retrieval import CutoffExampleIndex
from backend.simulation.action_policy import Action


@dataclass(frozen=True, slots=True)
class ReplayGenerationSummary:
    source_mode: str
    model: str
    selected_temporal_model: str
    candidate_test_cases: int
    requested_cases: int
    generated_cases: int
    temporal_wait_misses: int
    mean_burst_size_absolute_error: float | None
    mean_total_char_length_absolute_error: float | None
    mean_char_bigram_f1: float | None
    laugh_presence_match_rate: float | None
    cry_presence_match_rate: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "source_mode": self.source_mode,
            "model": self.model,
            "selected_temporal_model": self.selected_temporal_model,
            "candidate_test_cases": self.candidate_test_cases,
            "requested_cases": self.requested_cases,
            "generated_cases": self.generated_cases,
            "temporal_wait_misses": self.temporal_wait_misses,
            "mean_burst_size_absolute_error": self.mean_burst_size_absolute_error,
            "mean_total_char_length_absolute_error": self.mean_total_char_length_absolute_error,
            "mean_char_bigram_f1": self.mean_char_bigram_f1,
            "laugh_presence_match_rate": self.laugh_presence_match_rate,
            "cry_presence_match_rate": self.cry_presence_match_rate,
        }


def _filter_evidence(evidence: PersonEvidence, source_mode: str) -> PersonEvidence:
    if source_mode == "fused":
        return evidence
    if source_mode == "kakao":
        return PersonEvidence(
            person_id=evidence.person_id,
            conversations=tuple(
                conversation
                for conversation in evidence.conversations
                if conversation.platform == "kakao"
            ),
        )
    raise ValueError(f"unknown source mode: {source_mode}")


def _evenly_limit(cases: list[ReplayCase], limit: int) -> list[ReplayCase]:
    if limit <= 0 or len(cases) <= limit:
        return cases
    if limit == 1:
        return [cases[-1]]
    indexes = sorted(
        {
            round(index * (len(cases) - 1) / (limit - 1))
            for index in range(limit)
        }
    )
    return [cases[index] for index in indexes]


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 6) if values else None


def run_nvidia_replay(
    *,
    evidence: PersonEvidence,
    self_person_id: str,
    source_mode: str = "kakao",
    limit: int = 20,
    test_platform: str = "kakao",
    model_name: str = "nvidia/nemotron-3-ultra-550b-a55b",
) -> ReplayGenerationSummary:
    evidence = _filter_evidence(evidence, source_mode)
    cases = build_replay_cases(evidence, self_person_id=self_person_id)
    if len(cases) < 3:
        raise ValueError("not enough replay cases for chronological evaluation")
    split = chronological_split(cases)
    if not split.train or not split.validation or not split.test:
        raise ValueError("chronological replay split is empty")

    selection, baseline, hazard, _, _ = select_temporal_model(
        split.train,
        split.validation,
    )
    index = CutoffExampleIndex.from_replay_cases(cases)
    model = NvidiaNIMLanguageModel(model=model_name)

    test_cases = [case for case in split.test if case.platform == test_platform]
    requested = _evenly_limit(test_cases, limit)

    burst_errors: list[float] = []
    length_errors: list[float] = []
    bigram_scores: list[float] = []
    laugh_matches: list[float] = []
    cry_matches: list[float] = []
    temporal_wait_misses = 0

    for case in requested:
        if selection.selected_model == "hazard":
            chosen_action = hazard.predict_action(
                case,
                elapsed_seconds=case.observed_delay_seconds,
            )
        else:
            chosen_action = baseline.predict_action(
                case,
                elapsed_seconds=case.observed_delay_seconds,
            )

        if chosen_action == Action.WAIT:
            temporal_wait_misses += 1
            continue

        packet = build_generation_context(
            case,
            evidence,
            index,
            chosen_action=chosen_action,
        )
        generated = model.generate_burst(packet)
        metrics = evaluate_generated_burst(generated, case)
        burst_errors.append(float(metrics.burst_size_absolute_error))
        length_errors.append(float(metrics.total_char_length_absolute_error))
        bigram_scores.append(metrics.char_bigram_f1)
        laugh_matches.append(float(metrics.laugh_presence_match))
        cry_matches.append(float(metrics.cry_presence_match))

    generated_cases = len(bigram_scores)
    return ReplayGenerationSummary(
        source_mode=source_mode,
        model=model.model,
        selected_temporal_model=selection.selected_model,
        candidate_test_cases=len(test_cases),
        requested_cases=len(requested),
        generated_cases=generated_cases,
        temporal_wait_misses=temporal_wait_misses,
        mean_burst_size_absolute_error=_mean(burst_errors),
        mean_total_char_length_absolute_error=_mean(length_errors),
        mean_char_bigram_f1=_mean(bigram_scores),
        laugh_presence_match_rate=_mean(laugh_matches),
        cry_presence_match_rate=_mean(cry_matches),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run privacy-preserving NVIDIA NIM generation on held-out replay cases"
    )
    parser.add_argument("kakao_archive")
    parser.add_argument("instagram_archive")
    parser.add_argument("identity_map")
    parser.add_argument("person_id")
    parser.add_argument(
        "--sources",
        choices=("kakao", "fused"),
        default="kakao",
        help="Kakao is the production default; fused adds supplemental Instagram evidence",
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--test-platform", default="kakao")
    parser.add_argument(
        "--model",
        default="nvidia/nemotron-3-ultra-550b-a55b",
    )
    args = parser.parse_args()

    identities = IdentityMap.from_json(args.identity_map)
    self_person_id = identities.self_person_id
    if self_person_id is None:
        raise SystemExit("identity map must contain exactly one is_self=true person")

    kakao = load_kakao_archive(args.kakao_archive)
    instagram = load_instagram_export(args.instagram_archive)
    evidence = collect_person_evidence(
        args.person_id,
        identities,
        kakao_conversations=kakao,
        instagram_threads=instagram.threads,
    )

    summary = run_nvidia_replay(
        evidence=evidence,
        self_person_id=self_person_id,
        source_mode=args.sources,
        limit=args.limit,
        test_platform=args.test_platform,
        model_name=args.model,
    )
    output = summary.to_dict()
    output["privacy"] = (
        "Only aggregate evaluation metrics are printed. Private prompts, retrieved "
        "examples, generated messages, and held-out real messages are not logged."
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
