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
    ambiguous_test_cases: int
    temporal_wait_misses: int
    temporal_early_predictions: int
    temporal_late_predictions: int
    mean_burst_size_absolute_error: float | None
    mean_total_char_length_absolute_error: float | None
    mean_char_bigram_f1: float | None
    mean_token_f1: float | None
    mean_ending_f1: float | None
    laugh_presence_match_rate: float | None
    cry_presence_match_rate: float | None
    question_presence_match_rate: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "source_mode": self.source_mode,
            "model": self.model,
            "selected_temporal_model": self.selected_temporal_model,
            "candidate_test_cases": self.candidate_test_cases,
            "requested_cases": self.requested_cases,
            "generated_cases": self.generated_cases,
            "ambiguous_test_cases": self.ambiguous_test_cases,
            "temporal_wait_misses": self.temporal_wait_misses,
            "temporal_early_predictions": self.temporal_early_predictions,
            "temporal_late_predictions": self.temporal_late_predictions,
            "mean_burst_size_absolute_error": self.mean_burst_size_absolute_error,
            "mean_total_char_length_absolute_error": self.mean_total_char_length_absolute_error,
            "mean_char_bigram_f1": self.mean_char_bigram_f1,
            "mean_token_f1": self.mean_token_f1,
            "mean_ending_f1": self.mean_ending_f1,
            "laugh_presence_match_rate": self.laugh_presence_match_rate,
            "cry_presence_match_rate": self.cry_presence_match_rate,
            "question_presence_match_rate": self.question_presence_match_rate,
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


def _candidate_action(case: ReplayCase) -> Action:
    """Infer the coarse REPLY vs INITIATE proxy from visible context."""

    if not case.context:
        raise ValueError("ReplayCase has no visible context")
    previous_sender = case.context[-1].sender_person_id
    return Action.INITIATE if previous_sender == case.person_id else Action.REPLY


def _fixed_kakao_split(
    evidence: PersonEvidence,
    *,
    self_person_id: str,
):
    """Use Kakao-only history to define one shared chronological benchmark.

    Source ablations must evaluate the same future cases. Instagram may enrich
    generation evidence, but it must not move the train/test boundary or change
    which Kakao events are scored.
    """

    kakao_evidence = _filter_evidence(evidence, "kakao")
    cases = build_replay_cases(kakao_evidence, self_person_id=self_person_id)
    if len(cases) < 3:
        raise ValueError("not enough Kakao replay cases for chronological evaluation")
    split = chronological_split(cases)
    if not split.train or not split.validation or not split.test:
        raise ValueError("chronological Kakao replay split is empty")
    return kakao_evidence, cases, split


def run_nvidia_replay(
    *,
    evidence: PersonEvidence,
    self_person_id: str,
    source_mode: str = "kakao",
    limit: int = 20,
    test_platform: str = "kakao",
    model_name: str = "nvidia/nemotron-3-ultra-550b-a55b",
    raw_response_examples: int = 0,
) -> ReplayGenerationSummary:
    if source_mode not in {"kakao", "fused"}:
        raise ValueError("source_mode must be 'kakao' or 'fused'")

    kakao_evidence, _, split = _fixed_kakao_split(
        evidence,
        self_person_id=self_person_id,
    )
    selection, baseline, hazard, _, _ = select_temporal_model(
        split.train,
        split.validation,
    )

    generation_evidence = _filter_evidence(evidence, source_mode)
    generation_cases = build_replay_cases(
        generation_evidence,
        self_person_id=self_person_id,
    )
    index = CutoffExampleIndex.from_replay_cases(generation_cases)
    model = NvidiaNIMLanguageModel(model=model_name)

    # The scored set is always the same Kakao-only chronological test set. This
    # makes Kakao vs fused a real source ablation instead of two different tests.
    test_cases = [case for case in split.test if case.platform == test_platform]
    requested = _evenly_limit(test_cases, limit)

    burst_errors: list[float] = []
    length_errors: list[float] = []
    bigram_scores: list[float] = []
    token_scores: list[float] = []
    ending_scores: list[float] = []
    laugh_matches: list[float] = []
    cry_matches: list[float] = []
    question_matches: list[float] = []
    temporal_early_predictions = 0
    temporal_late_predictions = 0

    for case in requested:
        if selection.selected_model == "hazard":
            predicted_delay = hazard.predict_median_delay_seconds(case)
        else:
            predicted_delay = baseline.predict_delay_seconds(case)

        if predicted_delay < case.delay_lower_seconds:
            temporal_early_predictions += 1
        elif predicted_delay > case.delay_upper_seconds:
            temporal_late_predictions += 1

        # Content quality is evaluated independently from timing quality. For a
        # long-gap event, the sender-order action is only a proxy and therefore
        # does not restrict retrieval to REPLY or INITIATE examples.
        chosen_action = _candidate_action(case)
        packet = build_generation_context(
            case,
            generation_evidence,
            index,
            chosen_action=chosen_action,
            raw_response_examples=raw_response_examples,
            action_specific_retrieval=not case.action_is_ambiguous,
        )
        generated = model.generate_burst(packet)
        metrics = evaluate_generated_burst(generated, case)
        burst_errors.append(float(metrics.burst_size_absolute_error))
        length_errors.append(float(metrics.total_char_length_absolute_error))
        bigram_scores.append(metrics.char_bigram_f1)
        token_scores.append(metrics.token_f1)
        ending_scores.append(metrics.ending_f1)
        laugh_matches.append(float(metrics.laugh_presence_match))
        cry_matches.append(float(metrics.cry_presence_match))
        question_matches.append(float(metrics.question_presence_match))

    generated_cases = len(bigram_scores)
    return ReplayGenerationSummary(
        source_mode=source_mode,
        model=model.model,
        selected_temporal_model=selection.selected_model,
        candidate_test_cases=len(test_cases),
        requested_cases=len(requested),
        generated_cases=generated_cases,
        ambiguous_test_cases=sum(case.action_is_ambiguous for case in requested),
        # Retained as a backwards-compatible name for late timing predictions.
        temporal_wait_misses=temporal_late_predictions,
        temporal_early_predictions=temporal_early_predictions,
        temporal_late_predictions=temporal_late_predictions,
        mean_burst_size_absolute_error=_mean(burst_errors),
        mean_total_char_length_absolute_error=_mean(length_errors),
        mean_char_bigram_f1=_mean(bigram_scores),
        mean_token_f1=_mean(token_scores),
        mean_ending_f1=_mean(ending_scores),
        laugh_presence_match_rate=_mean(laugh_matches),
        cry_presence_match_rate=_mean(cry_matches),
        question_presence_match_rate=_mean(question_matches),
    )


def _delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return round(right - left, 6)


def compare_source_summaries(
    kakao: ReplayGenerationSummary,
    fused: ReplayGenerationSummary,
) -> dict[str, object]:
    """Report fused-minus-Kakao deltas on the exact same held-out cases."""

    return {
        "mean_char_bigram_f1_delta": _delta(
            kakao.mean_char_bigram_f1,
            fused.mean_char_bigram_f1,
        ),
        "mean_token_f1_delta": _delta(kakao.mean_token_f1, fused.mean_token_f1),
        "mean_ending_f1_delta": _delta(kakao.mean_ending_f1, fused.mean_ending_f1),
        "mean_burst_size_absolute_error_delta": _delta(
            kakao.mean_burst_size_absolute_error,
            fused.mean_burst_size_absolute_error,
        ),
        "mean_total_char_length_absolute_error_delta": _delta(
            kakao.mean_total_char_length_absolute_error,
            fused.mean_total_char_length_absolute_error,
        ),
    }


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
        choices=("kakao", "fused", "both"),
        default="kakao",
        help=(
            "Kakao is the production default; fused adds supplemental Instagram. "
            "both runs a fair same-case ablation."
        ),
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--test-platform", default="kakao")
    parser.add_argument(
        "--model",
        default="nvidia/nemotron-3-ultra-550b-a55b",
    )
    parser.add_argument(
        "--raw-rag-responses",
        type=int,
        default=0,
        help="Explicit copy-risk ablation only; production default is 0",
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

    privacy = (
        "Only aggregate evaluation metrics are printed. Private prompts, retrieved "
        "examples, generated messages, and held-out real messages are not logged."
    )

    if args.sources == "both":
        kakao_summary = run_nvidia_replay(
            evidence=evidence,
            self_person_id=self_person_id,
            source_mode="kakao",
            limit=args.limit,
            test_platform=args.test_platform,
            model_name=args.model,
            raw_response_examples=args.raw_rag_responses,
        )
        fused_summary = run_nvidia_replay(
            evidence=evidence,
            self_person_id=self_person_id,
            source_mode="fused",
            limit=args.limit,
            test_platform=args.test_platform,
            model_name=args.model,
            raw_response_examples=args.raw_rag_responses,
        )
        output: dict[str, object] = {
            "kakao": kakao_summary.to_dict(),
            "fused": fused_summary.to_dict(),
            "fused_minus_kakao": compare_source_summaries(
                kakao_summary,
                fused_summary,
            ),
            "privacy": privacy,
        }
    else:
        summary = run_nvidia_replay(
            evidence=evidence,
            self_person_id=self_person_id,
            source_mode=args.sources,
            limit=args.limit,
            test_platform=args.test_platform,
            model_name=args.model,
            raw_response_examples=args.raw_rag_responses,
        )
        output = summary.to_dict()
        output["privacy"] = privacy

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
