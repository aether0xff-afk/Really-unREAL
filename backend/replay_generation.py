from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass

from backend.fusion import PersonEvidence
from backend.generation import BurstLanguageModel, evaluate_generated_burst
from backend.generation_context import build_generation_context
from backend.replay import ReplayCase, build_replay_cases, chronological_split
from backend.replay_hazard import select_temporal_model
from backend.retrieval import CutoffExampleIndex, EmbeddingProvider
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
    retrieval_backend: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def filter_evidence(evidence: PersonEvidence, source_mode: str) -> PersonEvidence:
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
    raise ValueError("source_mode must be 'kakao' or 'fused'")


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
    if not case.context:
        raise ValueError("ReplayCase has no visible context")
    return (
        Action.INITIATE
        if case.context[-1].sender_person_id == case.person_id
        else Action.REPLY
    )


def fixed_kakao_split(
    evidence: PersonEvidence,
    *,
    self_person_id: str,
):
    kakao_evidence = filter_evidence(evidence, "kakao")
    cases = build_replay_cases(kakao_evidence, self_person_id=self_person_id)
    if len(cases) < 3:
        raise ValueError("not enough Kakao replay cases for chronological evaluation")
    split = chronological_split(cases)
    if not split.train or not split.validation or not split.test:
        raise ValueError("chronological Kakao replay split is empty")
    return kakao_evidence, cases, split


def run_generation_replay(
    *,
    evidence: PersonEvidence,
    self_person_id: str,
    language_model: BurstLanguageModel,
    source_mode: str = "kakao",
    limit: int = 20,
    test_platform: str = "kakao",
    raw_response_examples: int = 0,
    embedding_provider: EmbeddingProvider | None = None,
    embedding_weight: float = 0.70,
) -> ReplayGenerationSummary:
    _, _, split = fixed_kakao_split(evidence, self_person_id=self_person_id)
    selection, baseline, hazard, _, _ = select_temporal_model(
        split.train,
        split.validation,
    )

    generation_evidence = filter_evidence(evidence, source_mode)
    generation_cases = build_replay_cases(
        generation_evidence,
        self_person_id=self_person_id,
    )
    index = CutoffExampleIndex.from_replay_cases(
        generation_cases,
        embedding_provider=embedding_provider,
        embedding_weight=embedding_weight,
    )

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
        predicted_delay = (
            hazard.predict_median_delay_seconds(case)
            if selection.selected_model == "hazard"
            else baseline.predict_delay_seconds(case)
        )
        if predicted_delay < case.delay_lower_seconds:
            temporal_early_predictions += 1
        elif predicted_delay > case.delay_upper_seconds:
            temporal_late_predictions += 1

        chosen_action = _candidate_action(case)
        packet = build_generation_context(
            case,
            generation_evidence,
            index,
            chosen_action=chosen_action,
            raw_response_examples=raw_response_examples,
            action_specific_retrieval=not case.action_is_ambiguous,
        )
        generated = language_model.generate_burst(packet)
        metrics = evaluate_generated_burst(generated, case)
        burst_errors.append(float(metrics.burst_size_absolute_error))
        length_errors.append(float(metrics.total_char_length_absolute_error))
        bigram_scores.append(metrics.char_bigram_f1)
        token_scores.append(metrics.token_f1)
        ending_scores.append(metrics.ending_f1)
        laugh_matches.append(float(metrics.laugh_presence_match))
        cry_matches.append(float(metrics.cry_presence_match))
        question_matches.append(float(metrics.question_presence_match))

    return ReplayGenerationSummary(
        source_mode=source_mode,
        model=str(getattr(language_model, "model", type(language_model).__name__)),
        selected_temporal_model=selection.selected_model,
        candidate_test_cases=len(test_cases),
        requested_cases=len(requested),
        generated_cases=len(bigram_scores),
        ambiguous_test_cases=sum(case.action_is_ambiguous for case in requested),
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
        retrieval_backend=("dense+lexical" if embedding_provider else "lexical"),
    )


def _delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return round(right - left, 6)


def compare_source_summaries(
    kakao: ReplayGenerationSummary,
    fused: ReplayGenerationSummary,
) -> dict[str, float | None]:
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
