from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable

from backend.generation import BurstLanguageModel, evaluate_generated_burst
from backend.generation_context import build_generation_context
from backend.gui_support import (
    LOCAL_BASE_URL,
    NVIDIA_BASE_URL,
    NVIDIA_MODEL,
    _target_evidence,
)
from backend.ingest.archive import ConversationExport
from backend.privacy import require_private_context_route
from backend.providers.nvidia import NvidiaNIMLanguageModel
from backend.providers.openai_compatible import OpenAICompatibleLanguageModel
from backend.replay import build_replay_cases
from backend.replay_generation import (
    _candidate_action,
    _evenly_limit,
    filter_evidence,
    fixed_kakao_split,
)
from backend.replay_hazard import select_temporal_model
from backend.retrieval import CutoffExampleIndex


ProgressCallback = Callable[[int, int, int, str], None]
CancelCheck = Callable[[], bool]

NVIDIA_GUI_TIMEOUT_SECONDS = 45.0
NVIDIA_GUI_MAX_ATTEMPTS = 1
NVIDIA_GUI_FORMAT_ATTEMPTS = 1

_T_CRITICAL_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 6) if values else None


def _mean_ci95(values: list[float], *, bounded: bool = True) -> dict[str, float] | None:
    """Student-t interval for a mean; tiny samples no longer use z=1.96."""

    if len(values) < 2:
        return None
    mean = statistics.fmean(values)
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    df = len(values) - 1
    critical = _T_CRITICAL_975.get(df, 1.96)
    margin = critical * standard_error
    lower = mean - margin
    upper = mean + margin
    if bounded:
        lower = max(0.0, lower)
        upper = min(1.0, upper)
    return {"lower": round(lower, 6), "upper": round(upper, 6)}


def _wilson_ci95(values: list[float]) -> dict[str, float] | None:
    """Wilson score interval for binary match rates."""

    if not values:
        return None
    n = len(values)
    successes = sum(1 for value in values if value >= 0.5)
    p = successes / n
    z = 1.96
    denominator = 1.0 + (z * z) / n
    center = (p + (z * z) / (2 * n)) / denominator
    margin = (
        z
        * math.sqrt((p * (1 - p) / n) + (z * z) / (4 * n * n))
        / denominator
    )
    return {
        "lower": round(max(0.0, center - margin), 6),
        "upper": round(min(1.0, center + margin), 6),
    }


def _sample_note(generated_cases: int) -> str:
    if generated_cases < 10:
        return "표본이 작아 빠른 smoke 확인용입니다. 모델 비교에는 10~20개 이상을 권장합니다."
    if generated_cases < 20:
        return "기초 비교가 가능한 표본입니다. 더 안정적인 비교에는 20개 이상을 권장합니다."
    return "비교용 표본 수를 확보했습니다. 그래도 사람별 데이터 양과 테스트 구간을 함께 확인하세요."


def run_generation_replay_interactive(
    *,
    evidence,
    self_person_id: str,
    language_model: BurstLanguageModel,
    limit: int,
    progress: ProgressCallback | None = None,
    is_cancelled: CancelCheck | None = None,
) -> dict[str, object]:
    """Run replay generation while keeping timing independent of provider health."""

    started = time.monotonic()
    _, _, split = fixed_kakao_split(evidence, self_person_id=self_person_id)
    selection, baseline, hazard, _, _ = select_temporal_model(split.train, split.validation)

    generation_evidence = filter_evidence(evidence, "kakao")
    generation_cases = build_replay_cases(
        generation_evidence,
        self_person_id=self_person_id,
    )
    index = CutoffExampleIndex.from_replay_cases(generation_cases)
    test_cases = [case for case in split.test if case.platform == "kakao"]
    requested = _evenly_limit(test_cases, max(1, int(limit)))
    total = len(requested)

    burst_errors: list[float] = []
    length_errors: list[float] = []
    bigram_scores: list[float] = []
    token_scores: list[float] = []
    ending_scores: list[float] = []
    laugh_matches: list[float] = []
    cry_matches: list[float] = []
    question_matches: list[float] = []
    timing_matches: list[float] = []
    temporal_early_predictions = 0
    temporal_late_predictions = 0
    failed_cases = 0
    cancelled = False

    if progress:
        progress(0, total, 0, "준비 완료")

    for index_number, case in enumerate(requested, start=1):
        if is_cancelled and is_cancelled():
            cancelled = True
            break

        if progress:
            progress(
                index_number - 1,
                total,
                failed_cases,
                f"Case {index_number}/{total} · 모델 응답 대기 중",
            )

        predicted_delay = (
            hazard.predict_median_delay_seconds(case)
            if selection.selected_model == "hazard"
            else baseline.predict_delay_seconds(case)
        )
        timing_inside = True
        if predicted_delay < case.delay_lower_seconds:
            temporal_early_predictions += 1
            timing_inside = False
        elif predicted_delay > case.delay_upper_seconds:
            temporal_late_predictions += 1
            timing_inside = False
        # Timing is a behavior-model metric. Record it even if the provider later
        # fails to generate content for this case.
        timing_matches.append(float(timing_inside))

        chosen_action = _candidate_action(case)
        packet = build_generation_context(
            case,
            generation_evidence,
            index,
            chosen_action=chosen_action,
            raw_response_examples=0,
            action_specific_retrieval=not case.action_is_ambiguous,
        )

        try:
            generated = language_model.generate_burst(packet)
            metrics = evaluate_generated_burst(generated, case)
        except (RuntimeError, ValueError, TimeoutError):
            failed_cases += 1
            if progress:
                progress(
                    index_number,
                    total,
                    failed_cases,
                    f"Case {index_number}/{total} 실패 · 다음 케이스 진행",
                )
            continue

        burst_errors.append(float(metrics.burst_size_absolute_error))
        length_errors.append(float(metrics.total_char_length_absolute_error))
        bigram_scores.append(metrics.char_bigram_f1)
        token_scores.append(metrics.token_f1)
        ending_scores.append(metrics.ending_f1)
        laugh_matches.append(float(metrics.laugh_presence_match))
        cry_matches.append(float(metrics.cry_presence_match))
        question_matches.append(float(metrics.question_presence_match))

        if progress:
            progress(index_number, total, failed_cases, f"Case {index_number}/{total} 완료")

    if is_cancelled and is_cancelled():
        cancelled = True

    generated_cases = len(bigram_scores)
    return {
        "source_mode": "kakao",
        "model": str(getattr(language_model, "model", type(language_model).__name__)),
        "selected_temporal_model": selection.selected_model,
        "candidate_test_cases": len(test_cases),
        "requested_cases": total,
        "generated_cases": generated_cases,
        "failed_cases": failed_cases,
        "cancelled": cancelled,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "ambiguous_test_cases": sum(case.action_is_ambiguous for case in requested),
        "temporal_early_predictions": temporal_early_predictions,
        "temporal_late_predictions": temporal_late_predictions,
        "timing_evaluated_cases": len(timing_matches),
        "timing_inside_rate": _mean(timing_matches),
        "timing_inside_rate_ci95": _wilson_ci95(timing_matches),
        "mean_burst_size_absolute_error": _mean(burst_errors),
        "mean_total_char_length_absolute_error": _mean(length_errors),
        "mean_char_bigram_f1": _mean(bigram_scores),
        "mean_char_bigram_f1_ci95": _mean_ci95(bigram_scores),
        "mean_token_f1": _mean(token_scores),
        "mean_token_f1_ci95": _mean_ci95(token_scores),
        "mean_ending_f1": _mean(ending_scores),
        "mean_ending_f1_ci95": _mean_ci95(ending_scores),
        "laugh_presence_match_rate": _mean(laugh_matches),
        "laugh_presence_match_rate_ci95": _wilson_ci95(laugh_matches),
        "cry_presence_match_rate": _mean(cry_matches),
        "cry_presence_match_rate_ci95": _wilson_ci95(cry_matches),
        "question_presence_match_rate": _mean(question_matches),
        "question_presence_match_rate_ci95": _wilson_ci95(question_matches),
        "evaluation_sample_note": _sample_note(generated_cases),
        "retrieval_backend": "lexical",
    }


def run_quick_generation_interactive(
    conversations: list[ConversationExport],
    *,
    self_alias: str,
    target_alias: str,
    provider: str,
    model: str,
    base_url: str,
    api_key: str | None,
    allow_remote_private_context: bool,
    limit: int,
    progress: ProgressCallback | None = None,
    is_cancelled: CancelCheck | None = None,
) -> dict[str, object]:
    _, _, evidence = _target_evidence(conversations, self_alias, target_alias)
    require_private_context_route(
        base_url,
        allow_remote_private_context=allow_remote_private_context,
    )

    if provider == "nvidia":
        language_model: BurstLanguageModel = NvidiaNIMLanguageModel(
            api_key=api_key or None,
            model=model or NVIDIA_MODEL,
            base_url=base_url or NVIDIA_BASE_URL,
            timeout_seconds=NVIDIA_GUI_TIMEOUT_SECONDS,
            max_attempts=NVIDIA_GUI_MAX_ATTEMPTS,
            max_format_attempts=NVIDIA_GUI_FORMAT_ATTEMPTS,
        )
    elif provider == "local":
        if not model.strip():
            raise ValueError("로컬 모델 이름을 입력하세요. 예: 현재 LM Studio에 로드된 model id")
        language_model = OpenAICompatibleLanguageModel(
            model=model.strip(),
            base_url=base_url or LOCAL_BASE_URL,
            api_key=api_key or None,
            timeout_seconds=60.0,
            max_attempts=1,
            format_attempts=1,
        )
    else:
        raise ValueError(f"지원하지 않는 provider: {provider}")

    result = run_generation_replay_interactive(
        evidence=evidence,
        self_person_id="self",
        language_model=language_model,
        limit=limit,
        progress=progress,
        is_cancelled=is_cancelled,
    )
    result["provider"] = provider
    result["target"] = target_alias
    return result
