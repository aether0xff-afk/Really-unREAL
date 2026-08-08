from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable

from backend.fusion import collect_person_evidence
from backend.identity import IdentityMap, PersonEntity, normalize_alias
from backend.ingest.archive import ConversationExport, load_kakao_archive
from backend.models import MessageType
from backend.privacy import require_private_context_route
from backend.providers.nvidia import NvidiaNIMLanguageModel
from backend.providers.openai_compatible import OpenAICompatibleLanguageModel
from backend.replay import (
    audit_replay,
    build_action_snapshots,
    build_replay_cases,
    chronological_split,
)
from backend.replay_baseline import EmpiricalTimingBaseline, evaluate_empirical_baseline
from backend.replay_generation import run_generation_replay
from backend.replay_hazard import evaluate_hazard_model, select_temporal_model


LOCAL_BASE_URL = "http://127.0.0.1:1234/v1"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"


def _normalize_archive_paths(
    paths: str | Path | Iterable[str | Path],
) -> tuple[Path, ...]:
    if isinstance(paths, (str, Path)):
        raw_paths = (paths,)
    else:
        raw_paths = tuple(paths)
    if not raw_paths:
        raise ValueError("카카오톡 ZIP을 하나 이상 선택하세요.")

    unique: list[Path] = []
    seen: set[str] = set()
    for raw in raw_paths:
        path = Path(raw)
        key = str(path.absolute())
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return tuple(unique)


def _conversation_fingerprint(conversation: ConversationExport) -> str:
    """Fingerprint message content so the same export selected twice is ignored."""

    digest = hashlib.sha256()
    for message in conversation.messages:
        payload = (
            message.timestamp.isoformat(),
            message.sender,
            message.text,
            message.message_type.value,
        )
        digest.update(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def load_quick_kakao(
    paths: str | Path | Iterable[str | Path],
) -> list[ConversationExport]:
    """Load one or many Kakao ZIPs for the desktop quick path.

    Every selected ZIP can itself be either a single-chat Kakao export or an
    outer bundle containing several per-chat ZIPs. Exact duplicate
    conversations are removed after loading, which prevents accidentally
    selecting the same archive twice from double-counting evidence.
    """

    conversations: list[ConversationExport] = []
    for path in _normalize_archive_paths(paths):
        try:
            conversations.extend(load_kakao_archive(path))
        except Exception as exc:
            raise ValueError(f"{path.name}을(를) 불러오지 못했습니다: {exc}") from exc

    deduplicated: list[ConversationExport] = []
    fingerprints: set[str] = set()
    for conversation in conversations:
        fingerprint = _conversation_fingerprint(conversation)
        if fingerprint in fingerprints:
            continue
        fingerprints.add(fingerprint)
        deduplicated.append(conversation)

    if not deduplicated:
        raise ValueError(
            "카카오톡 대화를 찾지 못했습니다. 선택한 ZIP 안에 Talk_*.txt가 있는지 확인하세요."
        )

    deduplicated.sort(
        key=lambda conversation: (
            conversation.messages[0].timestamp if conversation.messages else datetime.max,
            conversation.chat_name,
            conversation.source_archive,
        )
    )
    return deduplicated


def _visible_aliases(conversation: ConversationExport) -> set[str]:
    return {
        message.sender
        for message in conversation.messages
        if message.message_type != MessageType.SYSTEM and normalize_alias(message.sender)
    }


def rank_self_aliases(conversations: Iterable[ConversationExport]) -> list[str]:
    """Rank likely self aliases by cross-conversation presence, then message count.

    In a bundle of many direct Kakao exports the user's own display name usually
    appears across many different chats. This is a suggestion only; the GUI asks
    the user to confirm the choice before any replay is built.
    """

    conversation_counts: Counter[str] = Counter()
    message_counts: Counter[str] = Counter()
    for conversation in conversations:
        aliases = _visible_aliases(conversation)
        conversation_counts.update(aliases)
        for message in conversation.messages:
            if message.message_type != MessageType.SYSTEM and message.sender in aliases:
                message_counts[message.sender] += 1
    return sorted(
        conversation_counts,
        key=lambda alias: (-conversation_counts[alias], -message_counts[alias], alias),
    )


def direct_targets_for_self(
    conversations: Iterable[ConversationExport],
    self_alias: str,
) -> list[str]:
    counts: Counter[str] = Counter()
    for conversation in conversations:
        aliases = _visible_aliases(conversation)
        if len(aliases) != 2 or self_alias not in aliases:
            continue
        target = next(iter(aliases - {self_alias}))
        counts[target] += sum(
            1
            for message in conversation.messages
            if message.message_type != MessageType.SYSTEM and message.sender == target
        )
    return [alias for alias, _ in counts.most_common()]


def _person_id(alias: str) -> str:
    digest = hashlib.sha256(normalize_alias(alias).encode("utf-8")).hexdigest()[:10]
    return f"person-{digest}"


def build_quick_identity_map(
    conversations: Iterable[ConversationExport],
    self_alias: str,
) -> IdentityMap:
    aliases = sorted({alias for conversation in conversations for alias in _visible_aliases(conversation)})
    if self_alias not in aliases:
        raise ValueError("선택한 내 이름이 카카오톡 기록에 없습니다.")
    people = [PersonEntity(person_id="self", aliases={"kakao": (self_alias,)}, is_self=True)]
    people.extend(
        PersonEntity(person_id=_person_id(alias), aliases={"kakao": (alias,)})
        for alias in aliases
        if alias != self_alias
    )
    return IdentityMap(people)


def _target_evidence(
    conversations: list[ConversationExport],
    self_alias: str,
    target_alias: str,
):
    identities = build_quick_identity_map(conversations, self_alias)
    target_id = identities.resolve("kakao", target_alias)
    if target_id is None:
        raise ValueError("선택한 상대를 identity map에서 찾지 못했습니다.")
    evidence = collect_person_evidence(
        target_id,
        identities,
        kakao_conversations=conversations,
    )
    if not evidence.conversations:
        raise ValueError("선택한 상대와의 분석 가능한 대화를 찾지 못했습니다.")
    return identities, target_id, evidence


def run_quick_audit(
    conversations: list[ConversationExport],
    *,
    self_alias: str,
    target_alias: str,
) -> dict[str, object]:
    _, target_id, evidence = _target_evidence(conversations, self_alias, target_alias)
    cases = build_replay_cases(evidence, self_person_id="self")
    snapshots = build_action_snapshots(cases)
    output: dict[str, object] = {
        "mode": "PERSON",
        "target": target_alias,
        "target_person_id": target_id,
        "source": "kakao",
        "evidence_conversations": len(evidence.conversations),
        "target_messages": len(evidence.target_messages()),
        "replay_cases": len(cases),
        "audit": audit_replay(cases, snapshots).to_dict(),
    }
    if len(cases) < 3:
        output["warning"] = "리플레이 케이스가 너무 적어 train/validation/test 평가를 생략했습니다."
        return output

    split = chronological_split(cases)
    output["split"] = {
        "train": len(split.train),
        "validation": len(split.validation),
        "test": len(split.test),
    }
    if not split.train or not split.test:
        return output

    baseline = EmpiricalTimingBaseline.fit(split.train)
    test_snapshots = build_action_snapshots(split.test)
    baseline_test = evaluate_empirical_baseline(baseline, split.test, test_snapshots)
    output["empirical_timing"] = baseline_test.to_dict()

    if split.validation:
        selection, selected_baseline, hazard, baseline_val, hazard_val = select_temporal_model(
            split.train,
            split.validation,
        )
        selected_baseline_test = evaluate_empirical_baseline(
            selected_baseline,
            split.test,
            test_snapshots,
        )
        hazard_test = evaluate_hazard_model(hazard, split.test, test_snapshots)
        output["temporal_selection"] = {
            "selected_model": selection.selected_model,
            "validation": {
                "empirical": baseline_val.to_dict(),
                "hazard": hazard_val.to_dict(),
            },
            "test": {
                "empirical": selected_baseline_test.to_dict(),
                "hazard": hazard_test.to_dict(),
            },
        }
    return output


def run_quick_generation(
    conversations: list[ConversationExport],
    *,
    self_alias: str,
    target_alias: str,
    provider: str,
    model: str,
    base_url: str,
    api_key: str | None = None,
    allow_remote_private_context: bool = False,
    limit: int = 10,
) -> dict[str, object]:
    _, _, evidence = _target_evidence(conversations, self_alias, target_alias)
    require_private_context_route(
        base_url,
        allow_remote_private_context=allow_remote_private_context,
    )
    if provider == "nvidia":
        language_model = NvidiaNIMLanguageModel(
            api_key=api_key or None,
            model=model or NVIDIA_MODEL,
            base_url=base_url or NVIDIA_BASE_URL,
        )
    elif provider == "local":
        if not model.strip():
            raise ValueError("로컬 모델 이름을 입력하세요. 예: 현재 LM Studio에 로드된 model id")
        language_model = OpenAICompatibleLanguageModel(
            model=model.strip(),
            base_url=base_url or LOCAL_BASE_URL,
            api_key=api_key or None,
        )
    else:
        raise ValueError(f"지원하지 않는 provider: {provider}")

    summary = run_generation_replay(
        evidence=evidence,
        self_person_id="self",
        source_mode="kakao",
        language_model=language_model,
        limit=max(1, int(limit)),
        raw_response_examples=0,
    )
    result = summary.to_dict()
    result["provider"] = provider
    result["target"] = target_alias
    return result
