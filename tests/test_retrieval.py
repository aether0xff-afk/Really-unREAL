from datetime import datetime, timedelta

from backend.fusion import EvidenceContext, EvidenceMessage
from backend.models import ChatMessage
from backend.replay import ReplayCase
from backend.retrieval import CutoffExampleIndex
from backend.simulation.action_policy import Action


BASE = datetime(2026, 8, 7, 20, 0)


def _evidence(
    at: datetime,
    sender_id: str,
    text: str,
    *,
    platform: str = "kakao",
    conversation_id: str = "c",
    weight: float = 1.0,
) -> EvidenceMessage:
    context = (
        EvidenceContext.KAKAO_DIRECT
        if platform == "kakao"
        else EvidenceContext.INSTAGRAM_DIRECT
    )
    return EvidenceMessage(
        message=ChatMessage(at, sender_id, text),
        platform=platform,
        conversation_id=conversation_id,
        context=context,
        sender_person_id=sender_id,
        evidence_weight=weight,
    )


def _case(
    case_id: str,
    *,
    action_at: datetime,
    context_text: str,
    target_text: str,
    platform: str = "kakao",
    weight: float = 1.0,
    conversation_id: str = "c",
) -> ReplayCase:
    observation_end = action_at - timedelta(minutes=5)
    context_kind = (
        EvidenceContext.KAKAO_DIRECT
        if platform == "kakao"
        else EvidenceContext.INSTAGRAM_DIRECT
    )
    visible = _evidence(
        observation_end,
        "self",
        context_text,
        platform=platform,
        conversation_id=conversation_id,
        weight=weight,
    )
    target = _evidence(
        action_at,
        "target",
        target_text,
        platform=platform,
        conversation_id=conversation_id,
        weight=weight,
    )
    return ReplayCase(
        case_id=case_id,
        person_id="target",
        platform=platform,
        conversation_id=conversation_id,
        evidence_context=context_kind,
        evidence_weight=weight,
        action=Action.REPLY,
        observation_end=observation_end,
        action_at=action_at,
        observed_delay_seconds=300.0,
        delay_lower_seconds=240.0,
        delay_upper_seconds=360.0,
        context=(visible,),
        target_burst=(target,),
        burst_size=1,
        session_restart=False,
    )


def test_future_examples_are_never_retrieved() -> None:
    past = _case(
        "past",
        action_at=BASE - timedelta(days=2),
        context_text="시험 어땠어",
        target_text="망함ㅋㅋ",
    )
    future = _case(
        "future",
        action_at=BASE + timedelta(days=1),
        context_text="시험 어땠어",
        target_text="잘봄",
    )
    query = _case(
        "query",
        action_at=BASE + timedelta(minutes=5),
        context_text="시험 어땠어",
        target_text="hidden",
        conversation_id="query-thread",
    )
    index = CutoffExampleIndex.from_replay_cases([past, future, query])

    results = index.search(query, cutoff=BASE, k=10)

    assert [result.example.case_id for result in results] == ["past"]


def test_same_timestamp_as_cutoff_is_excluded_conservatively() -> None:
    same_minute = _case(
        "same-minute",
        action_at=BASE,
        context_text="뭐해",
        target_text="집",
    )
    query = _case(
        "query",
        action_at=BASE + timedelta(minutes=5),
        context_text="뭐해",
        target_text="hidden",
        conversation_id="query-thread",
    )
    index = CutoffExampleIndex.from_replay_cases([same_minute, query])

    assert index.search(query, cutoff=BASE, k=10) == []


def test_relevant_past_context_outranks_unrelated_context() -> None:
    relevant = _case(
        "relevant",
        action_at=BASE - timedelta(days=10),
        context_text="수학 시험 망했어",
        target_text="ㅋㅋ 나도",
        conversation_id="old-a",
    )
    unrelated = _case(
        "unrelated",
        action_at=BASE - timedelta(days=1),
        context_text="오늘 저녁 뭐 먹지",
        target_text="치킨",
        conversation_id="old-b",
    )
    query = _case(
        "query",
        action_at=BASE + timedelta(minutes=5),
        context_text="수학 시험 진짜 망했다",
        target_text="hidden",
        conversation_id="query-thread",
    )
    index = CutoffExampleIndex.from_replay_cases([relevant, unrelated, query])

    results = index.search(query, cutoff=BASE, k=2)

    assert results[0].example.case_id == "relevant"
    assert results[0].semantic_similarity > results[1].semantic_similarity


def test_dense_embedding_can_resolve_low_lexical_overlap() -> None:
    class FakeEmbeddingProvider:
        def embed(self, texts):
            vectors = []
            for text in texts:
                if "시험" in text or "공부" in text:
                    vectors.append((1.0, 0.0))
                elif "저녁" in text or "치킨" in text:
                    vectors.append((0.0, 1.0))
                else:
                    vectors.append((0.5, 0.5))
            return vectors

    semantically_related = _case(
        "study",
        action_at=BASE - timedelta(days=5),
        context_text="시험 준비 끝?",
        target_text="아직",
        conversation_id="old-study",
    )
    unrelated = _case(
        "dinner",
        action_at=BASE - timedelta(days=1),
        context_text="저녁 메뉴 정함?",
        target_text="치킨",
        conversation_id="old-dinner",
    )
    query = _case(
        "query",
        action_at=BASE + timedelta(minutes=5),
        context_text="공부 다 했냐",
        target_text="hidden",
        conversation_id="query-thread",
    )
    index = CutoffExampleIndex.from_replay_cases(
        [semantically_related, unrelated],
        embedding_provider=FakeEmbeddingProvider(),
    )

    results = index.search(query, cutoff=BASE, k=2)

    assert results[0].example.case_id == "study"
    assert results[0].embedding_similarity == 1.0
    assert results[0].semantic_similarity > results[0].lexical_similarity


def test_kakao_primary_weight_breaks_equal_similarity_tie() -> None:
    kakao = _case(
        "kakao",
        action_at=BASE - timedelta(days=2),
        context_text="과제 다함?",
        target_text="아직",
        platform="kakao",
        weight=1.0,
        conversation_id="kakao-old",
    )
    instagram = _case(
        "instagram",
        action_at=BASE - timedelta(days=2),
        context_text="과제 다함?",
        target_text="ㄴㄴ",
        platform="instagram",
        weight=0.55,
        conversation_id="ig-old",
    )
    query = _case(
        "query",
        action_at=BASE + timedelta(minutes=5),
        context_text="과제 다함?",
        target_text="hidden",
        platform="kakao",
        weight=1.0,
        conversation_id="query-thread",
    )
    index = CutoffExampleIndex.from_replay_cases([instagram, kakao, query])

    results = index.search(query, cutoff=BASE, k=2)

    assert [result.example.case_id for result in results] == ["kakao", "instagram"]


def test_query_ranking_does_not_depend_on_hidden_target_text() -> None:
    candidate = _case(
        "candidate",
        action_at=BASE - timedelta(days=1),
        context_text="면접 준비 됨?",
        target_text="아직ㅋㅋ",
        conversation_id="old",
    )
    query_a = _case(
        "query-a",
        action_at=BASE + timedelta(minutes=5),
        context_text="면접 준비 됨?",
        target_text="secret A",
        conversation_id="query-a-thread",
    )
    query_b = _case(
        "query-b",
        action_at=BASE + timedelta(minutes=5),
        context_text="면접 준비 됨?",
        target_text="totally different secret B",
        conversation_id="query-b-thread",
    )
    index = CutoffExampleIndex.from_replay_cases([candidate])

    score_a = index.search(query_a, cutoff=BASE, k=1)[0].score
    score_b = index.search(query_b, cutoff=BASE, k=1)[0].score

    assert score_a == score_b
