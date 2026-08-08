from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

from backend.generation_context import (
    GenerationContextPacket,
    RetrievedGenerationExample,
    RetrievedResponseShape,
    VisibleGenerationMessage,
)
from backend.persona.cutoff import CutoffLanguageProfile
from backend.providers.nvidia import NvidiaNIMLanguageModel
from backend.topic_memory import ObservableTopicCue, TopicMemorySnapshot


USER_TURNS = (
    "뭐함",
    "아 나 지금 과제하다가 개빡침ㅋㅋ",
    "너는 다 했냐",
    "낼 학교 몇시에 감?",
)


def _profile() -> CutoffLanguageProfile:
    return CutoffLanguageProfile(
        person_id="synthetic-target",
        cutoff="2026-08-08T14:00:00",
        message_count=240,
        effective_message_weight=240.0,
        weighted_mean_char_length=8.4,
        weighted_short_message_ratio=0.71,
        weighted_laugh_expression_ratio=0.28,
        weighted_cry_expression_ratio=0.03,
        frequent_tokens=(("ㅋㅋ", 44.0), ("아니", 21.0), ("근데", 17.0), ("ㅇㅇ", 15.0)),
        platform_message_counts={"kakao": 240},
        weighted_question_ratio=0.12,
        weighted_exclamation_ratio=0.01,
        weighted_no_terminal_punctuation_ratio=0.94,
        frequent_endings=(("ㅋㅋ", 32.0), ("ㅇㅇ", 18.0), ("아님", 9.0)),
        profile_scope="relationship_blend",
        focused_message_count=80,
        focus_weight_multiplier=2.0,
    )


def _topic_memory(now: datetime) -> TopicMemorySnapshot:
    return TopicMemorySnapshot(
        cutoff=now.isoformat(),
        horizon_days=120.0,
        cues=(
            ObservableTopicCue(
                token="과제",
                score=3.2,
                mention_count=8,
                focused_mention_count=6,
                last_seen_at=(now - timedelta(days=1)).isoformat(),
            ),
            ObservableTopicCue(
                token="학교",
                score=2.4,
                mention_count=6,
                focused_mention_count=5,
                last_seen_at=(now - timedelta(days=2)).isoformat(),
            ),
        ),
    )


def _shape(
    *lengths: int,
    question_count: int = 0,
    laugh_count: int = 0,
    endings: tuple[str, ...] = (),
) -> RetrievedResponseShape:
    return RetrievedResponseShape(
        message_lengths=tuple(lengths),
        question_count=question_count,
        laugh_expression_count=laugh_count,
        cry_expression_count=0,
        endings=endings,
    )


def _examples() -> tuple[RetrievedGenerationExample, ...]:
    return (
        RetrievedGenerationExample(
            platform="kakao",
            action="REPLY",
            context_texts=("뭐해",),
            burst_size=1,
            retrieval_score=0.91,
            response_shape=_shape(1, endings=("집",)),
        ),
        RetrievedGenerationExample(
            platform="kakao",
            action="REPLY",
            context_texts=("과제 다함?",),
            burst_size=2,
            retrieval_score=0.86,
            response_shape=_shape(4, 2, laugh_count=1, endings=("ㅋㅋ", "아직")),
        ),
        RetrievedGenerationExample(
            platform="kakao",
            action="REPLY",
            context_texts=("낼 몇시에 감",),
            burst_size=2,
            retrieval_score=0.82,
            response_shape=_shape(2, 7, endings=("몰라", "갈듯")),
        ),
    )


def main() -> None:
    model = NvidiaNIMLanguageModel(
        temperature=0.55,
        top_p=0.9,
        max_tokens=160,
    )
    profile = _profile()
    examples = _examples()
    now = datetime(2026, 8, 8, 14, 0)
    visible: list[VisibleGenerationMessage] = []
    transcript: list[dict[str, object]] = []

    for index, user_text in enumerate(USER_TURNS, start=1):
        now += timedelta(minutes=2)
        visible.append(
            VisibleGenerationMessage(
                timestamp=now.isoformat(),
                sender_person_id="self",
                text=user_text,
                platform="kakao",
            )
        )
        packet = GenerationContextPacket(
            person_id="synthetic-target",
            observation_end=now.isoformat(),
            chosen_action="REPLY",
            visible_context=tuple(visible[-12:]),
            language_profile=profile,
            retrieved_examples=examples,
            topic_memory=_topic_memory(now),
        )
        burst = model.generate_burst(packet)
        transcript.append(
            {
                "turn": index,
                "user": user_text,
                "target": list(burst.messages),
            }
        )
        for message in burst.messages:
            now += timedelta(seconds=12)
            visible.append(
                VisibleGenerationMessage(
                    timestamp=now.isoformat(),
                    sender_person_id="synthetic-target",
                    text=message,
                    platform="kakao",
                )
            )

    output = {
        "provider": "nvidia-nim",
        "model": model.model,
        "synthetic_persona": True,
        "raw_retrieved_responses": False,
        "topic_memory": True,
        "turns": transcript,
    }
    result_path = Path(os.environ.get("RESULT_PATH", "nvidia-conversation-smoke.json"))
    result_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
