from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

from backend.generation_context import (
    GenerationContextPacket,
    RetrievedGenerationExample,
    VisibleGenerationMessage,
)
from backend.persona.cutoff import CutoffLanguageProfile
from backend.providers.nvidia import NvidiaNIMLanguageModel


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
    )


def _examples() -> tuple[RetrievedGenerationExample, ...]:
    return (
        RetrievedGenerationExample(
            platform="kakao",
            action="REPLY",
            context_texts=("뭐해",),
            response_texts=("집",),
            burst_size=1,
            retrieval_score=0.91,
        ),
        RetrievedGenerationExample(
            platform="kakao",
            action="REPLY",
            context_texts=("과제 다함?",),
            response_texts=("아니ㅋㅋ", "아직"),
            burst_size=2,
            retrieval_score=0.86,
        ),
        RetrievedGenerationExample(
            platform="kakao",
            action="REPLY",
            context_texts=("낼 몇시에 감",),
            response_texts=("몰라", "평소대로 갈듯"),
            burst_size=2,
            retrieval_score=0.82,
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
        "turns": transcript,
    }
    result_path = Path(os.environ.get("RESULT_PATH", "nvidia-conversation-smoke.json"))
    result_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
