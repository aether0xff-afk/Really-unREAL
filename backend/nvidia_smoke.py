from __future__ import annotations

import json

from backend.generation_context import (
    GenerationContextPacket,
    RetrievedGenerationExample,
    RetrievedResponseShape,
    VisibleGenerationMessage,
)
from backend.persona.cutoff import CutoffLanguageProfile
from backend.providers.nvidia import NvidiaNIMLanguageModel


def _synthetic_packet() -> GenerationContextPacket:
    return GenerationContextPacket(
        person_id="synthetic-target",
        observation_end="2026-08-08T12:00:00",
        chosen_action="REPLY",
        visible_context=(
            VisibleGenerationMessage(
                timestamp="2026-08-08T11:59:00",
                sender_person_id="self",
                text="뭐함",
                platform="kakao",
            ),
        ),
        language_profile=CutoffLanguageProfile(
            person_id="synthetic-target",
            cutoff="2026-08-08T12:00:00",
            message_count=30,
            effective_message_weight=30.0,
            weighted_mean_char_length=5.4,
            weighted_short_message_ratio=0.75,
            weighted_laugh_expression_ratio=0.35,
            weighted_cry_expression_ratio=0.02,
            frequent_tokens=(("ㅋㅋ", 9.0), ("아니", 4.0), ("근데", 3.0)),
            platform_message_counts={"kakao": 30},
            weighted_question_ratio=0.08,
            weighted_exclamation_ratio=0.01,
            weighted_no_terminal_punctuation_ratio=0.91,
            frequent_endings=(("ㅋㅋ", 8.0), ("ㅇㅇ", 3.0)),
        ),
        retrieved_examples=(
            RetrievedGenerationExample(
                platform="kakao",
                action="REPLY",
                context_texts=("뭐해",),
                burst_size=1,
                retrieval_score=0.82,
                response_shape=RetrievedResponseShape(
                    message_lengths=(1,),
                    question_count=0,
                    laugh_expression_count=0,
                    cry_expression_count=0,
                    endings=("집",),
                ),
            ),
            RetrievedGenerationExample(
                platform="kakao",
                action="REPLY",
                context_texts=("지금 뭐함",),
                burst_size=1,
                retrieval_score=0.77,
                response_shape=RetrievedResponseShape(
                    message_lengths=(6,),
                    question_count=0,
                    laugh_expression_count=1,
                    cry_expression_count=0,
                    endings=("ㅋㅋ",),
                ),
            ),
        ),
    )


def main() -> None:
    model = NvidiaNIMLanguageModel()
    burst = model.generate_burst(_synthetic_packet())
    print(
        json.dumps(
            {
                "provider": "nvidia_nim",
                "model": model.model,
                "status": "ok",
                "burst_size": len(burst.messages),
                "messages": list(burst.messages),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
