from backend.generation_context import (
    GenerationContextPacket,
    RetrievedGenerationExample,
    RetrievedResponseShape,
    VisibleGenerationMessage,
)
from backend.persona.cutoff import CutoffLanguageProfile
from backend.providers.openai_compatible import OpenAICompatibleLanguageModel


def _packet() -> GenerationContextPacket:
    return GenerationContextPacket(
        person_id="target",
        observation_end="2026-08-08T12:00:00",
        chosen_action="REPLY",
        visible_context=(
            VisibleGenerationMessage(
                timestamp="2026-08-08T11:59:00",
                sender_person_id="other",
                text="뭐함",
                platform="kakao",
            ),
        ),
        language_profile=CutoffLanguageProfile(
            person_id="target",
            cutoff="2026-08-08T12:00:00",
            message_count=10,
            effective_message_weight=10.0,
            weighted_mean_char_length=4.0,
            weighted_short_message_ratio=0.8,
            weighted_laugh_expression_ratio=0.2,
            weighted_cry_expression_ratio=0.0,
            frequent_tokens=(("ㅋㅋ", 2.0),),
            platform_message_counts={"kakao": 10},
        ),
        retrieved_examples=(
            RetrievedGenerationExample(
                platform="kakao",
                action="REPLY",
                context_texts=("뭐해",),
                burst_size=1,
                retrieval_score=0.7,
                response_shape=RetrievedResponseShape(
                    message_lengths=(2,),
                    question_count=0,
                    laugh_expression_count=0,
                    cry_expression_count=0,
                    endings=("집",),
                ),
            ),
        ),
    )


def test_openai_compatible_provider_parses_local_completion_without_api_key() -> None:
    captured = {}

    def transport(url, headers, payload, timeout):
        captured.update(url=url, headers=headers, payload=payload, timeout=timeout)
        return {"choices": [{"message": {"content": '{"messages":["집"]}'}}]}

    model = OpenAICompatibleLanguageModel(
        model="local-model",
        transport=transport,
    )
    result = model.generate_burst(_packet())

    assert result.messages == ("집",)
    assert captured["url"] == "http://127.0.0.1:1234/v1/chat/completions"
    assert "Authorization" not in captured["headers"]
    assert captured["payload"]["model"] == "local-model"
