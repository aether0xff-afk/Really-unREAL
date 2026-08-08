from backend.generation_context import (
    GenerationContextPacket,
    RetrievedGenerationExample,
    RetrievedResponseShape,
    VisibleGenerationMessage,
)
from backend.persona.cutoff import CutoffLanguageProfile
from backend.providers.nvidia import NvidiaNIMLanguageModel


def _packet() -> GenerationContextPacket:
    return GenerationContextPacket(
        person_id="target",
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
            person_id="target",
            cutoff="2026-08-08T12:00:00",
            message_count=20,
            effective_message_weight=20.0,
            weighted_mean_char_length=5.0,
            weighted_short_message_ratio=0.8,
            weighted_laugh_expression_ratio=0.4,
            weighted_cry_expression_ratio=0.0,
            frequent_tokens=(("ㅋㅋ", 8.0),),
            platform_message_counts={"kakao": 20},
        ),
        retrieved_examples=(
            RetrievedGenerationExample(
                platform="kakao",
                action="REPLY",
                context_texts=("뭐해",),
                burst_size=1,
                retrieval_score=0.8,
                response_shape=RetrievedResponseShape(
                    message_lengths=(3,),
                    question_count=0,
                    laugh_expression_count=1,
                    cry_expression_count=0,
                    endings=("ㅋㅋ",),
                ),
            ),
        ),
    )


def test_nvidia_adapter_uses_bearer_key_and_parses_json() -> None:
    captured = {}

    def transport(url, headers, payload, timeout):
        captured.update(
            url=url,
            headers=headers,
            payload=payload,
            timeout=timeout,
        )
        return {
            "choices": [
                {
                    "message": {
                        "content": '```json\n{"messages":["집ㅋㅋ"]}\n```'
                    }
                }
            ]
        }

    model = NvidiaNIMLanguageModel(api_key="secret-test-key", transport=transport)
    result = model.generate_burst(_packet())

    assert result.messages == ("집ㅋㅋ",)
    assert captured["url"].endswith("/v1/chat/completions")
    assert captured["headers"]["Authorization"] == "Bearer secret-test-key"
    assert captured["payload"]["model"] == "nvidia/nemotron-3-ultra-550b-a55b"
    assert captured["payload"]["chat_template_kwargs"]["enable_thinking"] is False
    assert "secret-test-key" not in captured["payload"]["messages"][0]["content"]
    assert "집ㅋㅋ" not in captured["payload"]["messages"][0]["content"]


def test_nvidia_adapter_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    try:
        NvidiaNIMLanguageModel()
    except ValueError as exc:
        assert "NVIDIA_API_KEY" in str(exc)
    else:
        raise AssertionError("missing NVIDIA_API_KEY should fail")
