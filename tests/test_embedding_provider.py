from backend.providers.embeddings import OpenAICompatibleEmbeddingProvider


def test_embedding_provider_supports_local_endpoint_without_key() -> None:
    captured = {}

    def transport(url, headers, payload, timeout):
        captured.update(url=url, headers=headers, payload=payload, timeout=timeout)
        return {
            "data": [
                {"index": 0, "embedding": [1.0, 0.0]},
                {"index": 1, "embedding": [0.0, 1.0]},
            ]
        }

    provider = OpenAICompatibleEmbeddingProvider(
        base_url="http://127.0.0.1:1234/v1",
        model="local-embed",
        transport=transport,
    )
    vectors = provider.embed(["시험", "저녁"])

    assert vectors == ((1.0, 0.0), (0.0, 1.0))
    assert captured["url"] == "http://127.0.0.1:1234/v1/embeddings"
    assert captured["payload"]["model"] == "local-embed"
    assert "Authorization" not in captured["headers"]


def test_embedding_provider_uses_bearer_key_only_when_configured() -> None:
    def transport(url, headers, payload, timeout):
        assert headers["Authorization"] == "Bearer secret"
        return {"data": [{"index": 0, "embedding": [1.0]}]}

    provider = OpenAICompatibleEmbeddingProvider(
        base_url="https://example.invalid/v1",
        model="embed",
        api_key="secret",
        transport=transport,
    )

    assert provider.embed(["x"]) == ((1.0,),)
