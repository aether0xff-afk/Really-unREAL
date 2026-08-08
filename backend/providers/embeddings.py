from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Callable, Sequence


EmbeddingTransport = Callable[[str, dict[str, str], dict[str, object], float], dict[str, object]]


def _default_transport(
    url: str,
    headers: dict[str, str],
    payload: dict[str, object],
    timeout: float,
) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


@dataclass(slots=True)
class OpenAICompatibleEmbeddingProvider:
    """Small dependency-free adapter for local or remote embedding servers.

    The provider is intentionally generic. A local OpenAI-compatible server can
    be used without an API key; a remote endpoint may use one. Callers must make
    the privacy choice explicitly because text sent to a remote endpoint leaves
    the device.
    """

    base_url: str
    model: str
    api_key: str | None = None
    timeout: float = 30.0
    transport: EmbeddingTransport = _default_transport

    @classmethod
    def from_env(cls) -> "OpenAICompatibleEmbeddingProvider":
        base_url = os.environ.get("EMBEDDING_BASE_URL")
        model = os.environ.get("EMBEDDING_MODEL")
        if not base_url or not model:
            raise ValueError("EMBEDDING_BASE_URL and EMBEDDING_MODEL are required")
        return cls(
            base_url=base_url,
            model=model,
            api_key=os.environ.get("EMBEDDING_API_KEY"),
        )

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        if not texts:
            return ()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload: dict[str, object] = {
            "model": self.model,
            "input": list(texts),
        }
        response = self.transport(
            self.base_url.rstrip("/") + "/embeddings",
            headers,
            payload,
            self.timeout,
        )
        data = response.get("data")
        if not isinstance(data, list) or len(data) != len(texts):
            raise ValueError("embedding endpoint returned unexpected data length")

        ordered = sorted(
            data,
            key=lambda item: int(item.get("index", 0)) if isinstance(item, dict) else 0,
        )
        vectors: list[tuple[float, ...]] = []
        dimension: int | None = None
        for item in ordered:
            if not isinstance(item, dict) or not isinstance(item.get("embedding"), list):
                raise ValueError("embedding endpoint returned malformed vector")
            vector = tuple(float(value) for value in item["embedding"])
            if not vector:
                raise ValueError("embedding vector must not be empty")
            if dimension is None:
                dimension = len(vector)
            elif len(vector) != dimension:
                raise ValueError("embedding endpoint returned inconsistent dimensions")
            vectors.append(vector)
        return tuple(vectors)
