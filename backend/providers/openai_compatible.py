from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from backend.generation import GeneratedBurst, generation_prompt
from backend.generation_context import GenerationContextPacket
from backend.providers.errors import PermanentGenerationError, TransientGenerationError


Transport = Callable[[str, dict[str, str], dict[str, Any], float], dict[str, Any]]


def _urllib_transport(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise ValueError("model response did not contain a JSON object")
    return stripped[start : end + 1]


class OpenAICompatibleLanguageModel:
    """Minimal OpenAI-compatible chat-completions adapter."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://127.0.0.1:1234/v1",
        api_key: str | None = None,
        api_key_env: str | None = None,
        temperature: float = 0.65,
        top_p: float = 0.9,
        max_tokens: int = 256,
        timeout_seconds: float = 90.0,
        max_attempts: int = 3,
        format_attempts: int = 2,
        transport: Transport | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model is required")
        if max_attempts < 1 or format_attempts < 1:
            raise ValueError("attempt counts must be >= 1")
        resolved_key = api_key
        if resolved_key is None and api_key_env:
            resolved_key = os.environ.get(api_key_env)

        self.model = model
        self.base_url = base_url.rstrip("/")
        self._api_key = resolved_key
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_tokens = int(max_tokens)
        self.timeout_seconds = float(timeout_seconds)
        self.max_attempts = int(max_attempts)
        self.format_attempts = int(format_attempts)
        self._transport = transport or _urllib_transport

    def _payload(self, packet: GenerationContextPacket) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": generation_prompt(packet)}],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

    def _request(self, packet: GenerationContextPacket) -> dict[str, Any]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        url = f"{self.base_url}/chat/completions"
        payload = self._payload(packet)

        for attempt in range(1, self.max_attempts + 1):
            try:
                return self._transport(url, headers, payload, self.timeout_seconds)
            except urllib.error.HTTPError as exc:
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if retryable:
                    if attempt >= self.max_attempts:
                        raise TransientGenerationError(
                            f"Local/OpenAI-compatible provider temporarily unavailable (HTTP {exc.code})"
                        ) from exc
                else:
                    raise PermanentGenerationError(
                        f"Local/OpenAI-compatible request rejected (HTTP {exc.code})"
                    ) from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt >= self.max_attempts:
                    raise TransientGenerationError(
                        "Local/OpenAI-compatible provider temporarily unreachable"
                    ) from exc
            time.sleep(min(2 ** (attempt - 1), 4))
        raise TransientGenerationError("Local/OpenAI-compatible request temporarily failed")

    def generate_burst(self, packet: GenerationContextPacket) -> GeneratedBurst:
        last_error: Exception | None = None
        for _ in range(self.format_attempts):
            data = self._request(packet)
            try:
                content = data["choices"][0]["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("model returned empty assistant content")
                return GeneratedBurst.from_json(_extract_json_object(content))
            except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
        raise PermanentGenerationError(
            "Local/OpenAI-compatible model returned an invalid message format"
        ) from last_error
