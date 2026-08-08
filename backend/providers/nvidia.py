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


def _extract_json_payload(text: str) -> str:
    """Recover the JSON object when a model wraps it in Markdown fences/prose."""

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
    if start == -1 or end == -1 or end < start:
        raise ValueError("NVIDIA NIM response did not contain a JSON object")
    return stripped[start : end + 1]


class NvidiaNIMLanguageModel:
    """Hosted NVIDIA NIM adapter for the provider-agnostic burst contract.

    The API key is read from ``NVIDIA_API_KEY`` by default and is never included
    in prompts, return values, or exception messages created by this class.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "nvidia/nemotron-3-ultra-550b-a55b",
        base_url: str = "https://integrate.api.nvidia.com/v1",
        temperature: float = 0.65,
        top_p: float = 0.9,
        max_tokens: int = 256,
        timeout_seconds: float = 90.0,
        max_attempts: int = 3,
        transport: Transport | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("NVIDIA_API_KEY")
        if not resolved_key:
            raise ValueError("NVIDIA_API_KEY is required")
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")

        self._api_key = resolved_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_tokens = int(max_tokens)
        self.timeout_seconds = float(timeout_seconds)
        self.max_attempts = int(max_attempts)
        self._transport = transport or _urllib_transport

    def _completion_payload(self, packet: GenerationContextPacket) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": generation_prompt(packet),
                }
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stream": False,
            # Persona replay needs terse observable text, not hidden reasoning.
            "chat_template_kwargs": {"enable_thinking": False},
        }

    def _request_completion(self, packet: GenerationContextPacket) -> dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = self._completion_payload(packet)

        for attempt in range(1, self.max_attempts + 1):
            try:
                return self._transport(url, headers, payload, self.timeout_seconds)
            except urllib.error.HTTPError as exc:
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable or attempt >= self.max_attempts:
                    raise RuntimeError(
                        f"NVIDIA NIM request failed with HTTP {exc.code}"
                    ) from exc
            except urllib.error.URLError as exc:
                if attempt >= self.max_attempts:
                    raise RuntimeError("NVIDIA NIM request failed") from exc

            time.sleep(min(2 ** (attempt - 1), 4))

        raise RuntimeError("NVIDIA NIM request failed")

    def generate_burst(self, packet: GenerationContextPacket) -> GeneratedBurst:
        data = self._request_completion(packet)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("NVIDIA NIM response is missing assistant content") from exc
        if not isinstance(content, str) or not content.strip():
            raise ValueError("NVIDIA NIM returned empty assistant content")
        return GeneratedBurst.from_json(_extract_json_payload(content))
