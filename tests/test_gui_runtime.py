from __future__ import annotations

import backend.gui_runtime as gui_runtime


def test_nvidia_gui_path_uses_bounded_request_budget(monkeypatch) -> None:
    captured: dict[str, object] = {}
    fake_evidence = object()

    monkeypatch.setattr(
        gui_runtime,
        "_target_evidence",
        lambda conversations, self_alias, target_alias: (None, None, fake_evidence),
    )
    monkeypatch.setattr(
        gui_runtime,
        "require_private_context_route",
        lambda base_url, allow_remote_private_context=False: None,
    )

    class FakeNvidia:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)
            self.model = kwargs["model"]

    monkeypatch.setattr(gui_runtime, "NvidiaNIMLanguageModel", FakeNvidia)
    monkeypatch.setattr(
        gui_runtime,
        "run_generation_replay_interactive",
        lambda **kwargs: {"generated_cases": 0, "requested_cases": 0},
    )

    result = gui_runtime.run_quick_generation_interactive(
        [],
        self_alias="me",
        target_alias="friend",
        provider="nvidia",
        model="model",
        base_url="https://example.invalid/v1",
        api_key="secret",
        allow_remote_private_context=True,
        limit=3,
    )

    assert captured["timeout_seconds"] == 45.0
    assert captured["max_attempts"] == 1
    assert captured["max_format_attempts"] == 1
    assert result["provider"] == "nvidia"
    assert result["target"] == "friend"


def test_gui_timeout_budget_is_far_below_previous_worst_case() -> None:
    previous_per_case_worst_seconds = 90 * 3 * 2
    new_per_case_worst_seconds = (
        gui_runtime.NVIDIA_GUI_TIMEOUT_SECONDS
        * gui_runtime.NVIDIA_GUI_MAX_ATTEMPTS
        * gui_runtime.NVIDIA_GUI_FORMAT_ATTEMPTS
    )
    assert new_per_case_worst_seconds == 45.0
    assert new_per_case_worst_seconds < previous_per_case_worst_seconds / 10
