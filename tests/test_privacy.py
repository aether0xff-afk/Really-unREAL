import pytest

from backend.privacy import (
    EndpointScope,
    RemotePrivateContextConsentRequired,
    decide_private_context_route,
    endpoint_scope,
    require_private_context_route,
)


def test_loopback_provider_is_local() -> None:
    assert endpoint_scope("http://127.0.0.1:1234/v1") == EndpointScope.LOCAL
    assert endpoint_scope("http://localhost:8000/v1") == EndpointScope.LOCAL
    decision = require_private_context_route("http://localhost:1234/v1")
    assert decision.private_context_allowed is True


def test_remote_private_context_requires_explicit_consent() -> None:
    decision = decide_private_context_route("https://example.com/v1")
    assert decision.scope == EndpointScope.REMOTE
    assert decision.private_context_allowed is False

    with pytest.raises(RemotePrivateContextConsentRequired):
        require_private_context_route("https://example.com/v1")

    allowed = require_private_context_route(
        "https://example.com/v1",
        allow_remote_private_context=True,
    )
    assert allowed.private_context_allowed is True
