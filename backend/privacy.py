from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from enum import StrEnum
from urllib.parse import urlparse


class EndpointScope(StrEnum):
    LOCAL = "local"
    REMOTE = "remote"


@dataclass(frozen=True, slots=True)
class PrivacyDecision:
    base_url: str
    scope: EndpointScope
    private_context_allowed: bool
    reason: str


class RemotePrivateContextConsentRequired(RuntimeError):
    pass


def endpoint_scope(base_url: str) -> EndpointScope:
    parsed = urlparse(base_url)
    hostname = (parsed.hostname or "").strip().lower()
    if hostname in {"localhost", "localhost.localdomain"}:
        return EndpointScope.LOCAL
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return EndpointScope.REMOTE
    return EndpointScope.LOCAL if address.is_loopback else EndpointScope.REMOTE


def decide_private_context_route(
    base_url: str,
    *,
    allow_remote_private_context: bool = False,
) -> PrivacyDecision:
    scope = endpoint_scope(base_url)
    if scope == EndpointScope.LOCAL:
        return PrivacyDecision(
            base_url=base_url,
            scope=scope,
            private_context_allowed=True,
            reason="loopback endpoint keeps the request on this device",
        )
    if allow_remote_private_context:
        return PrivacyDecision(
            base_url=base_url,
            scope=scope,
            private_context_allowed=True,
            reason="remote private-context transmission was explicitly enabled",
        )
    return PrivacyDecision(
        base_url=base_url,
        scope=scope,
        private_context_allowed=False,
        reason="remote endpoints require explicit consent for private conversation context",
    )


def require_private_context_route(
    base_url: str,
    *,
    allow_remote_private_context: bool = False,
) -> PrivacyDecision:
    decision = decide_private_context_route(
        base_url,
        allow_remote_private_context=allow_remote_private_context,
    )
    if not decision.private_context_allowed:
        raise RemotePrivateContextConsentRequired(
            "Private conversation context cannot be sent to a remote model endpoint "
            "without explicit --allow-remote-private-context consent."
        )
    return decision
