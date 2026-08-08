from __future__ import annotations


class GenerationProviderError(RuntimeError):
    """Base class for provider failures that are safe to show without prompt data."""


class TransientGenerationError(GenerationProviderError):
    """Temporary provider/network failure; the scheduled behavior should be retried."""


class PermanentGenerationError(GenerationProviderError):
    """Non-retryable provider failure until configuration or credentials change."""
