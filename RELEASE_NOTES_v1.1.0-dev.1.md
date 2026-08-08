# Really-unREAL v1.1.0-dev.1

First 1.1 development build: provider-failure resilience.

## What changed

- A scheduled `REPLY` or `INITIATE` is now independent from model-provider availability.
- HTTP 429/5xx, network failures, and timeouts are classified as transient generation failures.
- Transient failures preserve the scheduled action and retry generation with bounded backoff.
- Credential/configuration and invalid-format failures become `BLOCKED` instead of silently deleting behavior.
- The live GUI shows retry/blocked state and exposes an explicit **생성 재시도** action.
- Existing <=1.0.5 local SQLite stores are migrated in place with retry metadata.
- No raw chat content or API keys are added to release artifacts.

## Why

In 1.0.4/1.0.5, an NVIDIA HTTP 503 could erase a reply that the behavior model had already scheduled. In dev.1, provider availability changes only generation delivery, not the simulated person's behavioral decision.
