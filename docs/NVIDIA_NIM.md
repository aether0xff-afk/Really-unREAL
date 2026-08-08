# NVIDIA NIM generation

Really-unREAL uses NVIDIA NIM only after the behavior/timing layer has selected an observable action. The language model never decides whether the simulator should talk merely because it is running.

## Hosted endpoint

Default hosted configuration:

```text
base URL: https://integrate.api.nvidia.com/v1
model:    nvidia/nemotron-3-ultra-550b-a55b
secret:   NVIDIA_API_KEY
```

`backend.providers.nvidia.NvidiaNIMLanguageModel` calls the OpenAI-compatible `/chat/completions` API with the Python standard library. Reasoning mode is disabled for terse observable message generation, and malformed JSON output receives a limited format retry.

The provider-independent output contract is:

```json
{"messages": ["...", "..."]}
```

## Secret handling

Never commit the API key. Local runs read `NVIDIA_API_KEY` from the environment. GitHub Actions uses a repository secret of the same name and never prints it or places it in the prompt.

## Synthetic GitHub Actions smoke

`.github/workflows/nvidia-smoke.yml` uses synthetic conversation context only. No real KakaoTalk or Instagram archive is committed or uploaded to Actions.

The workflow verifies:

- the v1 core pytest suite;
- the hosted NVIDIA endpoint/secret/model contract;
- a four-turn synthetic project conversation;
- the combined `v1-release-smoke` commit status.

## Private replay requires explicit remote consent

The v1 default is local-first. A loopback model endpoint is allowed automatically, but private conversation context may not be sent to a remote endpoint unless the user explicitly opts in.

The recommended provider-agnostic command is:

```bash
python -m backend.replay_generate \
  ./data/raw/kakao_bundle.zip \
  ./data/raw/instagram_export.zip \
  ./identity.local.json \
  PERSON_ID \
  --provider nvidia \
  --sources kakao \
  --limit 20 \
  --allow-remote-private-context
```

SELF twin:

```bash
python -m backend.replay_generate \
  ./data/raw/kakao_bundle.zip \
  ./data/raw/instagram_export.zip \
  ./identity.local.json \
  --self-twin \
  --provider nvidia \
  --allow-remote-private-context
```

`backend.nvidia_replay` remains as a compatibility runner and enforces the same remote-context gate.

The CLI reports aggregate evaluation metrics only. It does not print prompts, held-out real responses, or generated private message text.

## Evaluation boundary

Timing and content are evaluated separately.

For each held-out real event:

1. the selected temporal model predicts a timing distribution/representative delay;
2. timing is scored against the timestamp interval;
3. the observable REPLY-vs-INITIATE proxy is derived only from visible context when trustworthy;
4. a cutoff-safe persona/topic/event/RAG packet is built;
5. NVIDIA generates a message burst;
6. only afterward does the evaluator read the held-out real burst for content/style metrics.

A timing miss does not erase the language-model measurement. This prevents timing quality and language quality from being accidentally conflated.

## Strongest privacy mode

Use `backend.replay_generate --provider local` with a loopback OpenAI-compatible model server. If dense retrieval is enabled, use a loopback embedding endpoint as well. In that configuration, private generation and retrieval text does not need to leave the device.
