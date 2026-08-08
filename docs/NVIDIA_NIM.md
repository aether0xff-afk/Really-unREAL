# NVIDIA NIM generation

Really-unREAL uses NVIDIA hosted NIM only after the temporal policy has already chosen `REPLY` or `INITIATE`. The language model does not decide whether a message should exist.

## Hosted endpoint

The default adapter targets:

```text
base URL: https://integrate.api.nvidia.com/v1
model:    nvidia/nemotron-3-ultra-550b-a55b
secret:   NVIDIA_API_KEY
```

`backend.providers.nvidia.NvidiaNIMLanguageModel` uses the OpenAI-compatible `/chat/completions` endpoint through the Python standard library, so the project does not need a provider SDK dependency.

The adapter disables reasoning mode for persona-message generation and requests a small response budget. The provider-independent prompt still requires exactly:

```json
{"messages": ["...", "..."]}
```

A small recovery layer accepts accidental Markdown code fences but rejects responses with no JSON object or no non-empty messages.

## Secret handling

Never commit an NVIDIA API key. Local runs read it from the environment:

```bash
export NVIDIA_API_KEY='...'
```

GitHub Actions uses the repository secret named `NVIDIA_API_KEY`. The workflow never prints the secret or sends it inside the model prompt.

## GitHub Actions smoke test

`.github/workflows/nvidia-smoke.yml` sends only a synthetic conversation packet. No real KakaoTalk or Instagram data is committed or uploaded to Actions for this test.

The workflow can be started manually and also runs when its own workflow file changes. It publishes the `nvidia-nim-smoke` commit status so external tooling can verify whether the hosted endpoint, secret, model ID, and response contract are working.

## Private Historical Replay

Real replay stays local because the private archives and `identity.local.json` are intentionally gitignored.

```bash
python -m backend.nvidia_replay \
  ./data/raw/kakao_bundle.zip \
  ./data/raw/instagram_export.zip \
  ./identity.local.json \
  PERSON_ID \
  --sources kakao \
  --limit 20
```

The first production experiment defaults to Kakao-only evidence. `--sources fused` adds supplemental Instagram evidence for the later ablation.

The CLI does not print prompts, retrieved private examples, generated message text, or held-out real responses. It reports aggregate generation metrics only.

## Evaluation boundary

For each held-out case:

1. the selected temporal model predicts `WAIT` or an observable action at the real evaluation time;
2. if it predicts `WAIT`, no LLM call is made and the case is counted as a temporal miss;
3. otherwise cutoff-safe persona statistics and cutoff-safe RAG examples build the generation packet;
4. NVIDIA NIM generates the burst;
5. only then does the evaluator read the held-out real burst.

This keeps temporal-policy errors separate from language-generation quality and prevents future-message leakage.
