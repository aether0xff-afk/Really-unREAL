from __future__ import annotations

import argparse
import json
import os

from backend.fusion import collect_person_evidence
from backend.identity import IdentityMap
from backend.ingest.archive import load_kakao_archive
from backend.ingest.instagram import load_instagram_export
from backend.privacy import require_private_context_route
from backend.providers.embeddings import OpenAICompatibleEmbeddingProvider
from backend.providers.nvidia import NvidiaNIMLanguageModel
from backend.providers.openai_compatible import OpenAICompatibleLanguageModel
from backend.replay_generation import compare_source_summaries, run_generation_replay
from backend.retrieval import EmbeddingProvider
from backend.twin import TwinMode, resolve_twin_spec


LOCAL_BASE_URL = "http://127.0.0.1:1234/v1"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run leakage-safe content/timing replay with a local or hosted model"
    )
    parser.add_argument("kakao_archive")
    parser.add_argument("instagram_archive")
    parser.add_argument("identity_map")
    parser.add_argument("person_id", nargs="?")
    parser.add_argument("--self-twin", action="store_true")
    parser.add_argument("--provider", choices=("local", "nvidia"), default="local")
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env", default="LOCAL_LLM_API_KEY")
    parser.add_argument("--allow-remote-private-context", action="store_true")
    parser.add_argument("--sources", choices=("kakao", "fused", "both"), default="kakao")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--raw-rag-responses", type=int, default=0)
    parser.add_argument("--embedding-base-url")
    parser.add_argument("--embedding-model")
    parser.add_argument("--embedding-weight", type=float, default=0.70)
    args = parser.parse_args()

    identities = IdentityMap.from_json(args.identity_map)
    try:
        spec = resolve_twin_spec(
            identities,
            mode=TwinMode.SELF if args.self_twin else TwinMode.PERSON,
            person_id=args.person_id,
        )
    except (KeyError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    if args.provider == "nvidia":
        base_url = args.base_url or NVIDIA_BASE_URL
        model_name = args.model or NVIDIA_MODEL
        require_private_context_route(
            base_url,
            allow_remote_private_context=args.allow_remote_private_context,
        )
        language_model = NvidiaNIMLanguageModel(model=model_name, base_url=base_url)
    else:
        base_url = args.base_url or LOCAL_BASE_URL
        if not args.model:
            raise SystemExit("--model is required for --provider local")
        model_name = args.model
        require_private_context_route(
            base_url,
            allow_remote_private_context=args.allow_remote_private_context,
        )
        language_model = OpenAICompatibleLanguageModel(
            model=model_name,
            base_url=base_url,
            api_key=os.environ.get(args.api_key_env),
        )

    if bool(args.embedding_base_url) != bool(args.embedding_model):
        raise SystemExit("--embedding-base-url and --embedding-model must be supplied together")
    embedding_provider: EmbeddingProvider | None = None
    if args.embedding_base_url:
        require_private_context_route(
            args.embedding_base_url,
            allow_remote_private_context=args.allow_remote_private_context,
        )
        embedding_provider = OpenAICompatibleEmbeddingProvider(
            base_url=args.embedding_base_url,
            model=args.embedding_model,
            api_key=os.environ.get("EMBEDDING_API_KEY"),
        )

    kakao = load_kakao_archive(args.kakao_archive)
    instagram = load_instagram_export(args.instagram_archive)
    evidence = collect_person_evidence(
        spec.target_person_id,
        identities,
        kakao_conversations=kakao,
        instagram_threads=instagram.threads,
    )

    common = dict(
        evidence=evidence,
        self_person_id=spec.self_person_id,
        language_model=language_model,
        limit=args.limit,
        raw_response_examples=args.raw_rag_responses,
        embedding_provider=embedding_provider,
        embedding_weight=args.embedding_weight,
    )
    if args.sources == "both":
        kakao_summary = run_generation_replay(source_mode="kakao", **common)
        fused_summary = run_generation_replay(source_mode="fused", **common)
        result: dict[str, object] = {
            "twin_mode": spec.mode.value,
            "provider": args.provider,
            "kakao": kakao_summary.to_dict(),
            "fused": fused_summary.to_dict(),
            "fused_minus_kakao": compare_source_summaries(kakao_summary, fused_summary),
        }
    else:
        summary = run_generation_replay(source_mode=args.sources, **common)
        result = summary.to_dict()
        result["twin_mode"] = spec.mode.value
        result["provider"] = args.provider

    result["privacy"] = (
        "Aggregate metrics only are printed. Remote private-context transmission is "
        "disabled unless explicitly enabled. Raw RAG responses are disabled by default."
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
