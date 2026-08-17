from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from evaluate_dual_encoder_with_llm import (
    SAFETENSORS_AVAILABLE,
    CATALOG_EMBEDDING_FILE_NAME,
    build_model_kwargs_from_checkpoint_metadata,
    build_unique_model_output_names,
    iter_source_parquet_files,
    load_feature_frequency_from_vector_dir,
    load_metadata_payload,
    load_model_state_dict_from_checkpoint,
    materialize_candidates_by_index,
    require_catalog_embedding_path,
    resolve_artifact_reference,
    resolve_catalog_vector_dir,
    resolve_device,
    safetensors_load_file,
)
from hybrid_dual_path_model import HybridDualPathTransformer, build_model_from_tokenizer_json


def resolve_search_catalog_vector_dir(
    checkpoint_path: str,
    *,
    model_output_name: str,
    catalog_vector_root: Path | None,
) -> Path:
    if catalog_vector_root is None:
        return resolve_catalog_vector_dir(
            checkpoint_path,
            model_output_name=model_output_name,
            catalog_vector_root=None,
        )
    if (catalog_vector_root / CATALOG_EMBEDDING_FILE_NAME).exists():
        return catalog_vector_root
    return resolve_catalog_vector_dir(
        checkpoint_path,
        model_output_name=model_output_name,
        catalog_vector_root=catalog_vector_root,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive online search over precomputed product catalog embeddings."
    )
    parser.add_argument("--data-dir", required=True, help="Directory containing data0001.parquet ... data0029.parquet")
    parser.add_argument("--checkpoint-path", required=True, help="Checkpoint directory, last_checkpoint.pt, .model.* or query_encoder sidecar path")
    parser.add_argument("--tokenizer-path", default=None, help="Optional tokenizer override. By default it is restored from checkpoint metadata.")
    parser.add_argument("--catalog-vector-root", "--vector-dir", dest="catalog_vector_root", default=None, help="Root with precomputed catalog vectors. If omitted, vectors are read from <checkpoint_dir>/catalog_vectors.")
    parser.add_argument("--device", default="auto", help="Device string, for example cuda, cuda:0 or cpu")
    parser.add_argument("--top-k", type=int, default=10, help="How many products to return per query.")
    parser.add_argument("--query-max-length", type=int, default=256, help="Maximum query sequence length for the query encoder.")
    parser.add_argument("--product-max-features", type=int, default=14, help="Must match vectorized catalog max features unless --allow-stale-catalog-vectors is set.")
    parser.add_argument("--product-max-length", type=int, default=1024, help="Must match vectorized catalog product max length unless --allow-stale-catalog-vectors is set.")
    parser.add_argument("--similarity-chunk-size", type=int, default=262144, help="Catalog embedding rows scored per chunk.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature used only for displayed top-k softmax probabilities.")
    parser.add_argument("--allow-stale-catalog-vectors", action="store_true", help="Use catalog vectors even if manifest does not match current metadata.")
    parser.add_argument("--cache-dir", default=None, help="Optional cache directory for downloaded checkpoints.")
    parser.add_argument("--once", default=None, help="Run one query and exit instead of interactive REPL.")
    parser.add_argument("--output-jsonl", default=None, help="Optional JSONL file with all query results.")
    parser.add_argument("--output-csv", default=None, help="Optional CSV file with flattened query results.")
    parser.add_argument("--show-features", action=argparse.BooleanOptionalAction, default=False, help="Print selected product features in console output.")
    return parser.parse_args()


def _load_state_artifact(
    *,
    checkpoint_payload: dict[str, Any],
    path_key: str,
    format_key: str,
    metadata_local_path: Path,
    metadata_source: str | Path,
    cache_dir: Path,
) -> dict[str, Any] | None:
    state_path = resolve_artifact_reference(
        checkpoint_payload.get(path_key),
        metadata_source=metadata_source,
        metadata_local_path=metadata_local_path,
        cache_dir=cache_dir,
    )
    if state_path is None or not state_path.exists():
        return None
    state_format = str(checkpoint_payload.get(format_key) or "")
    if state_format == "safetensors":
        if not SAFETENSORS_AVAILABLE:
            raise RuntimeError(
                f"Weights are stored in {state_path.name}, but the safetensors package is not installed."
            )
        return safetensors_load_file(str(state_path), device="cpu")
    return torch.load(state_path, map_location="cpu")


def _strip_distributed_state_prefix(key: str) -> str:
    for prefix in ("_fsdp_wrapped_module.", "module."):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _extract_query_encoder_state_dict(model_state_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
    query_state: dict[str, torch.Tensor] = {}
    prefix = "query_encoder."
    for key, value in model_state_dict.items():
        normalized_key = _strip_distributed_state_prefix(str(key))
        if normalized_key.startswith(prefix) and torch.is_tensor(value):
            query_state[normalized_key[len(prefix) :]] = value
    return query_state


def build_query_encoder_from_checkpoint_metadata(
    tokenizer_path: str,
    model_kwargs: dict[str, Any],
    checkpoint_payload: dict[str, Any],
) -> HybridDualPathTransformer:
    query_model_size = dict(dict(checkpoint_payload.get("build_spec", {}) or {}).get("query_model_size", {}) or {})
    d_model = query_model_size.get("d_model")
    if d_model is None:
        raise ValueError(
            "Checkpoint metadata does not contain build_spec.query_model_size.d_model. "
            "Use a V4 checkpoint metadata file or run search via full evaluate loader."
        )
    return build_model_from_tokenizer_json(
        tokenizer_path,
        d_model=int(d_model),
        num_heads=int(model_kwargs["query_num_heads"]),
        num_layers=int(model_kwargs["query_num_layers"]),
        token_type_router_layers=int(model_kwargs["query_token_type_router_layers"]),
        ff_multiplier=int(model_kwargs["query_ff_multiplier"]),
        max_position_embeddings=int(model_kwargs["query_max_position_embeddings"]),
        dropout=float(model_kwargs["query_dropout"]),
        causal=False,
        output_logits=False,
        pooling="attention",
        retrieval_projection_dim=int(model_kwargs["embedding_dim"]),
    )


def load_query_encoder_for_search(
    checkpoint_source: str,
    *,
    tokenizer_path_override: str | None,
    device: torch.device,
    cache_dir: Path,
) -> tuple[HybridDualPathTransformer, dict[str, Any], str, int]:
    checkpoint_payload, metadata_local_path, metadata_source = load_metadata_payload(checkpoint_source, cache_dir)
    tokenizer_path, model_kwargs = build_model_kwargs_from_checkpoint_metadata(
        checkpoint_payload,
        tokenizer_path_override=tokenizer_path_override,
    )
    query_encoder = build_query_encoder_from_checkpoint_metadata(tokenizer_path, model_kwargs, checkpoint_payload)

    query_state = _load_state_artifact(
        checkpoint_payload=checkpoint_payload,
        path_key="query_encoder_state_path",
        format_key="query_encoder_state_format",
        metadata_local_path=metadata_local_path,
        metadata_source=metadata_source,
        cache_dir=cache_dir,
    )
    state_source = "query_encoder"
    if query_state is None:
        full_state = load_model_state_dict_from_checkpoint(
            checkpoint_payload,
            metadata_local_path=metadata_local_path,
            metadata_source=metadata_source,
            cache_dir=cache_dir,
        )
        query_state = _extract_query_encoder_state_dict(full_state)
        state_source = "full_model"
    if not query_state:
        raise FileNotFoundError("Query encoder weights were not found in checkpoint metadata or full model state.")

    query_encoder.load_state_dict(query_state, strict=True)
    query_encoder.to(device)
    query_encoder.eval()
    checkpoint_payload["_search_loaded_state_source"] = state_source
    return query_encoder, checkpoint_payload, tokenizer_path, int(model_kwargs["embedding_dim"])


def encode_query(
    query_encoder: HybridDualPathTransformer,
    *,
    query_text: str,
    tokenizer_path: str,
    max_length: int,
    device: torch.device,
) -> np.ndarray:
    use_amp = device.type == "cuda"
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            output = query_encoder.encode_texts_from_tokenizer_json(
                texts=[query_text],
                tokenizer_path=tokenizer_path,
                max_length=max_length,
                device=device,
            )
    return output.normalized_embedding[0].detach().float().cpu().numpy()


def retrieve_top_k(
    *,
    query_embedding: np.ndarray,
    catalog_embeddings: np.ndarray,
    top_k: int,
    chunk_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    top_k = min(int(top_k), int(catalog_embeddings.shape[0]))
    chunk_size = max(1, int(chunk_size))
    query = torch.from_numpy(query_embedding.astype(np.float32, copy=False)).to(device=device)
    best_scores = torch.full((top_k,), -torch.inf, device=device, dtype=torch.float32)
    best_indices = torch.full((top_k,), -1, device=device, dtype=torch.long)

    for catalog_start in range(0, catalog_embeddings.shape[0], chunk_size):
        chunk_np = np.asarray(catalog_embeddings[catalog_start : catalog_start + chunk_size], dtype=np.float32)
        chunk = torch.from_numpy(chunk_np).to(device=device)
        scores = chunk @ query
        chunk_top_scores, chunk_top_local_indices = torch.topk(scores, k=min(top_k, scores.numel()))
        chunk_top_indices = chunk_top_local_indices + catalog_start
        merged_scores = torch.cat((best_scores, chunk_top_scores), dim=0)
        merged_indices = torch.cat((best_indices, chunk_top_indices), dim=0)
        best_scores, merged_positions = torch.topk(merged_scores, k=top_k)
        best_indices = torch.gather(merged_indices, 0, merged_positions)
        if device.type == "cuda":
            del chunk, scores, chunk_top_scores, chunk_top_local_indices, chunk_top_indices, merged_scores, merged_indices
            torch.cuda.empty_cache()

    return best_scores.detach().cpu().numpy(), best_indices.detach().cpu().numpy()


def top_probabilities(scores: np.ndarray, *, temperature: float) -> np.ndarray:
    temperature = max(float(temperature), 1e-6)
    scaled = scores.astype(np.float64) / temperature
    scaled = scaled - np.max(scaled)
    exp_scores = np.exp(scaled)
    return exp_scores / max(float(exp_scores.sum()), 1e-40)


def format_features(features: Sequence[dict[str, Any]], *, max_items: int = 6) -> str:
    parts = []
    for feature in list(features)[:max_items]:
        name = str(feature.get("name") or "").strip()
        value = str(feature.get("value") or "").strip()
        if name or value:
            parts.append(f"{name}: {value}".strip(": "))
    return "; ".join(parts)


def print_results(
    *,
    query_text: str,
    results: Sequence[dict[str, Any]],
    show_features: bool,
) -> None:
    print(f"\nQuery: {query_text}")
    if not results:
        print("No results.")
        return
    for item in results:
        print(
            f"{item['rank']:>2}. score={item['score']:.6f} "
            f"p_top={item['top_probability']:.4f} "
            f"{item['source_file']}:{item['source_row_index']} | {item['title']}"
        )
        if item.get("description"):
            print(f"    description: {item['description'][:300]}")
        if show_features and item.get("features"):
            print(f"    features: {format_features(item['features'])}")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def append_csv(path: Path, *, query_text: str, results: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "query_text",
                "rank",
                "catalog_index",
                "score",
                "top_probability",
                "source_file",
                "source_row_index",
                "title",
                "description",
                "features_json",
            ],
        )
        if not file_exists:
            writer.writeheader()
        for item in results:
            writer.writerow(
                {
                    "query_text": query_text,
                    "rank": item["rank"],
                    "catalog_index": item["catalog_index"],
                    "score": item["score"],
                    "top_probability": item["top_probability"],
                    "source_file": item["source_file"],
                    "source_row_index": item["source_row_index"],
                    "title": item["title"],
                    "description": item["description"],
                    "features_json": json.dumps(item["features"], ensure_ascii=False),
                }
            )


def run_search_once(
    *,
    query_encoder: HybridDualPathTransformer,
    tokenizer_path: str,
    catalog_embeddings: np.ndarray,
    files: Sequence[Path],
    feature_frequency: dict[str, int],
    args: argparse.Namespace,
    device: torch.device,
    query_text: str,
) -> list[dict[str, Any]]:
    query_embedding = encode_query(
        query_encoder,
        query_text=query_text,
        tokenizer_path=tokenizer_path,
        max_length=args.query_max_length,
        device=device,
    )
    scores, indices = retrieve_top_k(
        query_embedding=query_embedding,
        catalog_embeddings=catalog_embeddings,
        top_k=args.top_k,
        chunk_size=args.similarity_chunk_size,
        device=device,
    )
    probs = top_probabilities(scores, temperature=args.temperature)
    needed_indices = {int(index) for index in indices.tolist() if int(index) >= 0}
    candidates = materialize_candidates_by_index(
        files,
        needed_indices,
        feature_frequency,
        max_features=args.product_max_features,
    )
    results: list[dict[str, Any]] = []
    for rank, (score, probability, catalog_index) in enumerate(zip(scores, probs, indices), start=1):
        catalog_index = int(catalog_index)
        candidate = candidates.get(catalog_index)
        results.append(
            {
                "rank": rank,
                "catalog_index": catalog_index,
                "score": float(score),
                "top_probability": float(probability),
                "source_file": candidate.source_file if candidate else "",
                "source_row_index": candidate.source_row_index if candidate else -1,
                "title": candidate.title if candidate else "",
                "description": candidate.description if candidate else "",
                "features": candidate.features if candidate else [],
            }
        )
    return results


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(args.catalog_vector_root or ".") / "interactive_search_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    query_encoder, checkpoint_payload, tokenizer_path, embedding_dim = load_query_encoder_for_search(
        args.checkpoint_path,
        tokenizer_path_override=args.tokenizer_path,
        device=device,
        cache_dir=cache_dir,
    )
    model_output_name = build_unique_model_output_names([args.checkpoint_path])[args.checkpoint_path]
    catalog_vector_root = Path(args.catalog_vector_root) if args.catalog_vector_root else None
    catalog_vector_dir = resolve_search_catalog_vector_dir(
        args.checkpoint_path,
        model_output_name=model_output_name,
        catalog_vector_root=catalog_vector_root,
    )
    files = iter_source_parquet_files(Path(args.data_dir))
    embedding_path, manifest = require_catalog_embedding_path(
        vector_dir=catalog_vector_dir,
        files=files,
        embedding_dim=embedding_dim,
        tokenizer_path=tokenizer_path,
        checkpoint_path=args.checkpoint_path,
        product_max_features=args.product_max_features,
        product_max_length=args.product_max_length,
        allow_stale=args.allow_stale_catalog_vectors,
    )
    feature_frequency = load_feature_frequency_from_vector_dir(catalog_vector_dir)
    if feature_frequency is None:
        raise FileNotFoundError(
            f"Feature frequency file not found in {catalog_vector_dir}. "
            "Run vectorize_dual_encoder_catalog.py first."
        )
    catalog_embeddings = np.load(embedding_path, mmap_mode="r")
    if int(catalog_embeddings.shape[1]) != int(embedding_dim):
        raise ValueError(f"Catalog embedding dim={catalog_embeddings.shape[1]} does not match query dim={embedding_dim}.")

    print(
        f"Loaded search index: catalog_size={catalog_embeddings.shape[0]}, dim={catalog_embeddings.shape[1]}, "
        f"vectors={embedding_path}"
    )
    print(f"Loaded query encoder from {checkpoint_payload.get('_search_loaded_state_source')}. Type 'exit' or empty line to quit.")

    def handle_query(query_text: str) -> None:
        normalized_query = query_text.strip()
        if not normalized_query:
            return
        results = run_search_once(
            query_encoder=query_encoder,
            tokenizer_path=tokenizer_path,
            catalog_embeddings=catalog_embeddings,
            files=files,
            feature_frequency=feature_frequency,
            args=args,
            device=device,
            query_text=normalized_query,
        )
        print_results(query_text=normalized_query, results=results, show_features=args.show_features)
        payload = {
            "query_text": normalized_query,
            "checkpoint_path": args.checkpoint_path,
            "catalog_vector_dir": str(catalog_vector_dir),
            "top_k": int(args.top_k),
            "results": results,
        }
        if args.output_jsonl:
            append_jsonl(Path(args.output_jsonl), payload)
        if args.output_csv:
            append_csv(Path(args.output_csv), query_text=normalized_query, results=results)

    if args.once is not None:
        handle_query(args.once)
        return

    while True:
        try:
            query_text = input("\nquery> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query_text.strip() or query_text.strip().lower() in {"exit", "quit", "q"}:
            break
        handle_query(query_text)


if __name__ == "__main__":
    main()
