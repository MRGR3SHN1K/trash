from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.multiprocessing as mp

from evaluate_dual_encoder_with_llm import (
    SAFETENSORS_AVAILABLE,
    CATALOG_EMBEDDING_FILE_NAME,
    CATALOG_MANIFEST_FILE_NAME,
    build_catalog_manifest,
    build_catalog_feature_frequency,
    build_model_kwargs_from_checkpoint_metadata,
    build_unique_model_output_names,
    encode_catalog_embeddings,
    estimate_product_vectorization_cost,
    iter_source_parquet_files,
    iter_catalog_products,
    is_cuda_out_of_memory,
    load_metadata_payload,
    load_model_state_dict_from_checkpoint,
    manifest_matches,
    resolve_catalog_vector_dir,
    resolve_device,
    resolve_artifact_reference,
    save_feature_frequency,
    safetensors_load_file,
)
from product_card_encoder import (
    BgeM3ProductCardEncoder,
    build_bge_m3_product_card_encoder_from_tokenizer_json,
)

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


class ProductEncoderVectorizationModel:
    def __init__(self, product_encoder: BgeM3ProductCardEncoder) -> None:
        self.product_encoder = product_encoder
        projection = getattr(product_encoder, "retrieval_projection", None)
        self.embedding_dim = int(projection.out_features) if projection is not None else int(product_encoder.d_model)


def get_torchrun_context() -> dict[str, int | bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = world_size > 1
    return {
        "enabled": enabled,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "is_main_process": rank == 0,
    }


def initialize_torchrun_context(context: dict[str, int | bool], device: torch.device) -> None:
    if not bool(context["enabled"]):
        return
    if not torch.distributed.is_available():
        raise RuntimeError("torchrun vectorization requires torch.distributed.")
    if not torch.distributed.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        torch.distributed.init_process_group(backend=backend)


def distributed_barrier_if_needed(context: dict[str, int | bool]) -> None:
    if bool(context["enabled"]) and torch.distributed.is_initialized():
        torch.distributed.barrier()


def distributed_broadcast_object(value: Any, *, context: dict[str, int | bool], src: int = 0) -> Any:
    if not bool(context["enabled"]):
        return value
    payload = [value if int(context["rank"]) == int(src) else None]
    torch.distributed.broadcast_object_list(payload, src=int(src))
    return payload[0]


def resolve_torchrun_device(args: argparse.Namespace, context: dict[str, int | bool]) -> tuple[torch.device, int]:
    if str(args.device).lower() == "cpu":
        return torch.device("cpu"), -1
    if not torch.cuda.is_available():
        raise RuntimeError("torchrun vectorization requested a CUDA device, but CUDA is not available.")

    world_size = int(context["world_size"])
    local_rank = int(context["local_rank"])
    if args.num_gpus is not None and int(args.num_gpus) != world_size:
        raise ValueError(
            f"torchrun already controls process count: WORLD_SIZE={world_size}, but --num-gpus={args.num_gpus}. "
            "Use either torchrun --nproc_per_node=N or python ... --num-gpus N, not both with different values."
        )

    explicit_gpu_ids = parse_gpu_id_list(args.gpu_ids)
    if explicit_gpu_ids:
        if len(explicit_gpu_ids) < world_size:
            raise ValueError(
                f"torchrun WORLD_SIZE={world_size}, but --gpu-ids contains only {len(explicit_gpu_ids)} ids."
            )
        gpu_id = int(explicit_gpu_ids[local_rank])
    else:
        gpu_id = local_rank

    available = torch.cuda.device_count()
    if gpu_id >= available:
        raise RuntimeError(f"torchrun local rank {local_rank} maps to cuda:{gpu_id}, but only {available} CUDA devices are visible.")
    torch.cuda.set_device(gpu_id)
    return torch.device(f"cuda:{gpu_id}"), gpu_id


def parse_gpu_id_list(raw_value: str | None) -> list[int]:
    if raw_value is None or not raw_value.strip():
        return []
    gpu_ids: list[int] = []
    for part in raw_value.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        gpu_ids.append(int(stripped))
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"GPU ids must be unique, got {gpu_ids}")
    if any(gpu_id < 0 for gpu_id in gpu_ids):
        raise ValueError(f"GPU ids must be non-negative, got {gpu_ids}")
    return gpu_ids


def resolve_vectorization_gpu_ids(args: argparse.Namespace) -> list[int]:
    explicit_gpu_ids = parse_gpu_id_list(args.gpu_ids)
    if explicit_gpu_ids:
        requested_count = int(args.num_gpus) if args.num_gpus is not None else len(explicit_gpu_ids)
        if requested_count <= 0:
            raise ValueError("--num-gpus must be positive when --gpu-ids is set.")
        if requested_count > len(explicit_gpu_ids):
            raise ValueError(
                f"--num-gpus={requested_count} is larger than --gpu-ids count={len(explicit_gpu_ids)}."
            )
        gpu_ids = explicit_gpu_ids[:requested_count]
    else:
        requested_count = int(args.num_gpus) if args.num_gpus is not None else 1
        if requested_count <= 0:
            raise ValueError("--num-gpus must be positive.")
        gpu_ids = list(range(requested_count))

    if len(gpu_ids) > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Multi-GPU vectorization requires CUDA.")
        available = torch.cuda.device_count()
        missing = [gpu_id for gpu_id in gpu_ids if gpu_id >= available]
        if missing:
            raise RuntimeError(f"Requested GPU ids {missing}, but torch sees only {available} CUDA devices.")
        if args.device not in (None, "auto", "cuda") and not str(args.device).startswith("cuda"):
            raise ValueError("--device must be auto/cuda/cuda:N when --num-gpus > 1.")
    return gpu_ids


def split_catalog_ranges(catalog_size: int, world_size: int) -> list[tuple[int, int]]:
    if catalog_size < 0:
        raise ValueError("catalog_size must be non-negative.")
    if world_size <= 0:
        raise ValueError("world_size must be positive.")
    ranges: list[tuple[int, int]] = []
    start = 0
    base = catalog_size // world_size
    remainder = catalog_size % world_size
    for rank in range(world_size):
        count = base + (1 if rank < remainder else 0)
        end = start + count
        ranges.append((start, end))
        start = end
    return ranges


def iter_catalog_products_in_range(
    files: Sequence[Path],
    feature_frequency: dict[str, int],
    *,
    max_features: int,
    start_index: int,
    end_index: int,
):
    for catalog_index, product in iter_catalog_products(
        files,
        feature_frequency,
        max_features=max_features,
    ):
        if catalog_index < start_index:
            continue
        if catalog_index >= end_index:
            break
        yield catalog_index, product


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


def _extract_product_encoder_state_dict(model_state_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
    product_state: dict[str, torch.Tensor] = {}
    prefix = "product_encoder."
    for key, value in model_state_dict.items():
        normalized_key = _strip_distributed_state_prefix(str(key))
        if normalized_key.startswith(prefix) and torch.is_tensor(value):
            product_state[normalized_key[len(prefix) :]] = value
    return product_state


def load_product_encoder_for_vectorization(
    checkpoint_source: str,
    *,
    tokenizer_path_override: str | None,
    device: torch.device,
    cache_dir: Path,
) -> tuple[ProductEncoderVectorizationModel, dict[str, Any], str]:
    checkpoint_payload, metadata_local_path, metadata_source = load_metadata_payload(checkpoint_source, cache_dir)
    tokenizer_path, model_kwargs = build_model_kwargs_from_checkpoint_metadata(
        checkpoint_payload,
        tokenizer_path_override=tokenizer_path_override,
    )
    product_encoder = build_bge_m3_product_card_encoder_from_tokenizer_json(
        tokenizer_path,
        d_model=int(model_kwargs["product_d_model"]),
        num_heads=int(model_kwargs["product_num_heads"]),
        num_layers=int(model_kwargs["product_num_layers"]),
        token_type_router_layers=int(model_kwargs["product_token_type_router_layers"]),
        ff_multiplier=int(model_kwargs["product_ff_multiplier"]),
        max_position_embeddings=int(model_kwargs["product_max_position_embeddings"]),
        max_features=int(model_kwargs["product_max_features"]),
        local_attention_window=int(model_kwargs["product_local_attention_window"]),
        max_routing_clusters=int(model_kwargs["product_max_routing_clusters"]),
        dropout=float(model_kwargs["product_dropout"]),
        retrieval_projection_dim=int(model_kwargs["embedding_dim"]),
    )

    product_state = _load_state_artifact(
        checkpoint_payload=checkpoint_payload,
        path_key="product_encoder_state_path",
        format_key="product_encoder_state_format",
        metadata_local_path=metadata_local_path,
        metadata_source=metadata_source,
        cache_dir=cache_dir,
    )
    state_source = "product_encoder"
    if product_state is None:
        full_state = load_model_state_dict_from_checkpoint(
            checkpoint_payload,
            metadata_local_path=metadata_local_path,
            metadata_source=metadata_source,
            cache_dir=cache_dir,
        )
        product_state = _extract_product_encoder_state_dict(full_state)
        state_source = "full_model"
    if not product_state:
        raise FileNotFoundError("Product encoder weights were not found in checkpoint metadata or full model state.")

    product_encoder.load_state_dict(product_state, strict=True)
    product_encoder.to(device)
    product_encoder.eval()
    checkpoint_payload["_vectorizer_loaded_state_source"] = state_source
    return ProductEncoderVectorizationModel(product_encoder), checkpoint_payload, tokenizer_path


def encode_catalog_embedding_shard(
    model: ProductEncoderVectorizationModel,
    *,
    tokenizer_path: str,
    files: Sequence[Path],
    feature_frequency: dict[str, int],
    shard_path: Path,
    shard_metadata_path: Path,
    start_index: int,
    end_index: int,
    args: argparse.Namespace,
    device: torch.device,
    rank: int,
    gpu_id: int,
) -> None:
    shard_size = max(0, int(end_index) - int(start_index))
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    memmap = np.lib.format.open_memmap(
        shard_path,
        mode="w+",
        dtype=np.float16,
        shape=(shard_size, model.embedding_dim),
    )
    use_amp = device.type == "cuda"
    batch_products: list[dict[str, Any]] = []
    batch_indices: list[int] = []
    batch_cost = 0
    encoded_count = 0
    dynamic_batching = bool(getattr(args, "dynamic_catalog_batching", True))
    max_batch_size = max(1, int(getattr(args, "catalog_batch_size", 8)))
    token_budget_arg = getattr(args, "catalog_token_budget", None)
    token_budget = int(token_budget_arg) if token_budget_arg else max_batch_size * int(args.product_max_length)
    chars_per_token = float(getattr(args, "catalog_chars_per_token", 4.0))
    progress = (
        tqdm(
            total=shard_size,
            desc=f"Encode catalog gpu{gpu_id}",
            unit="item",
            dynamic_ncols=True,
            position=rank,
            leave=True,
        )
        if tqdm
        else None
    )

    def encode_batch(products: Sequence[dict[str, Any]], indices: Sequence[int]) -> None:
        nonlocal encoded_count
        if not products:
            return
        try:
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    output = model.product_encoder.encode_cards_from_tokenizer_json(
                        cards=list(products),
                        tokenizer_path=tokenizer_path,
                        max_length=args.product_max_length,
                        device=device,
                    )
                embeddings = output.normalized_embedding.detach().float().cpu().numpy().astype(np.float16, copy=False)
            for row_offset, catalog_index in enumerate(indices):
                local_index = int(catalog_index) - int(start_index)
                memmap[local_index] = embeddings[row_offset]
            encoded_count += len(products)
        except RuntimeError as error:
            if len(products) <= 1 or not is_cuda_out_of_memory(error):
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
            midpoint = max(1, len(products) // 2)
            if progress is not None:
                progress.write(
                    f"GPU {gpu_id}: CUDA OOM while encoding batch_size={len(products)}; "
                    f"retrying as {midpoint}+{len(products) - midpoint}"
                )
            encode_batch(products[:midpoint], indices[:midpoint])
            encode_batch(products[midpoint:], indices[midpoint:])
            return
        if progress is not None:
            progress.update(len(products))

    def flush_batch() -> None:
        nonlocal batch_products, batch_indices, batch_cost
        if not batch_products:
            return
        encode_batch(batch_products, batch_indices)
        batch_products = []
        batch_indices = []
        batch_cost = 0

    for catalog_index, product in iter_catalog_products_in_range(
        files,
        feature_frequency,
        max_features=args.product_max_features,
        start_index=start_index,
        end_index=end_index,
    ):
        product_cost = estimate_product_vectorization_cost(
            product,
            max_length=args.product_max_length,
            chars_per_token=chars_per_token,
        )
        if dynamic_batching and batch_products and (
            len(batch_products) >= max_batch_size or batch_cost + product_cost > token_budget
        ):
            flush_batch()
        batch_products.append(product)
        batch_indices.append(catalog_index)
        batch_cost += product_cost
        if not dynamic_batching and len(batch_products) >= max_batch_size:
            flush_batch()
    flush_batch()
    if progress is not None:
        progress.close()

    del memmap
    if encoded_count != shard_size:
        raise RuntimeError(
            f"GPU {gpu_id} encoded {encoded_count} catalog items, expected {shard_size} "
            f"for range [{start_index}, {end_index})."
        )
    shard_metadata_path.write_text(
        json.dumps(
            {
                "rank": int(rank),
                "gpu_id": int(gpu_id),
                "start_index": int(start_index),
                "end_index": int(end_index),
                "encoded_count": int(encoded_count),
                "embedding_dim": int(model.embedding_dim),
                "shard_path": str(shard_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _multi_gpu_catalog_worker(
    rank: int,
    world_size: int,
    gpu_ids: Sequence[int],
    ranges: Sequence[tuple[int, int]],
    checkpoint_path: str,
    tokenizer_path_override: str | None,
    files: Sequence[str],
    feature_frequency: dict[str, int],
    output_dir: str,
    cache_dir: str,
    args_dict: dict[str, Any],
) -> None:
    if rank >= int(world_size):
        raise ValueError(f"Worker rank={rank} is outside world_size={world_size}.")
    gpu_id = int(gpu_ids[rank])
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    start_index, end_index = ranges[rank]
    args = argparse.Namespace(**args_dict)
    worker_files = [Path(path) for path in files]
    worker_output_dir = Path(output_dir)
    shard_dir = worker_output_dir / "_catalog_embedding_shards"
    shard_path = shard_dir / f"catalog_embeddings.rank{rank:02d}.float16.npy"
    shard_metadata_path = shard_dir / f"catalog_embeddings.rank{rank:02d}.json"

    if end_index <= start_index:
        shard_dir.mkdir(parents=True, exist_ok=True)
        empty = np.lib.format.open_memmap(
            shard_path,
            mode="w+",
            dtype=np.float16,
            shape=(0, int(args.embedding_dim)),
        )
        del empty
        shard_metadata_path.write_text(
            json.dumps(
                {
                    "rank": int(rank),
                    "gpu_id": int(gpu_id),
                    "start_index": int(start_index),
                    "end_index": int(end_index),
                    "encoded_count": 0,
                    "embedding_dim": int(args.embedding_dim),
                    "shard_path": str(shard_path),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return

    worker_cache_dir = Path(cache_dir) / f"rank_{rank:02d}"
    worker_cache_dir.mkdir(parents=True, exist_ok=True)
    model, _checkpoint_payload, tokenizer_path = load_product_encoder_for_vectorization(
        checkpoint_path,
        tokenizer_path_override=tokenizer_path_override,
        device=device,
        cache_dir=worker_cache_dir,
    )
    encode_catalog_embedding_shard(
        model,
        tokenizer_path=tokenizer_path,
        files=worker_files,
        feature_frequency=feature_frequency,
        shard_path=shard_path,
        shard_metadata_path=shard_metadata_path,
        start_index=start_index,
        end_index=end_index,
        args=args,
        device=device,
        rank=rank,
        gpu_id=gpu_id,
    )
    del model
    torch.cuda.empty_cache()


def merge_catalog_embedding_shards(
    *,
    output_dir: Path,
    embedding_dim: int,
    catalog_size: int,
    ranges: Sequence[tuple[int, int]],
    merge_chunk_size: int,
    keep_shards: bool,
) -> Path:
    embedding_path = output_dir / CATALOG_EMBEDDING_FILE_NAME
    shard_dir = output_dir / "_catalog_embedding_shards"
    output_dir.mkdir(parents=True, exist_ok=True)
    merged = np.lib.format.open_memmap(
        embedding_path,
        mode="w+",
        dtype=np.float16,
        shape=(catalog_size, embedding_dim),
    )
    progress = (
        tqdm(total=catalog_size, desc="Merge catalog shards", unit="item", dynamic_ncols=True)
        if tqdm
        else None
    )
    rows_per_copy = max(1, int(merge_chunk_size))
    for rank, (start_index, end_index) in enumerate(ranges):
        shard_path = shard_dir / f"catalog_embeddings.rank{rank:02d}.float16.npy"
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing catalog embedding shard: {shard_path}")
        shard = np.load(shard_path, mmap_mode="r")
        expected_rows = int(end_index) - int(start_index)
        if tuple(shard.shape) != (expected_rows, embedding_dim):
            raise ValueError(
                f"Invalid shard shape for {shard_path}: got {tuple(shard.shape)}, "
                f"expected {(expected_rows, embedding_dim)}."
            )
        for offset in range(0, expected_rows, rows_per_copy):
            row_end = min(offset + rows_per_copy, expected_rows)
            merged[start_index + offset : start_index + row_end] = shard[offset:row_end]
            if progress is not None:
                progress.update(row_end - offset)
        del shard
    if progress is not None:
        progress.close()
    del merged

    if not keep_shards:
        for rank in range(len(ranges)):
            shard_path = shard_dir / f"catalog_embeddings.rank{rank:02d}.float16.npy"
            shard_metadata_path = shard_dir / f"catalog_embeddings.rank{rank:02d}.json"
            if shard_path.exists():
                shard_path.unlink()
            if shard_metadata_path.exists():
                shard_metadata_path.unlink()
        try:
            shard_dir.rmdir()
        except OSError:
            pass
    return embedding_path


def encode_catalog_embeddings_multi_gpu(
    *,
    checkpoint_path: str,
    tokenizer_path_override: str | None,
    tokenizer_path: str,
    files: Sequence[Path],
    feature_frequency: dict[str, int],
    catalog_size: int,
    embedding_dim: int,
    output_dir: Path,
    args: argparse.Namespace,
    gpu_ids: Sequence[int],
    cache_dir: Path,
) -> Path:
    embedding_path = output_dir / CATALOG_EMBEDDING_FILE_NAME
    manifest_path = output_dir / CATALOG_MANIFEST_FILE_NAME
    expected_manifest = build_catalog_manifest(
        files=files,
        embedding_dim=embedding_dim,
        catalog_size=catalog_size,
        args=args,
    )
    if args.reuse_catalog_cache and embedding_path.exists() and manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_matches(existing_manifest, expected_manifest):
            return embedding_path

    output_dir.mkdir(parents=True, exist_ok=True)
    ranges = split_catalog_ranges(catalog_size, len(gpu_ids))
    worker_args = dict(vars(args))
    worker_args["embedding_dim"] = int(embedding_dim)
    print(
        f"Multi-GPU catalog vectorization: workers={len(gpu_ids)}, gpu_ids={list(gpu_ids)}, "
        f"catalog_size={catalog_size}",
        flush=True,
    )
    mp.spawn(
        _multi_gpu_catalog_worker,
        nprocs=len(gpu_ids),
        join=True,
        args=(
            len(gpu_ids),
            list(gpu_ids),
            ranges,
            checkpoint_path,
            tokenizer_path_override,
            [str(path) for path in files],
            feature_frequency,
            str(output_dir),
            str(cache_dir),
            worker_args,
        ),
    )

    embedding_path = merge_catalog_embedding_shards(
        output_dir=output_dir,
        embedding_dim=embedding_dim,
        catalog_size=catalog_size,
        ranges=ranges,
        merge_chunk_size=args.multi_gpu_merge_chunk_size,
        keep_shards=args.keep_catalog_shards,
    )
    manifest_path.write_text(json.dumps(expected_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return embedding_path


def encode_catalog_embeddings_torchrun(
    *,
    checkpoint_path: str,
    tokenizer_path_override: str | None,
    tokenizer_path: str,
    files: Sequence[Path],
    feature_frequency: dict[str, int],
    catalog_size: int,
    embedding_dim: int,
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
    gpu_id: int,
    cache_dir: Path,
    context: dict[str, int | bool],
) -> tuple[Path, str]:
    rank = int(context["rank"])
    world_size = int(context["world_size"])
    is_main_process = bool(context["is_main_process"])
    embedding_path = output_dir / CATALOG_EMBEDDING_FILE_NAME
    manifest_path = output_dir / CATALOG_MANIFEST_FILE_NAME
    expected_manifest = build_catalog_manifest(
        files=files,
        embedding_dim=embedding_dim,
        catalog_size=catalog_size,
        args=args,
    )

    should_reuse = False
    if is_main_process and args.reuse_catalog_cache and embedding_path.exists() and manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        should_reuse = manifest_matches(existing_manifest, expected_manifest)
    should_reuse = bool(distributed_broadcast_object(should_reuse, context=context))
    if should_reuse:
        return embedding_path, "cached"

    if is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"torchrun catalog vectorization: world_size={world_size}, catalog_size={catalog_size}",
            flush=True,
        )
    distributed_barrier_if_needed(context)

    ranges = split_catalog_ranges(catalog_size, world_size)
    start_index, end_index = ranges[rank]
    shard_dir = output_dir / "_catalog_embedding_shards"
    shard_path = shard_dir / f"catalog_embeddings.rank{rank:02d}.float16.npy"
    shard_metadata_path = shard_dir / f"catalog_embeddings.rank{rank:02d}.json"
    worker_cache_dir = cache_dir / f"rank_{rank:02d}"
    worker_cache_dir.mkdir(parents=True, exist_ok=True)

    if end_index <= start_index:
        shard_dir.mkdir(parents=True, exist_ok=True)
        empty = np.lib.format.open_memmap(
            shard_path,
            mode="w+",
            dtype=np.float16,
            shape=(0, embedding_dim),
        )
        del empty
        shard_metadata_path.write_text(
            json.dumps(
                {
                    "rank": rank,
                    "gpu_id": int(gpu_id),
                    "start_index": int(start_index),
                    "end_index": int(end_index),
                    "encoded_count": 0,
                    "embedding_dim": int(embedding_dim),
                    "shard_path": str(shard_path),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        loaded_state_source = "empty_shard"
    else:
        model, checkpoint_payload, loaded_tokenizer_path = load_product_encoder_for_vectorization(
            checkpoint_path,
            tokenizer_path_override=tokenizer_path_override,
            device=device,
            cache_dir=worker_cache_dir,
        )
        loaded_state_source = str(checkpoint_payload.get("_vectorizer_loaded_state_source") or "unknown")
        if str(loaded_tokenizer_path) != str(tokenizer_path):
            raise ValueError(
                f"Tokenizer path mismatch on rank {rank}: expected {tokenizer_path}, got {loaded_tokenizer_path}."
            )
        encode_catalog_embedding_shard(
            model,
            tokenizer_path=tokenizer_path,
            files=files,
            feature_frequency=feature_frequency,
            shard_path=shard_path,
            shard_metadata_path=shard_metadata_path,
            start_index=start_index,
            end_index=end_index,
            args=args,
            device=device,
            rank=rank,
            gpu_id=gpu_id,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    distributed_barrier_if_needed(context)
    if is_main_process:
        embedding_path = merge_catalog_embedding_shards(
            output_dir=output_dir,
            embedding_dim=embedding_dim,
            catalog_size=catalog_size,
            ranges=ranges,
            merge_chunk_size=args.multi_gpu_merge_chunk_size,
            keep_shards=args.keep_catalog_shards,
        )
        manifest_path.write_text(json.dumps(expected_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    distributed_barrier_if_needed(context)
    return embedding_path, loaded_state_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute full product-catalog embeddings for one or more dual-encoder checkpoints."
    )
    parser.add_argument("--data-dir", required=True, help="Directory containing data0001.parquet ... data0029.parquet")
    parser.add_argument("--checkpoint-path", nargs="+", required=True, help="One or more checkpoint directories or checkpoint metadata/full-model paths. Product encoder sidecar weights are loaded automatically when present.")
    parser.add_argument("--tokenizer-path", default=None, help="Optional tokenizer override. By default it is restored from checkpoint metadata.")
    parser.add_argument("--catalog-vector-root", "--output-dir", dest="catalog_vector_root", default=None, help="Optional root for vectors. If omitted, vectors are saved to <checkpoint_dir>/catalog_vectors.")
    parser.add_argument("--device", default="auto", help="Device string, for example cuda, cuda:0 or cpu")
    parser.add_argument("--num-gpus", type=int, default=None, help="Number of CUDA GPUs to use for catalog vectorization. Use 8 for eight A100 workers.")
    parser.add_argument("--gpu-ids", default=None, help="Comma-separated CUDA device ids for multi-GPU vectorization, for example 0,1,2,3,4,5,6,7.")
    parser.add_argument("--catalog-batch-size", type=int, default=64, help="Maximum catalog product batch size. With dynamic batching this is an item cap, not the only memory control.")
    parser.add_argument("--dynamic-catalog-batching", action=argparse.BooleanOptionalAction, default=True, help="Pack catalog encoding batches by approximate token cost and split OOM batches automatically.")
    parser.add_argument("--catalog-token-budget", type=int, default=8192, help="Approximate total product tokens per catalog encoding batch when dynamic batching is enabled.")
    parser.add_argument("--catalog-chars-per-token", type=float, default=4.0, help="Approximate characters-per-token ratio used before tokenizer execution.")
    parser.add_argument("--multi-gpu-merge-chunk-size", type=int, default=65536, help="Rows copied at once while merging multi-GPU catalog shards.")
    parser.add_argument("--keep-catalog-shards", action=argparse.BooleanOptionalAction, default=False, help="Keep temporary per-GPU catalog embedding shards after merging.")
    parser.add_argument("--product-max-features", type=int, default=14, help="How many product features to pass into the product encoder")
    parser.add_argument("--product-max-length", type=int, default=1024, help="Maximum product sequence length for the product encoder")
    parser.add_argument("--reuse-catalog-cache", action=argparse.BooleanOptionalAction, default=True, help="Reuse existing vectors when manifest matches")
    parser.add_argument("--cache-dir", default=None, help="Optional cache directory for downloaded checkpoints")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torchrun_context = get_torchrun_context()
    try:
        torchrun_enabled = bool(torchrun_context["enabled"])
        is_main_process = bool(torchrun_context["is_main_process"])
        if torchrun_enabled:
            device, torchrun_gpu_id = resolve_torchrun_device(args, torchrun_context)
            initialize_torchrun_context(torchrun_context, device)
            gpu_ids: list[int] = []
            if device.type == "cuda":
                gpu_ids = parse_gpu_id_list(args.gpu_ids) or list(range(int(torchrun_context["world_size"])))
        else:
            gpu_ids = resolve_vectorization_gpu_ids(args)
            explicit_gpu_selection = bool(args.gpu_ids) or args.num_gpus is not None
            if len(gpu_ids) == 1 and explicit_gpu_selection:
                if not torch.cuda.is_available():
                    raise RuntimeError("Explicit GPU vectorization was requested, but CUDA is not available.")
                device = torch.device(f"cuda:{gpu_ids[0]}")
            else:
                device = resolve_device(args.device)
            torchrun_gpu_id = int(gpu_ids[0]) if device.type == "cuda" and gpu_ids else -1

        checkpoint_paths = [str(path) for path in args.checkpoint_path]
        model_output_names = build_unique_model_output_names(checkpoint_paths)
        catalog_vector_root = Path(args.catalog_vector_root) if args.catalog_vector_root else None

        if is_main_process:
            files = iter_source_parquet_files(Path(args.data_dir))
            print(f"Found {len(files)} source parquet files in {args.data_dir}", flush=True)
            feature_frequency, catalog_size = build_catalog_feature_frequency(files)
            print(
                f"Catalog scan complete: catalog_size={catalog_size}, unique_features={len(feature_frequency)}",
                flush=True,
            )
            catalog_payload = {
                "files": [str(path) for path in files],
                "feature_frequency": feature_frequency,
                "catalog_size": int(catalog_size),
            }
        else:
            catalog_payload = None
        catalog_payload = distributed_broadcast_object(catalog_payload, context=torchrun_context)
        files = [Path(path) for path in catalog_payload["files"]]
        feature_frequency = dict(catalog_payload["feature_frequency"])
        catalog_size = int(catalog_payload["catalog_size"])

        used_vector_dirs: dict[Path, str] = {}
        overview: list[dict[str, object]] = []
        for checkpoint_path in checkpoint_paths:
            model_output_name = model_output_names[checkpoint_path]
            vector_dir = resolve_catalog_vector_dir(
                checkpoint_path,
                model_output_name=model_output_name,
                catalog_vector_root=catalog_vector_root,
            )
            resolved_vector_dir = vector_dir.resolve()
            previous_checkpoint = used_vector_dirs.get(resolved_vector_dir)
            if previous_checkpoint is not None and previous_checkpoint != checkpoint_path:
                raise ValueError(
                    f"Two checkpoints map to the same catalog vector directory: {resolved_vector_dir}. "
                    "Use --catalog-vector-root to create per-checkpoint vector subdirectories."
                )
            used_vector_dirs[resolved_vector_dir] = checkpoint_path

            if is_main_process:
                vector_dir.mkdir(parents=True, exist_ok=True)
            distributed_barrier_if_needed(torchrun_context)
            model_cache_dir = Path(args.cache_dir) / model_output_name if args.cache_dir else vector_dir / "cache"
            model_cache_dir.mkdir(parents=True, exist_ok=True)

            checkpoint_payload, _metadata_local_path, _metadata_source = load_metadata_payload(
                checkpoint_path,
                model_cache_dir,
            )
            tokenizer_path, model_kwargs = build_model_kwargs_from_checkpoint_metadata(
                checkpoint_payload,
                tokenizer_path_override=args.tokenizer_path,
            )
            embedding_dim = int(model_kwargs["embedding_dim"])
            encoder_max_features = int(model_kwargs["product_max_features"])
            effective_product_max_features = min(int(args.product_max_features), encoder_max_features)
            if effective_product_max_features <= 0:
                raise ValueError("--product-max-features must be positive.")
            if is_main_process and effective_product_max_features != int(args.product_max_features):
                print(
                    f"Requested product_max_features={args.product_max_features}, "
                    f"but checkpoint product encoder supports {encoder_max_features}; "
                    f"using {effective_product_max_features}.",
                    flush=True,
                )

            model_args = argparse.Namespace(**vars(args))
            model_args.checkpoint_path = checkpoint_path
            model_args.tokenizer_path = tokenizer_path
            model_args.product_max_features = effective_product_max_features

            if is_main_process:
                save_feature_frequency(vector_dir, feature_frequency)
            distributed_barrier_if_needed(torchrun_context)

            if torchrun_enabled:
                embedding_path, loaded_state_source = encode_catalog_embeddings_torchrun(
                    checkpoint_path=checkpoint_path,
                    tokenizer_path_override=args.tokenizer_path,
                    tokenizer_path=tokenizer_path,
                    files=files,
                    feature_frequency=feature_frequency,
                    catalog_size=catalog_size,
                    embedding_dim=embedding_dim,
                    output_dir=vector_dir,
                    args=model_args,
                    device=device,
                    gpu_id=torchrun_gpu_id,
                    cache_dir=model_cache_dir,
                    context=torchrun_context,
                )
            elif len(gpu_ids) > 1:
                embedding_path = encode_catalog_embeddings_multi_gpu(
                    checkpoint_path=checkpoint_path,
                    tokenizer_path_override=args.tokenizer_path,
                    tokenizer_path=tokenizer_path,
                    files=files,
                    feature_frequency=feature_frequency,
                    catalog_size=catalog_size,
                    embedding_dim=embedding_dim,
                    output_dir=vector_dir,
                    args=model_args,
                    gpu_ids=gpu_ids,
                    cache_dir=model_cache_dir,
                )
                loaded_state_source = "product_encoder" if checkpoint_payload.get("product_encoder_state_path") else "full_model"
            else:
                print(f"Loading product encoder: {checkpoint_path}", flush=True)
                model, checkpoint_payload, tokenizer_path = load_product_encoder_for_vectorization(
                    checkpoint_path,
                    tokenizer_path_override=args.tokenizer_path,
                    device=device,
                    cache_dir=model_cache_dir,
                )
                embedding_dim = int(model.embedding_dim)
                loaded_state_source = checkpoint_payload.get("_vectorizer_loaded_state_source")
                embedding_path = encode_catalog_embeddings(
                    model,
                    tokenizer_path=tokenizer_path,
                    files=files,
                    feature_frequency=feature_frequency,
                    catalog_size=catalog_size,
                    output_dir=vector_dir,
                    args=model_args,
                    device=device,
                )
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            if not is_main_process:
                continue

            metadata = {
                "checkpoint_path": checkpoint_path,
                "tokenizer_path": tokenizer_path,
                "catalog_size": int(catalog_size),
                "catalog_vector_dir": str(vector_dir),
                "catalog_embedding_path": str(embedding_path),
                "device": str(device),
                "num_gpus": int(torchrun_context["world_size"]) if torchrun_enabled else (int(len(gpu_ids)) if len(gpu_ids) > 1 else 1),
                "gpu_ids": [int(gpu_id) for gpu_id in gpu_ids] if (torchrun_enabled or len(gpu_ids) > 1) else [],
                "product_max_features": int(effective_product_max_features),
                "product_max_length": int(args.product_max_length),
                "catalog_batch_size": int(args.catalog_batch_size),
                "dynamic_catalog_batching": bool(args.dynamic_catalog_batching),
                "catalog_token_budget": int(args.catalog_token_budget),
                "catalog_chars_per_token": float(args.catalog_chars_per_token),
                "safetensors_available": SAFETENSORS_AVAILABLE,
                "loaded_state_source": loaded_state_source,
                "product_encoder_state_path": checkpoint_payload.get("product_encoder_state_path"),
                "product_encoder_state_format": checkpoint_payload.get("product_encoder_state_format"),
                "checkpoint_metadata_epoch": int(checkpoint_payload.get("epoch", 0)),
                "checkpoint_metadata_global_step": int(checkpoint_payload.get("global_step", 0)),
                "launch_mode": "torchrun" if torchrun_enabled else ("mp_spawn" if len(gpu_ids) > 1 else "single_process"),
            }
            (vector_dir / "catalog_vectorization_metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            overview.append(metadata)
            print(f"Saved catalog vectors: {embedding_path}", flush=True)

        if is_main_process and catalog_vector_root is not None:
            catalog_vector_root.mkdir(parents=True, exist_ok=True)
            (catalog_vector_root / "catalog_vectorization_overview.json").write_text(
                json.dumps(overview, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    finally:
        if bool(torchrun_context["enabled"]) and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
