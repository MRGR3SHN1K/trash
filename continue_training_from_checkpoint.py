from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler

from dual_encoder_retrieval import DualEncoderBuildSpec, ProductQueryDualEncoder, build_product_query_dual_encoder_from_tokenizer_json
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from train_dual_encoder_on_parquet import (
    DistributedContext,
    DistributedDynamicBatchSampler,
    QueryProductPairDataset,
    TrainingMonitor,
    cleanup_distributed,
    collate_training_examples,
    compute_in_batch_retrieval_metrics,
    compute_token_type_auxiliary_loss,
    configure_model_for_training,
    distributed_barrier,
    evaluate,
    flatten_grouped_queries,
    main_print,
    prepare_or_load_corpus,
    resolve_target_batch_cost,
    save_checkpoint,
    setup_distributed,
)

try:
    from safetensors.torch import load_file as safetensors_load_file

    SAFETENSORS_AVAILABLE = True
except ImportError:
    safetensors_load_file = None
    SAFETENSORS_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load an existing dual-encoder checkpoint and continue training for additional epochs."
    )
    parser.add_argument("--checkpoint-path", required=True, help="Path to last_checkpoint.pt or best_checkpoint.pt")
    parser.add_argument("--additional-epochs", type=int, default=1, help="How many extra epochs to train from the loaded checkpoint")
    parser.add_argument("--output-dir", "--save-dir", dest="output_dir", default=None, help="Directory for new checkpoints, logs and metrics. Defaults to <checkpoint_dir>/continued_training")
    parser.add_argument("--data-dir", default=None, help="Optional override for parquet data directory")
    parser.add_argument("--tokenizer-path", default=None, help="Optional override for tokenizer json path")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional override for max examples per device")
    parser.add_argument("--positive-queries-per-product", type=int, default=None, help="Optional override for positive SKU queries sampled for each product")
    parser.add_argument("--learning-rate", type=float, default=None, help="Optional override for AdamW learning rate")
    parser.add_argument("--weight-decay", type=float, default=None, help="Optional override for AdamW weight decay")
    parser.add_argument("--grad-clip-norm", type=float, default=None, help="Optional override for gradient clipping norm")
    parser.add_argument("--num-workers", type=int, default=None, help="Optional override for dataloader workers")
    parser.add_argument("--max-train-steps", type=int, default=None, help="Optional hard cap on optimizer steps during the continued run")
    parser.add_argument("--train-ratio", type=float, default=None, help="Optional override if train/valid splits need to be rebuilt")
    parser.add_argument("--reuse-prepared-splits", action=argparse.BooleanOptionalAction, default=None, help="Reuse prepared train/valid/frequency parquet files when present")
    parser.add_argument("--symmetric-loss", action=argparse.BooleanOptionalAction, default=None, help="Optional override for symmetric product<->query multi-positive contrastive loss")
    parser.add_argument("--fsdp", action=argparse.BooleanOptionalAction, default=None, help="Optional override for FSDP")
    parser.add_argument("--fsdp-sync-module-states", action=argparse.BooleanOptionalAction, default=None, help="Optional override for syncing module weights from rank 0 during FSDP init.")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=None, help="Optional override for gradient checkpointing")
    parser.add_argument("--save-optimizer-state", action=argparse.BooleanOptionalAction, default=None, help="Optional override for saving optimizer state in new checkpoints")
    parser.add_argument("--export-safetensors", action=argparse.BooleanOptionalAction, default=None, help="Optional override for saving model weights as .safetensors when available")
    parser.add_argument("--resume-optimizer", action=argparse.BooleanOptionalAction, default=True, help="Restore optimizer state if the checkpoint contains it and the current runtime can load it safely")
    parser.add_argument("--log-every", type=int, default=None, help="Optional override for CSV/TensorBoard logging frequency")
    parser.add_argument("--plot-every", type=int, default=None, help="Optional override for live loss plot refresh frequency")
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=None, help="Optional override for TensorBoard logging")
    parser.add_argument("--token-type-loss-weight", type=float, default=None, help="Optional override for auxiliary token-type loss weight")
    parser.add_argument("--dynamic-batching", action=argparse.BooleanOptionalAction, default=None, help="Optional override for dynamic batching")
    parser.add_argument("--dynamic-cost-per-device", type=int, default=None, help="Optional override for per-device dynamic batch cost")
    parser.add_argument("--dynamic-cost-headroom", type=float, default=None, help="Optional override for auto-derived dynamic batch headroom")
    parser.add_argument("--dynamic-bucket-multiplier", type=int, default=None, help="Optional override for dynamic batching bucket size")
    parser.add_argument("--dynamic-chars-per-token", type=float, default=None, help="Optional override for char->token length heuristic")
    parser.add_argument("--query-token-type-router-layers", type=int, default=None, help="Optional override for query token type/router classifier layers")
    parser.add_argument("--product-token-type-router-layers", type=int, default=None, help="Optional override for product token type/router classifier layers")
    parser.add_argument("--product-local-attention-window", type=int, default=None, help="Optional override for product encoder local sparse attention radius")
    parser.add_argument("--product-max-routing-clusters", type=int, default=None, help="Optional override for max product routing centroids")
    parser.add_argument("--distributed-backend", default="auto", help="Distributed backend: auto, nccl or gloo")
    parser.add_argument("--distributed-timeout-minutes", type=int, default=None, help="Optional override for distributed collective timeout in minutes.")
    parser.add_argument("--device", default="auto", help="Device string, for example cuda, cpu, cuda:0")
    parser.add_argument("--seed", type=int, default=None, help="Optional override for random seed")
    return parser.parse_args()


def default_training_args() -> dict[str, Any]:
    return {
        "data_dir": None,
        "tokenizer_path": "tokenizer (1).json",
        "output_dir": None,
        "train_ratio": 0.8,
        "seed": 42,
        "reuse_prepared_splits": True,
        "batch_size": 32,
        "positive_queries_per_product": 5,
        "dynamic_batching": True,
        "dynamic_cost_per_device": None,
        "dynamic_cost_headroom": 0.9,
        "dynamic_bucket_multiplier": 32,
        "dynamic_chars_per_token": 4.0,
        "epochs": 1,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "grad_clip_norm": 1.0,
        "max_train_steps": None,
        "num_workers": 0,
        "device": "auto",
        "prepare_only": False,
        "symmetric_loss": True,
        "fsdp": True,
        "fsdp_sync_module_states": True,
        "distributed_backend": "auto",
        "distributed_timeout_minutes": 60,
        "gradient_checkpointing": True,
        "save_optimizer_state": True,
        "export_safetensors": True,
        "log_every": 10,
        "plot_every": 50,
        "tensorboard": True,
        "token_type_loss_weight": 0.1,
        "query_max_length": 256,
        "product_max_length": 1024,
        "embedding_dim": 1024,
        "temperature": 0.05,
        "query_target_parameters": 3_000_000_000,
        "query_num_layers": 30,
        "query_token_type_router_layers": 4,
        "query_num_heads": 16,
        "query_ff_multiplier": 4,
        "query_max_position_embeddings": 2048,
        "query_dropout": 0.1,
        "product_d_model": 1024,
        "product_num_heads": 16,
        "product_num_layers": 30,
        "product_token_type_router_layers": 4,
        "product_ff_multiplier": 4,
        "product_max_position_embeddings": 8194,
        "product_max_features": 14,
        "product_local_attention_window": 128,
        "product_max_routing_clusters": 128,
        "product_dropout": 0.1,
    }


def load_checkpoint_metadata(
    checkpoint_path: Path,
    distributed_context: DistributedContext,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    checkpoint_payload: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None

    if distributed_context.use_fsdp and distributed_context.enabled:
        if distributed_context.is_main_process:
            checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
            metadata = {
                "epoch": int(checkpoint_payload.get("epoch", 0)),
                "global_step": int(checkpoint_payload.get("global_step", 0)),
                "metrics": checkpoint_payload.get("metrics", {}) or {},
                "args": checkpoint_payload.get("args", {}) or {},
                "build_spec": checkpoint_payload.get("build_spec", {}) or {},
                "model_state_format": checkpoint_payload.get("model_state_format"),
                "model_state_path": checkpoint_payload.get("model_state_path"),
                "optimizer_state_format": checkpoint_payload.get("optimizer_state_format"),
                "optimizer_state_path": checkpoint_payload.get("optimizer_state_path"),
                "has_optimizer_state": bool(
                    checkpoint_payload.get("has_optimizer_state")
                    or ("optimizer_state_dict" in checkpoint_payload)
                    or checkpoint_payload.get("optimizer_state_path")
                ),
            }
        object_list = [metadata]
        dist.broadcast_object_list(object_list, src=0)
        metadata = object_list[0]
    else:
        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
        metadata = {
            "epoch": int(checkpoint_payload.get("epoch", 0)),
            "global_step": int(checkpoint_payload.get("global_step", 0)),
            "metrics": checkpoint_payload.get("metrics", {}) or {},
            "args": checkpoint_payload.get("args", {}) or {},
            "build_spec": checkpoint_payload.get("build_spec", {}) or {},
            "model_state_format": checkpoint_payload.get("model_state_format"),
            "model_state_path": checkpoint_payload.get("model_state_path"),
            "optimizer_state_format": checkpoint_payload.get("optimizer_state_format"),
            "optimizer_state_path": checkpoint_payload.get("optimizer_state_path"),
            "has_optimizer_state": bool(
                checkpoint_payload.get("has_optimizer_state")
                or ("optimizer_state_dict" in checkpoint_payload)
                or checkpoint_payload.get("optimizer_state_path")
            ),
        }

    if metadata is None:
        raise RuntimeError("Failed to load checkpoint metadata.")
    return metadata, checkpoint_payload


def build_resume_args(
    cli_args: argparse.Namespace,
    metadata: dict[str, Any],
    checkpoint_path: Path,
) -> argparse.Namespace:
    merged = default_training_args()
    checkpoint_args = dict(metadata.get("args", {}) or {})
    build_spec = dict(metadata.get("build_spec", {}) or {})
    query_model_size = dict(build_spec.get("query_model_size", {}) or {})

    merged.update(checkpoint_args)

    merged["tokenizer_path"] = checkpoint_args.get("tokenizer_path") or build_spec.get("tokenizer_path") or merged["tokenizer_path"]
    merged["embedding_dim"] = int(build_spec.get("embedding_dim", merged["embedding_dim"]))
    merged["query_num_layers"] = int(query_model_size.get("num_layers", merged["query_num_layers"]))
    merged["query_token_type_router_layers"] = int(query_model_size.get("token_type_router_layers", merged["query_token_type_router_layers"]))
    merged["query_num_heads"] = int(query_model_size.get("num_heads", merged["query_num_heads"]))
    merged["query_ff_multiplier"] = int(query_model_size.get("ff_multiplier", merged["query_ff_multiplier"]))
    merged["query_max_position_embeddings"] = int(query_model_size.get("max_position_embeddings", merged["query_max_position_embeddings"]))
    merged["query_target_parameters"] = int(query_model_size.get("estimated_parameters", merged["query_target_parameters"]))
    merged["product_d_model"] = int(build_spec.get("product_d_model", merged["product_d_model"]))
    merged["product_num_heads"] = int(build_spec.get("product_num_heads", merged["product_num_heads"]))
    merged["product_num_layers"] = int(build_spec.get("product_num_layers", merged["product_num_layers"]))
    merged["product_token_type_router_layers"] = int(build_spec.get("product_token_type_router_layers", merged["product_token_type_router_layers"]))
    merged["product_max_position_embeddings"] = int(build_spec.get("product_max_position_embeddings", merged["product_max_position_embeddings"]))
    merged["product_max_features"] = int(build_spec.get("product_max_features", merged["product_max_features"]))
    merged["product_local_attention_window"] = int(build_spec.get("product_local_attention_window", merged["product_local_attention_window"]))
    merged["product_max_routing_clusters"] = int(build_spec.get("product_max_routing_clusters", merged["product_max_routing_clusters"]))

    override_keys = [
        "data_dir",
        "tokenizer_path",
        "output_dir",
        "batch_size",
        "positive_queries_per_product",
        "learning_rate",
        "weight_decay",
        "grad_clip_norm",
        "num_workers",
        "max_train_steps",
        "train_ratio",
        "reuse_prepared_splits",
        "symmetric_loss",
        "fsdp",
        "fsdp_sync_module_states",
        "gradient_checkpointing",
        "save_optimizer_state",
        "export_safetensors",
        "log_every",
        "plot_every",
        "tensorboard",
        "token_type_loss_weight",
        "dynamic_batching",
        "dynamic_cost_per_device",
        "dynamic_cost_headroom",
        "dynamic_bucket_multiplier",
        "dynamic_chars_per_token",
        "query_token_type_router_layers",
        "product_token_type_router_layers",
        "product_local_attention_window",
        "product_max_routing_clusters",
        "distributed_timeout_minutes",
        "seed",
    ]
    for key in override_keys:
        value = getattr(cli_args, key, None)
        if value is not None:
            merged[key] = value

    merged["device"] = cli_args.device
    merged["distributed_backend"] = cli_args.distributed_backend
    merged["prepare_only"] = False
    merged["epochs"] = int(cli_args.additional_epochs)

    if not merged.get("data_dir"):
        raise ValueError("data_dir is missing. Pass --data-dir or make sure it was saved in the checkpoint args.")

    if merged.get("output_dir") is None:
        merged["output_dir"] = str(checkpoint_path.resolve().parent / "continued_training")

    return argparse.Namespace(**merged)


def build_dataloaders(
    args: argparse.Namespace,
    *,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    distributed_context: DistributedContext,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = QueryProductPairDataset(
        train_df,
        query_max_length=args.query_max_length,
        product_max_length=args.product_max_length,
        positive_queries_per_product=args.positive_queries_per_product,
        chars_per_token=args.dynamic_chars_per_token,
    )
    valid_dataset = QueryProductPairDataset(
        valid_df,
        query_max_length=args.query_max_length,
        product_max_length=args.product_max_length,
        positive_queries_per_product=args.positive_queries_per_product,
        chars_per_token=args.dynamic_chars_per_token,
        allow_empty=True,
    )

    if args.dynamic_batching:
        train_target_batch_cost = resolve_target_batch_cost(
            example_costs=train_dataset.example_costs,
            max_batch_size=args.batch_size,
            override_cost=args.dynamic_cost_per_device,
            headroom=args.dynamic_cost_headroom,
        )
        valid_target_batch_cost = resolve_target_batch_cost(
            example_costs=valid_dataset.example_costs,
            max_batch_size=args.batch_size,
            override_cost=args.dynamic_cost_per_device,
            headroom=args.dynamic_cost_headroom,
        )
        train_batch_sampler = DistributedDynamicBatchSampler(
            train_dataset.example_costs,
            max_batch_size=args.batch_size,
            target_batch_cost=train_target_batch_cost,
            rank=distributed_context.rank,
            world_size=distributed_context.world_size,
            shuffle=True,
            drop_last=False,
            seed=args.seed,
            bucket_size_multiplier=args.dynamic_bucket_multiplier,
        )
        valid_batch_sampler = DistributedDynamicBatchSampler(
            valid_dataset.example_costs,
            max_batch_size=args.batch_size,
            target_batch_cost=valid_target_batch_cost,
            rank=distributed_context.rank,
            world_size=distributed_context.world_size,
            shuffle=False,
            drop_last=False,
            seed=args.seed,
            bucket_size_multiplier=args.dynamic_bucket_multiplier,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=args.num_workers,
            collate_fn=collate_training_examples,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_sampler=valid_batch_sampler,
            num_workers=args.num_workers,
            collate_fn=collate_training_examples,
        )
    else:
        train_sampler = (
            DistributedSampler(train_dataset, num_replicas=distributed_context.world_size, rank=distributed_context.rank, shuffle=True)
            if distributed_context.enabled
            else None
        )
        valid_sampler = (
            DistributedSampler(valid_dataset, num_replicas=distributed_context.world_size, rank=distributed_context.rank, shuffle=False)
            if distributed_context.enabled
            else None
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=args.num_workers,
            collate_fn=collate_training_examples,
            drop_last=False,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=valid_sampler,
            num_workers=args.num_workers,
            collate_fn=collate_training_examples,
            drop_last=False,
        )
    return train_loader, valid_loader


def build_model_from_resume_args(args: argparse.Namespace) -> tuple[ProductQueryDualEncoder, DualEncoderBuildSpec]:
    return build_product_query_dual_encoder_from_tokenizer_json(
        args.tokenizer_path,
        embedding_dim=args.embedding_dim,
        query_target_parameters=args.query_target_parameters,
        query_num_layers=args.query_num_layers,
        query_token_type_router_layers=args.query_token_type_router_layers,
        query_num_heads=args.query_num_heads,
        query_ff_multiplier=args.query_ff_multiplier,
        query_max_position_embeddings=args.query_max_position_embeddings,
        query_dropout=args.query_dropout,
        product_d_model=args.product_d_model,
        product_num_heads=args.product_num_heads,
        product_num_layers=args.product_num_layers,
        product_token_type_router_layers=args.product_token_type_router_layers,
        product_ff_multiplier=args.product_ff_multiplier,
        product_max_position_embeddings=args.product_max_position_embeddings,
        product_max_features=args.product_max_features,
        product_local_attention_window=args.product_local_attention_window,
        product_max_routing_clusters=args.product_max_routing_clusters,
        product_dropout=args.product_dropout,
        temperature=args.temperature,
    )


def resolve_checkpoint_artifact_path(
    checkpoint_path: Path,
    artifact_reference: str | None,
) -> Path | None:
    if not artifact_reference:
        return None
    candidate = Path(artifact_reference)
    if candidate.is_absolute():
        return candidate
    return checkpoint_path.parent / candidate


def load_model_state_dict_from_checkpoint(
    *,
    checkpoint_payload: dict[str, Any] | None,
    checkpoint_path: Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    model_state_path = resolve_checkpoint_artifact_path(
        checkpoint_path,
        str(metadata.get("model_state_path") or ""),
    )
    model_state_format = str(metadata.get("model_state_format") or "")

    if model_state_path is not None and model_state_path.exists():
        if model_state_format == "safetensors":
            if not SAFETENSORS_AVAILABLE:
                raise RuntimeError(
                    "Checkpoint weights were saved as .safetensors, but the safetensors package is not installed."
                )
            return safetensors_load_file(str(model_state_path), device="cpu")
        return torch.load(model_state_path, map_location="cpu")

    if checkpoint_payload is None:
        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in checkpoint_payload:
        return checkpoint_payload["model_state_dict"]
    raise FileNotFoundError(f"Model weights not found for checkpoint {checkpoint_path}")


def load_optimizer_state_from_checkpoint(
    *,
    checkpoint_payload: dict[str, Any] | None,
    checkpoint_path: Path,
    metadata: dict[str, Any],
) -> dict[str, Any] | None:
    optimizer_state_path = resolve_checkpoint_artifact_path(
        checkpoint_path,
        str(metadata.get("optimizer_state_path") or ""),
    )
    if optimizer_state_path is not None and optimizer_state_path.exists():
        return torch.load(optimizer_state_path, map_location="cpu")

    if checkpoint_payload is None:
        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
    return checkpoint_payload.get("optimizer_state_dict")


def maybe_load_model_weights(
    model: ProductQueryDualEncoder,
    *,
    checkpoint_payload: dict[str, Any] | None,
    checkpoint_path: Path,
    distributed_context: DistributedContext,
    sync_module_states: bool,
    metadata: dict[str, Any],
) -> None:
    if distributed_context.use_fsdp and distributed_context.enabled and sync_module_states:
        if distributed_context.is_main_process:
            model_state_dict = load_model_state_dict_from_checkpoint(
                checkpoint_payload=checkpoint_payload,
                checkpoint_path=checkpoint_path,
                metadata=metadata,
            )
            model.load_state_dict(model_state_dict, strict=True)
        return

    model_state_dict = load_model_state_dict_from_checkpoint(
        checkpoint_payload=checkpoint_payload,
        checkpoint_path=checkpoint_path,
        metadata=metadata,
    )
    model.load_state_dict(model_state_dict, strict=True)


def maybe_restore_optimizer_state(
    optimizer: torch.optim.Optimizer,
    *,
    model: nn.Module,
    checkpoint_payload: dict[str, Any] | None,
    checkpoint_path: Path,
    metadata: dict[str, Any],
    distributed_context: DistributedContext,
    resume_optimizer: bool,
) -> bool:
    if not resume_optimizer:
        return False

    if distributed_context.use_fsdp and distributed_context.enabled:
        has_optimizer_state = [bool(metadata.get("has_optimizer_state", False))]
        if dist.is_initialized():
            dist.broadcast_object_list(has_optimizer_state, src=0)
        if not has_optimizer_state[0]:
            return False
        optimizer_state_dict = (
            load_optimizer_state_from_checkpoint(
                checkpoint_payload=checkpoint_payload,
                checkpoint_path=checkpoint_path,
                metadata=metadata,
            )
            if distributed_context.is_main_process
            else None
        )
        full_optimizer_state = optimizer_state_dict if distributed_context.is_main_process else None
        sharded_optimizer_state = FSDP.scatter_full_optim_state_dict(
            full_optimizer_state,
            model,
            optim=optimizer,
        )
        optimizer.load_state_dict(sharded_optimizer_state)
        return True

    optimizer_state_dict = load_optimizer_state_from_checkpoint(
        checkpoint_payload=checkpoint_payload,
        checkpoint_path=checkpoint_path,
        metadata=metadata,
    )
    if optimizer_state_dict is None:
        return False

    optimizer.load_state_dict(optimizer_state_dict)
    return True


def resume_train(
    *,
    args: argparse.Namespace,
    model: nn.Module,
    build_spec: DualEncoderBuildSpec,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    distributed_context: DistributedContext,
    model_description: dict[str, int | float],
    checkpoint_payload: dict[str, Any] | None,
    checkpoint_path: Path,
    metadata: dict[str, Any],
    resume_optimizer: bool,
) -> None:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    optimizer_restored = maybe_restore_optimizer_state(
        optimizer,
        model=model,
        checkpoint_payload=checkpoint_payload,
        checkpoint_path=checkpoint_path,
        metadata=metadata,
        distributed_context=distributed_context,
        resume_optimizer=resume_optimizer,
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=False)
    resumed_epoch = int(metadata.get("epoch", 0))
    start_epoch = resumed_epoch + 1
    final_epoch = resumed_epoch + args.epochs
    global_step = int(metadata.get("global_step", 0))
    best_valid_loss = float((metadata.get("metrics", {}) or {}).get("valid_loss", math.inf))

    metrics_log: list[dict[str, float | int]] = []
    train_epoch_controller = None
    for candidate in (getattr(train_loader, "batch_sampler", None), getattr(train_loader, "sampler", None)):
        if hasattr(candidate, "set_epoch"):
            train_epoch_controller = candidate
            break

    total_planned_steps = len(train_loader) * args.epochs
    if args.max_train_steps is not None:
        total_planned_steps = min(total_planned_steps, args.max_train_steps)
    monitor = TrainingMonitor(
        output_dir=output_dir,
        total_steps=total_planned_steps,
        total_epochs=args.epochs,
        enabled=distributed_context.is_main_process,
        log_every=args.log_every,
        plot_every=args.plot_every,
        use_tensorboard=args.tensorboard,
    )

    try:
        for epoch in range(start_epoch, final_epoch + 1):
            model.train()
            if train_epoch_controller is not None:
                train_epoch_controller.set_epoch(epoch)
            wall_started = time.perf_counter()
            running_loss = 0.0
            running_batches = 0

            for grouped_queries, products in train_loader:
                _flat_queries, query_group_ids = flatten_grouped_queries(grouped_queries)
                query_group_ids = query_group_ids.to(device=device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    output = model(
                        queries=_flat_queries,
                        products=products,
                        tokenizer_path=args.tokenizer_path,
                        query_max_length=args.query_max_length,
                        product_max_length=args.product_max_length,
                        device=device,
                        compute_loss=False,
                        symmetric_loss=args.symmetric_loss,
                        compute_token_type_loss=args.token_type_loss_weight > 0,
                    )
                    _similarity, retrieval_loss, _labels = compute_in_batch_retrieval_metrics(
                        model,
                        query_embeddings=output.query_output.normalized_embedding,
                        product_embeddings=output.product_output.normalized_embedding,
                        query_group_ids=query_group_ids,
                        distributed_context=distributed_context,
                        symmetric_loss=args.symmetric_loss,
                    )
                    auxiliary_loss, token_type_loss_scalar = compute_token_type_auxiliary_loss(
                        output,
                        weight=args.token_type_loss_weight,
                    )
                    loss = retrieval_loss if auxiliary_loss is None else retrieval_loss + auxiliary_loss

                scaler.scale(loss).backward()
                if args.grad_clip_norm is not None and args.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()

                global_step += 1
                step_loss = float(loss.detach())
                running_loss += step_loss
                running_batches += 1
                monitor.update_step(
                    global_step=global_step,
                    epoch=epoch - resumed_epoch,
                    step_loss=step_loss,
                    avg_epoch_loss=running_loss / max(running_batches, 1),
                    retrieval_loss=float(retrieval_loss.detach().item()),
                    token_type_loss=token_type_loss_scalar if args.token_type_loss_weight > 0 else None,
                )

                if args.max_train_steps is not None and global_step >= args.max_train_steps:
                    break

            if distributed_context.enabled:
                train_totals = torch.tensor(
                    [running_loss, float(running_batches)],
                    device=device,
                    dtype=torch.float64,
                )
                dist.all_reduce(train_totals, op=dist.ReduceOp.SUM)
                running_loss = float(train_totals[0].item())
                running_batches = int(train_totals[1].item())

            train_loss = running_loss / max(running_batches, 1)
            valid_metrics = evaluate(
                model,
                valid_loader,
                tokenizer_path=args.tokenizer_path,
                query_max_length=args.query_max_length,
                product_max_length=args.product_max_length,
                device=device,
                symmetric_loss=args.symmetric_loss,
                distributed_context=distributed_context,
                token_type_loss_weight=args.token_type_loss_weight,
            )
            epoch_metrics = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": train_loss,
                "valid_loss": valid_metrics["loss"],
                "valid_accuracy_at_1": valid_metrics["accuracy_at_1"],
                "epoch_seconds": time.perf_counter() - wall_started,
            }
            metrics_log.append(epoch_metrics)
            monitor.log_epoch(
                epoch=epoch - resumed_epoch,
                global_step=global_step,
                train_loss=train_loss,
                valid_loss=valid_metrics["loss"],
                valid_accuracy_at_1=valid_metrics["accuracy_at_1"],
                epoch_seconds=epoch_metrics["epoch_seconds"],
            )

            main_print(
                distributed_context,
                f"Resume epoch {epoch}: "
                f"train_loss={train_loss:.6f}, "
                f"valid_loss={valid_metrics['loss']:.6f}, "
                f"valid_accuracy@1={valid_metrics['accuracy_at_1']:.4f}, "
                f"steps={global_step}",
            )

            save_checkpoint(
                output_dir / "last_checkpoint.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                build_spec=build_spec,
                metrics=epoch_metrics,
                args=args,
                distributed_context=distributed_context,
                save_optimizer_state=args.save_optimizer_state,
                export_safetensors=args.export_safetensors,
            )
            if valid_metrics["loss"] < best_valid_loss:
                best_valid_loss = valid_metrics["loss"]
                save_checkpoint(
                    output_dir / "best_checkpoint.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    build_spec=build_spec,
                    metrics=epoch_metrics,
                    args=args,
                    distributed_context=distributed_context,
                    save_optimizer_state=args.save_optimizer_state,
                    export_safetensors=args.export_safetensors,
                )

            if args.max_train_steps is not None and global_step >= args.max_train_steps:
                break
    finally:
        monitor.close()

    if distributed_context.is_main_process:
        pd.DataFrame(metrics_log).to_parquet(output_dir / "continued_training_metrics.parquet", index=False)
        with (output_dir / "resume_run_metadata.json").open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "resumed_from_checkpoint": str(checkpoint_path),
                    "resumed_from_epoch": resumed_epoch,
                    "starting_global_step": int(metadata.get("global_step", 0)),
                    "optimizer_restored": optimizer_restored,
                    "build_spec": asdict(build_spec),
                    "model_description": model_description,
                    "args": vars(args),
                },
                file,
                ensure_ascii=False,
                indent=2,
            )


def main() -> None:
    cli_args = parse_args()
    bootstrap_args = argparse.Namespace(
        distributed_backend=cli_args.distributed_backend,
        distributed_timeout_minutes=60 if cli_args.distributed_timeout_minutes is None else cli_args.distributed_timeout_minutes,
        device=cli_args.device,
        fsdp=True if cli_args.fsdp is None else cli_args.fsdp,
    )

    distributed_context = setup_distributed(bootstrap_args)
    checkpoint_path = Path(cli_args.checkpoint_path)
    try:
        metadata, checkpoint_payload = load_checkpoint_metadata(checkpoint_path, distributed_context)
        args = build_resume_args(cli_args, metadata, checkpoint_path)

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        train_df, valid_df, _frequency_df = prepare_or_load_corpus(args, distributed_context)
        train_loader, valid_loader = build_dataloaders(
            args,
            train_df=train_df,
            valid_df=valid_df,
            distributed_context=distributed_context,
        )

        output_dir = Path(args.output_dir)
        if distributed_context.is_main_process:
            output_dir.mkdir(parents=True, exist_ok=True)
        distributed_barrier(distributed_context)

        model, build_spec = build_model_from_resume_args(args)
        maybe_load_model_weights(
            model,
            checkpoint_payload=checkpoint_payload,
            checkpoint_path=checkpoint_path,
            distributed_context=distributed_context,
            sync_module_states=bool(args.fsdp_sync_module_states),
            metadata=metadata,
        )
        model = configure_model_for_training(
            model,
            args=args,
            distributed_context=distributed_context,
        )
        model_description = model.module.describe() if hasattr(model, "module") else model.describe()

        main_print(distributed_context, f"Using device: {distributed_context.device}")
        if distributed_context.use_fsdp:
            main_print(distributed_context, f"FSDP enabled across {distributed_context.world_size} GPUs")
        main_print(
            distributed_context,
            f"Resuming from checkpoint: {checkpoint_path} "
            f"(epoch={metadata.get('epoch', 0)}, global_step={metadata.get('global_step', 0)})",
        )
        if args.export_safetensors and not SAFETENSORS_AVAILABLE:
            main_print(
                distributed_context,
                "safetensors package is unavailable in this environment; resumed checkpoints will save weights as external .model.pt files instead.",
            )
        main_print(distributed_context, f"Saving continued training artifacts to: {output_dir}")
        if args.dynamic_batching:
            target_cost = resolve_target_batch_cost(
                example_costs=train_loader.dataset.example_costs,
                max_batch_size=args.batch_size,
                override_cost=args.dynamic_cost_per_device,
                headroom=args.dynamic_cost_headroom,
            )
            main_print(
                distributed_context,
                f"Dynamic batching enabled: max_examples_per_device={args.batch_size}, "
                f"target_cost_per_device={target_cost}, "
                f"chars_per_token={args.dynamic_chars_per_token}",
            )
        main_print(distributed_context, f"Model description: {model_description}")
        main_print(distributed_context, f"Build spec: {asdict(build_spec)}")

        resume_train(
            args=args,
            model=model,
            build_spec=build_spec,
            train_loader=train_loader,
            valid_loader=valid_loader,
            device=distributed_context.device,
            output_dir=output_dir,
            distributed_context=distributed_context,
            model_description=model_description,
            checkpoint_payload=checkpoint_payload,
            checkpoint_path=checkpoint_path,
            metadata=metadata,
            resume_optimizer=cli_args.resume_optimizer,
        )
    finally:
        cleanup_distributed(distributed_context)


if __name__ == "__main__":
    main()
