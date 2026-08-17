from __future__ import annotations

import argparse
import ast
import bisect
from collections import Counter
import csv
from dataclasses import asdict, dataclass
from datetime import timedelta
from functools import partial
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler

from dual_encoder_retrieval import (
    DualEncoderBuildSpec,
    ProductQueryDualEncoder,
    build_product_query_dual_encoder_from_tokenizer_json,
)
from hybrid_dual_path_model import DualPathTransformerLayer, TokenTypeTransformerLayer

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

try:
    from safetensors.torch import save_file as safetensors_save_file

    SAFETENSORS_AVAILABLE = True
except ImportError:
    safetensors_save_file = None
    SAFETENSORS_AVAILABLE = False


TITLE_COLUMN = "title"
FEATURES_COLUMN = "features"
SKU_COLUMN = "sku"
SKU_ERROR_COLUMN = "sku_error"
DESCRIPTION_COLUMN_CANDIDATES = ("description", "text", "body", "full_text")
DATA_FILE_TEMPLATE = "data{index:04d}.parquet"
DEFAULT_PRODUCT_MAX_FEATURES = 14

NAME_KEYS = {"name", "feature", "feature name", "attribute", "characteristic", "title", "key"}
VALUE_KEYS = {"value", "values", "text", "description", "value name", "value_name", "content"}
FEATURE_NUMERIC_VALUE_RE = re.compile(r"\d")


@dataclass(slots=True)
class FeatureItem:
    name: str
    value: str
    norm_name: str


@dataclass(slots=True)
class TrainingExample:
    queries: list[str]
    product: dict[str, Any]


@dataclass(slots=True)
class DistributedContext:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    use_fsdp: bool

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


class TrainingMonitor:
    def __init__(
        self,
        *,
        output_dir: Path,
        total_steps: int,
        total_epochs: int,
        enabled: bool,
        log_every: int = 10,
        plot_every: int = 50,
        use_tensorboard: bool = True,
    ) -> None:
        self.output_dir = output_dir
        self.enabled = enabled
        self.total_steps = total_steps
        self.total_epochs = total_epochs
        self.log_every = max(1, log_every)
        self.plot_every = max(1, plot_every)
        self.train_history: list[tuple[int, float]] = []
        self.valid_history: list[tuple[int, float, float]] = []
        self.progress = None
        self.csv_path = output_dir / "loss_history.csv"
        self.plot_path = output_dir / "live_loss.png"
        self.tensorboard_dir = output_dir / "tensorboard"
        self._csv_file = None
        self._csv_writer = None
        self._summary_writer = None
        self.tensorboard_enabled = False
        self.plot_enabled = False

        if not self.enabled:
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        self._csv_file = self.csv_path.open("w", encoding="utf-8", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(["kind", "global_step", "epoch", "value", "aux_value"])
        self._csv_file.flush()

        if tqdm is not None:
            self.progress = tqdm(
                total=total_steps,
                desc="Training",
                unit="batch",
                dynamic_ncols=True,
                smoothing=0.1,
            )

        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._summary_writer = SummaryWriter(log_dir=str(self.tensorboard_dir))
                self.tensorboard_enabled = True
            except Exception:
                self._summary_writer = None
                self.tensorboard_enabled = False

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            self._plt = plt
            self.plot_enabled = True
        except Exception:
            self._plt = None
            self.plot_enabled = False

    def _write_csv_row(self, row: list[object]) -> None:
        if self._csv_writer is None or self._csv_file is None:
            return
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def _render_plot(self) -> None:
        if not self.plot_enabled or self._plt is None:
            return
        if not self.train_history and not self.valid_history:
            return

        plt = self._plt
        fig, ax = plt.subplots(figsize=(10, 5))
        if self.train_history:
            train_steps, train_losses = zip(*self.train_history, strict=False)
            ax.plot(train_steps, train_losses, label="train_loss", color="#1f77b4", linewidth=1.5)
        if self.valid_history:
            valid_steps, valid_losses, _valid_acc = zip(*self.valid_history, strict=False)
            ax.plot(valid_steps, valid_losses, label="valid_loss", color="#d62728", linewidth=2.0, marker="o")

        ax.set_title("Dual-Encoder Loss")
        ax.set_xlabel("Global step")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.plot_path)
        plt.close(fig)

    def update_step(
        self,
        *,
        global_step: int,
        epoch: int,
        step_loss: float,
        avg_epoch_loss: float,
        retrieval_loss: float | None = None,
        token_type_loss: float | None = None,
    ) -> None:
        if not self.enabled:
            return

        if self.progress is not None:
            self.progress.update(1)
            postfix: dict[str, str] = {
                "epoch": f"{epoch}/{self.total_epochs}",
                "loss": f"{step_loss:.4f}",
                "avg": f"{avg_epoch_loss:.4f}",
            }
            if retrieval_loss is not None:
                postfix["ret"] = f"{retrieval_loss:.4f}"
            if token_type_loss is not None:
                postfix["type"] = f"{token_type_loss:.4f}"
            self.progress.set_postfix(**postfix)

        if global_step % self.log_every == 0 or global_step == 1:
            self.train_history.append((global_step, step_loss))
            self._write_csv_row(["train", global_step, epoch, step_loss, avg_epoch_loss])
            if self._summary_writer is not None:
                self._summary_writer.add_scalar("train/loss_step", step_loss, global_step)
                self._summary_writer.add_scalar("train/loss_running_avg", avg_epoch_loss, global_step)
                if retrieval_loss is not None:
                    self._summary_writer.add_scalar("train/retrieval_loss_step", retrieval_loss, global_step)
                if token_type_loss is not None:
                    self._summary_writer.add_scalar("train/token_type_loss_step", token_type_loss, global_step)

        if global_step % self.plot_every == 0 or global_step == 1:
            self._render_plot()

    def log_epoch(
        self,
        *,
        epoch: int,
        global_step: int,
        train_loss: float,
        valid_loss: float | None = None,
        valid_accuracy_at_1: float | None = None,
        epoch_seconds: float,
    ) -> None:
        if not self.enabled:
            return

        self._write_csv_row(["epoch_train", global_step, epoch, train_loss, epoch_seconds])
        if valid_loss is not None and valid_accuracy_at_1 is not None:
            self.valid_history.append((global_step, valid_loss, valid_accuracy_at_1))
            self._write_csv_row(["valid", global_step, epoch, valid_loss, valid_accuracy_at_1])

        if self._summary_writer is not None:
            self._summary_writer.add_scalar("train/loss_epoch", train_loss, global_step)
            if valid_loss is not None and valid_accuracy_at_1 is not None:
                self._summary_writer.add_scalar("valid/loss", valid_loss, global_step)
                self._summary_writer.add_scalar("valid/accuracy_at_1", valid_accuracy_at_1, global_step)
            self._summary_writer.add_scalar("train/epoch_seconds", epoch_seconds, global_step)
            self._summary_writer.flush()

        self._render_plot()

    def close(self) -> None:
        if self.progress is not None:
            self.progress.close()
            self.progress = None
        if self._summary_writer is not None:
            self._summary_writer.flush()
            self._summary_writer.close()
            self._summary_writer = None
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None

def normalize_spaces(value: str) -> str:
    import re

    return re.sub(r"\s+", " ", value).strip()


def normalize_feature_name(value: str) -> str:
    import re

    value = value.lower().replace("_", " ").replace("-", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip(" :;,.")


def clean_text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, float) and pd.isna(value):
        return ""
    if isinstance(value, (list, tuple, set)):
        parts = [clean_text_value(item) for item in value]
        parts = [part for part in parts if part]
        return ", ".join(dict.fromkeys(parts))
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            item_text = clean_text_value(item)
            if item_text:
                parts.append(f"{key}: {item_text}")
        return "; ".join(parts)
    return normalize_spaces(str(value))


def try_parse_json_like(raw_value: Any) -> Any:
    if raw_value is None or (isinstance(raw_value, float) and pd.isna(raw_value)):
        return None
    if isinstance(raw_value, (dict, list)):
        return raw_value

    text = str(raw_value).strip()
    if not text:
        return None

    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except Exception:
            continue
    return text


def looks_like_feature_record(value: Mapping[str, Any]) -> bool:
    normalized_keys = {normalize_feature_name(str(key)) for key in value.keys()}
    return bool(normalized_keys & NAME_KEYS) and bool(normalized_keys & VALUE_KEYS)


def extract_feature_name_from_record(value: Mapping[str, Any]) -> str:
    for key in ("name", "feature", "attribute", "characteristic", "title", "key"):
        if key in value:
            return clean_text_value(value[key])
    return ""


def extract_feature_value_from_record(value: Mapping[str, Any]) -> str:
    for key in ("value", "values", "text", "description", "value_name", "content"):
        if key in value:
            return clean_text_value(value[key])
    return ""


def flatten_features_level1(raw_features: Any) -> list[FeatureItem]:
    parsed = try_parse_json_like(raw_features)
    collected: list[FeatureItem] = []

    def add_item(name: Any, value: Any) -> None:
        name_text = clean_text_value(name)
        value_text = clean_text_value(value)
        if not name_text or not value_text:
            return
        norm_name = normalize_feature_name(name_text)
        if not norm_name:
            return
        collected.append(FeatureItem(name=name_text, value=value_text, norm_name=norm_name))

    if isinstance(parsed, Mapping):
        if looks_like_feature_record(parsed):
            add_item(extract_feature_name_from_record(parsed), extract_feature_value_from_record(parsed))
        else:
            for key, value in parsed.items():
                if isinstance(value, Mapping) and looks_like_feature_record(value):
                    add_item(extract_feature_name_from_record(value), extract_feature_value_from_record(value))
                elif not isinstance(value, (list, tuple, set, Mapping)):
                    add_item(key, value)
                elif isinstance(value, (list, tuple, set)):
                    for item in value:
                        if isinstance(item, Mapping) and looks_like_feature_record(item):
                            add_item(extract_feature_name_from_record(item), extract_feature_value_from_record(item))
                        elif not isinstance(item, (list, tuple, set, Mapping)):
                            add_item(key, item)
    elif isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, Mapping) and looks_like_feature_record(item):
                add_item(extract_feature_name_from_record(item), extract_feature_value_from_record(item))
            elif isinstance(item, Mapping):
                for key, value in item.items():
                    if not isinstance(value, (list, tuple, set, Mapping)):
                        add_item(key, value)

    deduped: list[FeatureItem] = []
    seen: set[tuple[str, str]] = set()
    for item in collected:
        dedupe_key = (item.norm_name, normalize_spaces(item.value.lower()))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        deduped.append(item)
    return deduped


def parse_sku_queries(raw_sku: Any) -> list[str]:
    payload = try_parse_json_like(raw_sku)
    queries: list[str] = []

    def walk(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str):
            text = normalize_spaces(value)
            if text:
                queries.append(text)
            return
        if isinstance(value, list):
            for item in value:
                walk(item)
            return
        if isinstance(value, Mapping):
            if "queries" in value:
                walk(value["queries"])
                return
            if "query" in value:
                walk(value["query"])
                return
            for item in value.values():
                walk(item)

    walk(payload)
    deduped: list[str] = []
    seen: set[str] = set()
    for query in queries:
        normalized = query.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(query)
    return deduped


def is_missing_error(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    text = normalize_spaces(str(value))
    return text == "" or text.lower() in {"none", "null", "nan"}


def detect_description_column(columns: Sequence[str]) -> str | None:
    available = set(columns)
    for column in DESCRIPTION_COLUMN_CANDIDATES:
        if column in available:
            return column
    return None


def iter_source_parquet_files(data_dir: Path) -> list[Path]:
    files = []
    for index in range(1, 30):
        candidate = data_dir / DATA_FILE_TEMPLATE.format(index=index)
        if candidate.exists():
            files.append(candidate)
    if not files:
        raise FileNotFoundError(f"No source parquet files data0001..data0029 found in {data_dir}")
    return files


def build_feature_frequency_from_files(files: Sequence[Path]) -> tuple[Counter[str], int, int]:
    feature_frequency: Counter[str] = Counter()
    total_rows = 0
    usable_rows = 0

    for path in files:
        df = pd.read_parquet(path)
        description_column = detect_description_column(df.columns.tolist())
        if TITLE_COLUMN not in df.columns or FEATURES_COLUMN not in df.columns or SKU_COLUMN not in df.columns:
            raise KeyError(f"Required columns not found in {path.name}")

        for row_index, row in enumerate(df.to_dict("records")):
            total_rows += 1
            title = clean_text_value(row.get(TITLE_COLUMN))
            if not title:
                continue
            sku_error = row.get(SKU_ERROR_COLUMN)
            if SKU_ERROR_COLUMN in row and not is_missing_error(sku_error):
                continue

            queries = parse_sku_queries(row.get(SKU_COLUMN))
            if not queries:
                continue

            features = flatten_features_level1(row.get(FEATURES_COLUMN))
            unique_feature_names = {feature.norm_name for feature in features if feature.norm_name}
            feature_frequency.update(unique_feature_names)
            usable_rows += 1

            _ = description_column  # keep branch symmetric with pass 2
            _ = row_index

    return feature_frequency, total_rows, usable_rows


def select_top_features(
    raw_features: Any,
    feature_frequency: Counter[str],
    *,
    max_features: int = DEFAULT_PRODUCT_MAX_FEATURES,
    title: str = "",
) -> list[dict[str, Any]]:
    features = flatten_features_level1(raw_features)
    ranked: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    normalized_title = normalize_feature_name(title)
    title_terms = set(normalized_title.split()) if normalized_title else set()

    for feature in features:
        if feature.norm_name in seen_names:
            continue
        seen_names.add(feature.norm_name)
        global_frequency = int(feature_frequency.get(feature.norm_name, 0))
        numeric_value_bonus = 1.0 if FEATURE_NUMERIC_VALUE_RE.search(str(feature.value)) else 0.0
        feature_terms = set(feature.norm_name.split())
        feature_name_in_title_bonus = 0.0
        if feature.norm_name and normalized_title:
            if feature.norm_name in normalized_title or (feature_terms and bool(feature_terms & title_terms)):
                feature_name_in_title_bonus = 1.0
        ranking_score = math.log1p(global_frequency) + numeric_value_bonus + feature_name_in_title_bonus
        ranked.append(
            {
                "name": feature.name,
                "value": feature.value,
                "norm_name": feature.norm_name,
                "global_frequency": global_frequency,
                "numeric_value_bonus": numeric_value_bonus,
                "feature_name_in_title_bonus": feature_name_in_title_bonus,
                "ranking_score": ranking_score,
            }
        )

    ranked.sort(
        key=lambda item: (
            -item["ranking_score"],
            -item["global_frequency"],
            item["norm_name"],
            item["name"],
            item["value"],
        )
    )
    return ranked[:max_features]


def build_processed_rows(
    files: Sequence[Path],
    feature_frequency: Counter[str],
    *,
    max_features: int = DEFAULT_PRODUCT_MAX_FEATURES,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for path in files:
        df = pd.read_parquet(path)
        description_column = detect_description_column(df.columns.tolist())
        for row_index, row in enumerate(df.to_dict("records")):
            title = clean_text_value(row.get(TITLE_COLUMN))
            if not title:
                continue
            sku_error = row.get(SKU_ERROR_COLUMN)
            if SKU_ERROR_COLUMN in row and not is_missing_error(sku_error):
                continue

            queries = parse_sku_queries(row.get(SKU_COLUMN))
            if not queries:
                continue

            selected_features = select_top_features(
                row.get(FEATURES_COLUMN),
                feature_frequency,
                max_features=max_features,
                title=title,
            )
            processed_row = dict(row)
            processed_row["source_file"] = path.name
            processed_row["source_row_index"] = row_index
            processed_row["description_for_training"] = clean_text_value(row.get(description_column)) if description_column else ""
            processed_row["training_queries"] = json.dumps(queries, ensure_ascii=False)
            processed_row["query_count"] = len(queries)
            processed_row["selected_features"] = json.dumps(selected_features, ensure_ascii=False)
            processed_row["selected_feature_count"] = len(selected_features)
            rows.append(processed_row)

    if not rows:
        raise ValueError("No valid training rows found after filtering.")
    return pd.DataFrame(rows)


def split_train_valid(
    frame: pd.DataFrame,
    *,
    train_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return frame.copy(), frame.copy()
    if train_ratio <= 0:
        return frame.iloc[0:0].copy(), frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if train_ratio >= 1:
        return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True), frame.iloc[0:0].copy()

    shuffled = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    train_size = int(len(shuffled) * train_ratio)
    train_size = min(max(train_size, 1), len(shuffled) - 1)
    train_df = shuffled.iloc[:train_size].reset_index(drop=True)
    valid_df = shuffled.iloc[train_size:].reset_index(drop=True)
    return train_df, valid_df


def estimate_token_count_from_chars(
    char_count: int,
    *,
    max_length: int,
    chars_per_token: float,
    reserve_tokens: int = 4,
) -> int:
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be positive.")
    if char_count <= 0:
        return min(max_length, reserve_tokens)
    estimated = reserve_tokens + math.ceil(char_count / chars_per_token)
    return max(1, min(max_length, estimated))


def estimate_product_char_length(product: Mapping[str, Any]) -> int:
    total = len(clean_text_value(product.get("title")))
    total += len(clean_text_value(product.get("description")))
    raw_features = product.get("features")
    if isinstance(raw_features, Sequence):
        for feature in raw_features:
            if not isinstance(feature, Mapping):
                continue
            total += len(clean_text_value(feature.get("name")))
            total += len(clean_text_value(feature.get("value")))
            total += 8
    return total


def estimate_training_example_cost_from_payload(
    *,
    query: str,
    product: Mapping[str, Any],
    query_max_length: int,
    product_max_length: int,
    chars_per_token: float,
) -> int:
    query_tokens = estimate_token_count_from_chars(
        len(normalize_spaces(query)),
        max_length=query_max_length,
        chars_per_token=chars_per_token,
    )
    product_tokens = estimate_token_count_from_chars(
        estimate_product_char_length(product),
        max_length=product_max_length,
        chars_per_token=chars_per_token,
    )
    attention_cost = product_tokens * product_tokens
    feedforward_cost = product_tokens * 12
    cross_cost = product_tokens * max(query_tokens, 1)
    return max(1, attention_cost + feedforward_cost + cross_cost)


class QueryProductPairDataset(Dataset[TrainingExample]):
    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        query_max_length: int,
        product_max_length: int,
        positive_queries_per_product: int = 5,
        chars_per_token: float = 4.0,
        allow_empty: bool = False,
    ) -> None:
        if positive_queries_per_product <= 0:
            raise ValueError("positive_queries_per_product must be positive.")
        self.products: list[dict[str, Any]] = []
        self.query_lists: list[list[str]] = []
        self.positive_queries_per_product = int(positive_queries_per_product)
        self.example_costs: list[int] = []
        self.query_token_estimates: list[int] = []
        self.product_token_estimates: list[int] = []

        for row in frame.to_dict("records"):
            queries = json.loads(row["training_queries"])
            selected_features = json.loads(row["selected_features"])
            product = {
                "title": clean_text_value(row.get(TITLE_COLUMN)),
                "description": clean_text_value(row.get("description_for_training")),
                "features": [{"name": feature["name"], "value": feature["value"]} for feature in selected_features],
            }
            if not queries:
                continue
            max_query_length_chars = max(len(normalize_spaces(query)) for query in queries)
            query_tokens = estimate_token_count_from_chars(
                max_query_length_chars,
                max_length=query_max_length,
                chars_per_token=chars_per_token,
            )
            product_tokens = estimate_token_count_from_chars(
                estimate_product_char_length(product),
                max_length=product_max_length,
                chars_per_token=chars_per_token,
            )
            query_multiplier = self.positive_queries_per_product
            product_cost = product_tokens * product_tokens + product_tokens * 12
            query_cost = query_multiplier * (query_tokens * query_tokens + query_tokens * 12)
            cross_cost = query_multiplier * product_tokens * max(query_tokens, 1)
            example_cost = max(1, product_cost + query_cost + cross_cost)
            self.products.append(product)
            self.query_lists.append(queries)
            self.example_costs.append(example_cost)
            self.query_token_estimates.append(query_tokens)
            self.product_token_estimates.append(product_tokens)

        if not self.products and not allow_empty:
            raise ValueError("Dataset produced no query-product training examples.")

    def __len__(self) -> int:
        return len(self.products)

    def __getitem__(self, index: int) -> TrainingExample:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        query_list = self.query_lists[index]
        if len(query_list) >= self.positive_queries_per_product:
            queries = random.sample(query_list, k=self.positive_queries_per_product)
        else:
            queries = list(query_list)
            while len(queries) < self.positive_queries_per_product:
                queries.append(random.choice(query_list))
        return TrainingExample(
            queries=queries,
            product=self.products[index],
        )


class DistributedDynamicBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        example_costs: Sequence[int],
        *,
        max_batch_size: int,
        target_batch_cost: int,
        rank: int = 0,
        world_size: int = 1,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
        bucket_size_multiplier: int = 32,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive.")
        if target_batch_cost <= 0:
            raise ValueError("target_batch_cost must be positive.")
        if world_size <= 0:
            raise ValueError("world_size must be positive.")
        if rank < 0 or rank >= world_size:
            raise ValueError("rank must be in [0, world_size).")

        self.example_costs = [max(1, int(cost)) for cost in example_costs]
        self.max_batch_size = int(max_batch_size)
        self.target_batch_cost = int(target_batch_cost)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = int(seed)
        self.bucket_size_multiplier = max(1, int(bucket_size_multiplier))
        self.epoch = 0
        self._cached_epoch: int | None = None
        self._cached_batches_per_rank: list[list[list[int]]] | None = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._cached_epoch = None
        self._cached_batches_per_rank = None

    def set_target_batch_cost(self, target_batch_cost: int) -> None:
        self.target_batch_cost = max(1, int(target_batch_cost))
        self._cached_epoch = None
        self._cached_batches_per_rank = None

    def _build_batches_per_rank(self) -> list[list[list[int]]]:
        if self._cached_epoch == self.epoch and self._cached_batches_per_rank is not None:
            return self._cached_batches_per_rank

        indices = list(range(len(self.example_costs)))
        rng = random.Random(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(indices)

        bucket_size = max(self.world_size, self.max_batch_size * self.world_size * self.bucket_size_multiplier)
        ordered_indices: list[int] = []
        for start in range(0, len(indices), bucket_size):
            chunk = indices[start : start + bucket_size]
            chunk.sort(key=lambda index: self.example_costs[index], reverse=True)
            ordered_indices.extend(chunk)

        global_target_cost = self.target_batch_cost * self.world_size
        global_max_examples = self.max_batch_size * self.world_size
        global_batches: list[list[int]] = []
        current_batch: list[int] = []
        current_cost = 0

        for index in ordered_indices:
            example_cost = self.example_costs[index]
            exceeds_example_budget = len(current_batch) >= global_max_examples
            exceeds_cost_budget = bool(current_batch) and (current_cost + example_cost > global_target_cost)
            if exceeds_example_budget or exceeds_cost_budget:
                global_batches.append(current_batch)
                current_batch = []
                current_cost = 0
            current_batch.append(index)
            current_cost += example_cost

        if current_batch:
            global_batches.append(current_batch)

        if self.drop_last:
            global_batches = [batch for batch in global_batches if len(batch) >= self.world_size]

        batches_per_rank: list[list[list[int]]] = [[] for _ in range(self.world_size)]
        for batch in global_batches:
            if not batch:
                continue
            padded_batch = list(batch)
            if len(padded_batch) < self.world_size and not self.drop_last:
                pad_cursor = 0
                while len(padded_batch) < self.world_size:
                    padded_batch.append(batch[pad_cursor % len(batch)])
                    pad_cursor += 1

            per_rank_indices: list[list[int]] = [[] for _ in range(self.world_size)]
            per_rank_costs = [0 for _ in range(self.world_size)]
            for index in sorted(padded_batch, key=lambda item: self.example_costs[item], reverse=True):
                eligible_ranks = [rank for rank in range(self.world_size) if len(per_rank_indices[rank]) < self.max_batch_size]
                if not eligible_ranks:
                    raise RuntimeError("No eligible rank found while packing a dynamic batch.")
                target_rank = min(
                    eligible_ranks,
                    key=lambda rank: (len(per_rank_indices[rank]), per_rank_costs[rank], rank),
                )
                per_rank_indices[target_rank].append(index)
                per_rank_costs[target_rank] += self.example_costs[index]

            if self.drop_last and any(not rank_batch for rank_batch in per_rank_indices):
                continue
            for rank, rank_batch in enumerate(per_rank_indices):
                if not rank_batch and batch:
                    rank_batch = [batch[rank % len(batch)]]
                batches_per_rank[rank].append(rank_batch)

        self._cached_epoch = self.epoch
        self._cached_batches_per_rank = batches_per_rank
        return batches_per_rank

    def __iter__(self):
        batches_per_rank = self._build_batches_per_rank()
        for batch in batches_per_rank[self.rank]:
            yield batch

    def __len__(self) -> int:
        batches_per_rank = self._build_batches_per_rank()
        return len(batches_per_rank[self.rank])


def collate_training_examples(batch: Sequence[TrainingExample]) -> tuple[list[list[str]], list[dict[str, Any]]]:
    queries = [item.queries for item in batch]
    products = [item.product for item in batch]
    return queries, products


def flatten_grouped_queries(grouped_queries: Sequence[Sequence[str]]) -> tuple[list[str], torch.Tensor]:
    flat_queries: list[str] = []
    group_ids: list[int] = []
    for group_index, queries in enumerate(grouped_queries):
        if not queries:
            raise ValueError("Each product must have at least one positive query.")
        for query in queries:
            flat_queries.append(query)
            group_ids.append(group_index)
    if not flat_queries:
        raise ValueError("Batch produced no queries.")
    return flat_queries, torch.tensor(group_ids, dtype=torch.long)


def setup_distributed(args: argparse.Namespace) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed_enabled = world_size > 1

    if distributed_enabled and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if distributed_enabled and not dist.is_initialized():
        backend = args.distributed_backend
        if backend == "auto":
            backend = "nccl" if torch.cuda.is_available() else "gloo"
        init_kwargs: dict[str, Any] = {
            "backend": backend,
            "timeout": timedelta(minutes=max(1, int(getattr(args, "distributed_timeout_minutes", 60)))),
        }
        if backend == "nccl" and torch.cuda.is_available():
            init_kwargs["device_id"] = torch.device("cuda", local_rank)
        dist.init_process_group(**init_kwargs)

    if distributed_enabled:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed multi-GPU training requires CUDA.")
        device = torch.device("cuda", local_rank)
    else:
        device = resolve_device(args.device)

    use_fsdp = bool(args.fsdp and distributed_enabled and device.type == "cuda")
    return DistributedContext(
        enabled=distributed_enabled,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        use_fsdp=use_fsdp,
    )


def cleanup_distributed(context: DistributedContext) -> None:
    if context.enabled and dist.is_initialized():
        dist.destroy_process_group()


def distributed_barrier(context: DistributedContext) -> None:
    if context.enabled and dist.is_initialized():
        if context.device.type == "cuda":
            dist.barrier(device_ids=[context.local_rank])
        else:
            dist.barrier()


def is_fsdp_model(model: nn.Module) -> bool:
    return isinstance(model, FSDP)


def unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, FSDP):
        return model.module
    wrapped = getattr(model, "module", None)
    return wrapped if isinstance(wrapped, nn.Module) else model


def main_print(context: DistributedContext, message: str) -> None:
    if context.is_main_process:
        print(message)


def forward_dual_encoder(
    model: nn.Module,
    *,
    queries: Sequence[str] | Sequence[Sequence[str]],
    products: Sequence[dict[str, Any]],
    tokenizer_path: str | Path,
    query_max_length: int,
    product_max_length: int,
    device: torch.device,
    compute_loss: bool,
    symmetric_loss: bool,
    compute_token_type_loss: bool = False,
):
    if queries and isinstance(queries[0], (list, tuple)):
        flat_queries, _group_ids = flatten_grouped_queries(queries)  # type: ignore[arg-type]
    else:
        flat_queries = list(queries)  # type: ignore[arg-type]
    return model(
        queries=flat_queries,
        products=products,
        tokenizer_path=tokenizer_path,
        query_max_length=query_max_length,
        product_max_length=product_max_length,
        device=device,
        compute_loss=compute_loss,
        symmetric_loss=symmetric_loss,
        compute_token_type_loss=compute_token_type_loss,
    )


def resolve_target_batch_cost(
    *,
    example_costs: Sequence[int],
    max_batch_size: int,
    override_cost: int | None,
    headroom: float,
) -> int:
    if override_cost is not None:
        return max(1, int(override_cost))
    if headroom <= 0:
        raise ValueError("headroom must be positive.")
    if not example_costs:
        return max(1, int(max_batch_size))
    average_cost = sum(example_costs) / max(len(example_costs), 1)
    estimated = int(max_batch_size * average_cost * headroom)
    return max(1, estimated)


def gather_variable_batch_embeddings(
    embeddings: torch.Tensor,
    *,
    distributed_context: DistributedContext,
) -> tuple[torch.Tensor, list[int]]:
    if not distributed_context.enabled:
        return embeddings, [embeddings.size(0)]

    local_batch_size = embeddings.size(0)
    size_tensor = torch.tensor([local_batch_size], device=embeddings.device, dtype=torch.long)
    gathered_sizes = [torch.zeros_like(size_tensor) for _ in range(distributed_context.world_size)]
    dist.all_gather(gathered_sizes, size_tensor)
    batch_sizes = [int(item.item()) for item in gathered_sizes]
    max_batch_size = max(batch_sizes, default=0)
    if max_batch_size <= 0:
        return embeddings.new_zeros((0, embeddings.size(-1))), batch_sizes

    detached_padded = embeddings.new_zeros((max_batch_size, embeddings.size(-1)))
    if local_batch_size > 0:
        detached_padded[:local_batch_size] = embeddings.detach()

    gathered_embeddings = [torch.zeros_like(detached_padded) for _ in range(distributed_context.world_size)]
    dist.all_gather(gathered_embeddings, detached_padded)

    live_padded = embeddings.new_zeros((max_batch_size, embeddings.size(-1)))
    if local_batch_size > 0:
        live_padded[:local_batch_size] = embeddings
    gathered_embeddings[distributed_context.rank] = live_padded

    trimmed = [tensor[:size] for tensor, size in zip(gathered_embeddings, batch_sizes, strict=False) if size > 0]
    if not trimmed:
        return embeddings.new_zeros((0, embeddings.size(-1))), batch_sizes
    return torch.cat(trimmed, dim=0), batch_sizes


def gather_variable_batch_long_values(
    values: torch.Tensor,
    *,
    distributed_context: DistributedContext,
) -> tuple[torch.Tensor, list[int]]:
    if values.dim() != 1:
        raise ValueError("values must be a 1D tensor.")
    values = values.to(dtype=torch.long)
    if not distributed_context.enabled:
        return values, [values.size(0)]

    local_batch_size = values.size(0)
    size_tensor = torch.tensor([local_batch_size], device=values.device, dtype=torch.long)
    gathered_sizes = [torch.zeros_like(size_tensor) for _ in range(distributed_context.world_size)]
    dist.all_gather(gathered_sizes, size_tensor)
    batch_sizes = [int(item.item()) for item in gathered_sizes]
    max_batch_size = max(batch_sizes, default=0)
    if max_batch_size <= 0:
        return values.new_zeros((0,)), batch_sizes

    padded = values.new_full((max_batch_size,), fill_value=-1)
    if local_batch_size > 0:
        padded[:local_batch_size] = values
    gathered_values = [torch.empty_like(padded) for _ in range(distributed_context.world_size)]
    dist.all_gather(gathered_values, padded)
    trimmed = [tensor[:size] for tensor, size in zip(gathered_values, batch_sizes, strict=False) if size > 0]
    if not trimmed:
        return values.new_zeros((0,)), batch_sizes
    return torch.cat(trimmed, dim=0), batch_sizes


def compute_in_batch_retrieval_metrics(
    model: nn.Module,
    *,
    query_embeddings: torch.Tensor,
    product_embeddings: torch.Tensor,
    query_group_ids: torch.Tensor | None = None,
    distributed_context: DistributedContext,
    symmetric_loss: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    temperature = float(getattr(unwrap_model(model), "temperature"))
    temperature = max(temperature, 1e-6)

    if query_group_ids is None:
        query_group_ids = torch.arange(query_embeddings.size(0), device=query_embeddings.device)
    else:
        query_group_ids = query_group_ids.to(device=query_embeddings.device, dtype=torch.long)
    if query_embeddings.size(0) == 0 or product_embeddings.size(0) == 0:
        raise ValueError("query_embeddings and product_embeddings must be non-empty.")
    if query_group_ids.numel() != query_embeddings.size(0):
        raise ValueError("query_group_ids length must match query_embeddings batch size.")

    if distributed_context.enabled:
        global_queries, query_batch_sizes = gather_variable_batch_embeddings(
            query_embeddings,
            distributed_context=distributed_context,
        )
        global_products, product_batch_sizes = gather_variable_batch_embeddings(
            product_embeddings,
            distributed_context=distributed_context,
        )
        global_local_query_group_ids, query_group_batch_sizes = gather_variable_batch_long_values(
            query_group_ids,
            distributed_context=distributed_context,
        )
        if query_batch_sizes != query_group_batch_sizes:
            raise RuntimeError("Query embedding and query group id batch sizes diverged across ranks.")
        local_product_batch_size = product_embeddings.size(0)
        if int(query_group_ids.max().item()) >= local_product_batch_size:
            raise ValueError("query_group_ids must point to local product indices.")
        label_offset = sum(product_batch_sizes[: distributed_context.rank])
        labels = query_group_ids + label_offset
        global_query_offsets: list[int] = []
        product_offset = 0
        for rank_index, query_count in enumerate(query_batch_sizes):
            global_query_offsets.extend([product_offset] * query_count)
            product_offset += product_batch_sizes[rank_index]
        global_query_offsets_tensor = torch.tensor(
            global_query_offsets,
            device=query_embeddings.device,
            dtype=torch.long,
        )
        global_query_group_ids = global_local_query_group_ids + global_query_offsets_tensor
        similarity = query_embeddings @ global_products.transpose(-1, -2)
        loss = F.cross_entropy(similarity / temperature, labels)
        if symmetric_loss:
            reverse_similarity = product_embeddings @ global_queries.transpose(-1, -2)
            local_product_labels = torch.arange(local_product_batch_size, device=product_embeddings.device) + label_offset
            positive_mask = local_product_labels[:, None] == global_query_group_ids[None, :]
            reverse_log_probs = F.log_softmax(reverse_similarity / temperature, dim=1)
            positive_counts = positive_mask.sum(dim=1).clamp_min(1).to(dtype=reverse_log_probs.dtype)
            reverse_loss = -(
                reverse_log_probs * positive_mask.to(dtype=reverse_log_probs.dtype)
            ).sum(dim=1) / positive_counts
            reverse_loss = reverse_loss.mean()
            loss = 0.5 * (loss + reverse_loss)
        return similarity, loss, labels

    if int(query_group_ids.max().item()) >= product_embeddings.size(0):
        raise ValueError("query_group_ids must point to product indices.")
    labels = query_group_ids
    similarity = query_embeddings @ product_embeddings.transpose(-1, -2)
    loss = F.cross_entropy(similarity / temperature, labels)
    if symmetric_loss:
        reverse_similarity = product_embeddings @ query_embeddings.transpose(-1, -2)
        product_labels = torch.arange(product_embeddings.size(0), device=product_embeddings.device)
        positive_mask = product_labels[:, None] == query_group_ids[None, :]
        reverse_log_probs = F.log_softmax(reverse_similarity / temperature, dim=1)
        positive_counts = positive_mask.sum(dim=1).clamp_min(1).to(dtype=reverse_log_probs.dtype)
        reverse_loss = -(
            reverse_log_probs * positive_mask.to(dtype=reverse_log_probs.dtype)
        ).sum(dim=1) / positive_counts
        reverse_loss = reverse_loss.mean()
        loss = 0.5 * (loss + reverse_loss)
    return similarity, loss, labels


def compute_token_type_auxiliary_loss(
    output,
    *,
    weight: float,
) -> tuple[torch.Tensor | None, float]:
    if weight <= 0:
        return None, 0.0

    losses = []
    query_loss = getattr(output.query_output, "token_type_loss", None)
    product_loss = getattr(output.product_output, "token_type_loss", None)
    if query_loss is not None:
        losses.append(query_loss)
    if product_loss is not None:
        losses.append(product_loss)
    if not losses:
        return None, 0.0

    aux_loss = torch.stack(losses).mean()
    return aux_loss * weight, float(aux_loss.detach().item())


def resolve_device(requested: str | None) -> torch.device:
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def checkpoint_artifact_paths(path: Path) -> dict[str, Path]:
    return {
        "metadata": path,
        "model_safetensors": path.with_suffix(".safetensors"),
        "model_torch": path.with_suffix(".model.pt"),
        "query_encoder_safetensors": path.with_suffix(".query_encoder.safetensors"),
        "query_encoder_torch": path.with_suffix(".query_encoder.pt"),
        "product_encoder_safetensors": path.with_suffix(".product_encoder.safetensors"),
        "product_encoder_torch": path.with_suffix(".product_encoder.pt"),
        "optimizer_torch": path.with_suffix(".optimizer.pt"),
    }


def _tensor_state_dict_for_serialization(model_state_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
    serialized: dict[str, torch.Tensor] = {}
    for key, value in model_state_dict.items():
        if torch.is_tensor(value):
            serialized[key] = value.detach().contiguous().cpu()
    return serialized


def _strip_distributed_state_prefix(key: str) -> str:
    for prefix in ("_fsdp_wrapped_module.", "module."):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _extract_encoder_state_dict(
    model_state_dict: dict[str, Any],
    *,
    encoder_prefix: str,
) -> dict[str, torch.Tensor]:
    extracted: dict[str, torch.Tensor] = {}
    prefix = f"{encoder_prefix}."
    for key, value in model_state_dict.items():
        normalized_key = _strip_distributed_state_prefix(key)
        if not normalized_key.startswith(prefix) or not torch.is_tensor(value):
            continue
        extracted[normalized_key[len(prefix) :]] = value.detach().contiguous().cpu()
    return extracted


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    build_spec: DualEncoderBuildSpec,
    metrics: dict[str, float],
    args: argparse.Namespace,
    distributed_context: DistributedContext,
    save_optimizer_state: bool,
    export_safetensors: bool,
) -> None:
    if is_fsdp_model(model):
        state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        optim_state_dict_config = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            state_dict_config,
            optim_state_dict_config,
        ):
            model_state_dict = model.state_dict()
            optimizer_state_dict = FSDP.optim_state_dict(model, optimizer) if save_optimizer_state else None
        if not distributed_context.is_main_process:
            return
    else:
        if not distributed_context.is_main_process:
            return
        model_state_dict = model.state_dict()
        optimizer_state_dict = optimizer.state_dict() if save_optimizer_state else None

    path.parent.mkdir(parents=True, exist_ok=True)
    artifact_paths = checkpoint_artifact_paths(path)
    model_state_format = "torch"
    model_state_path = artifact_paths["model_torch"]

    serialized_model_state = _tensor_state_dict_for_serialization(model_state_dict)
    query_encoder_state = _extract_encoder_state_dict(model_state_dict, encoder_prefix="query_encoder")
    product_encoder_state = _extract_encoder_state_dict(model_state_dict, encoder_prefix="product_encoder")
    if export_safetensors and SAFETENSORS_AVAILABLE:
        safetensors_save_file(serialized_model_state, str(artifact_paths["model_safetensors"]))
        if query_encoder_state:
            safetensors_save_file(query_encoder_state, str(artifact_paths["query_encoder_safetensors"]))
        if product_encoder_state:
            safetensors_save_file(product_encoder_state, str(artifact_paths["product_encoder_safetensors"]))
        model_state_format = "safetensors"
        model_state_path = artifact_paths["model_safetensors"]
        query_encoder_state_format = "safetensors" if query_encoder_state else None
        query_encoder_state_path = artifact_paths["query_encoder_safetensors"] if query_encoder_state else None
        product_encoder_state_format = "safetensors" if product_encoder_state else None
        product_encoder_state_path = artifact_paths["product_encoder_safetensors"] if product_encoder_state else None
    else:
        torch.save(model_state_dict, artifact_paths["model_torch"])
        if query_encoder_state:
            torch.save(query_encoder_state, artifact_paths["query_encoder_torch"])
        if product_encoder_state:
            torch.save(product_encoder_state, artifact_paths["product_encoder_torch"])
        query_encoder_state_format = "torch" if query_encoder_state else None
        query_encoder_state_path = artifact_paths["query_encoder_torch"] if query_encoder_state else None
        product_encoder_state_format = "torch" if product_encoder_state else None
        product_encoder_state_path = artifact_paths["product_encoder_torch"] if product_encoder_state else None

    optimizer_state_path = None
    if optimizer_state_dict is not None:
        optimizer_state_path = artifact_paths["optimizer_torch"]
        torch.save(optimizer_state_dict, optimizer_state_path)

    payload = {
        "checkpoint_format_version": 2,
        "epoch": epoch,
        "global_step": global_step,
        "build_spec": asdict(build_spec),
        "metrics": metrics,
        "args": vars(args),
        "model_state_format": model_state_format,
        "model_state_path": model_state_path.name,
        "query_encoder_state_format": query_encoder_state_format,
        "query_encoder_state_path": query_encoder_state_path.name if query_encoder_state_path is not None else None,
        "product_encoder_state_format": product_encoder_state_format,
        "product_encoder_state_path": product_encoder_state_path.name if product_encoder_state_path is not None else None,
        "optimizer_state_format": "torch" if optimizer_state_path is not None else None,
        "optimizer_state_path": optimizer_state_path.name if optimizer_state_path is not None else None,
        "has_optimizer_state": optimizer_state_path is not None,
        "safetensors_available_at_save": SAFETENSORS_AVAILABLE,
    }
    torch.save(payload, path)


def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[list[str], list[dict[str, Any]]]],
    *,
    tokenizer_path: str | Path,
    query_max_length: int,
    product_max_length: int,
    device: torch.device,
    symmetric_loss: bool,
    distributed_context: DistributedContext,
    token_type_loss_weight: float,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    total_examples = 0
    use_amp = device.type == "cuda"

    with torch.no_grad():
        for grouped_queries, products in loader:
            _flat_queries, query_group_ids = flatten_grouped_queries(grouped_queries)
            query_group_ids = query_group_ids.to(device=device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                output = forward_dual_encoder(
                    model,
                    queries=grouped_queries,
                    products=products,
                    tokenizer_path=tokenizer_path,
                    query_max_length=query_max_length,
                    product_max_length=product_max_length,
                    device=device,
                    compute_loss=False,
                    symmetric_loss=symmetric_loss,
                    compute_token_type_loss=token_type_loss_weight > 0,
                )
                similarity, loss, labels = compute_in_batch_retrieval_metrics(
                    model,
                    query_embeddings=output.query_output.normalized_embedding,
                    product_embeddings=output.product_output.normalized_embedding,
                    query_group_ids=query_group_ids,
                    distributed_context=distributed_context,
                    symmetric_loss=symmetric_loss,
                )
                auxiliary_loss, _aux_scalar = compute_token_type_auxiliary_loss(
                    output,
                    weight=token_type_loss_weight,
                )
                if auxiliary_loss is not None:
                    loss = loss + auxiliary_loss

            batch_size = int(query_group_ids.numel())
            accuracy = (similarity.argmax(dim=1) == labels).float().mean().item()
            total_loss += float(loss.detach()) * batch_size
            total_accuracy += accuracy * batch_size
            total_examples += batch_size

    if distributed_context.enabled:
        totals = torch.tensor(
            [total_loss, total_accuracy, float(total_examples)],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        total_loss = float(totals[0].item())
        total_accuracy = float(totals[1].item())
        total_examples = int(totals[2].item())

    if total_examples == 0:
        return {"loss": math.nan, "accuracy_at_1": math.nan}
    return {
        "loss": total_loss / total_examples,
        "accuracy_at_1": total_accuracy / total_examples,
    }


def train(
    *,
    args: argparse.Namespace,
    model: nn.Module,
    build_spec: DualEncoderBuildSpec,
    train_loader: DataLoader[tuple[list[str], list[dict[str, Any]]]],
    valid_loader: DataLoader[tuple[list[str], list[dict[str, Any]]]],
    device: torch.device,
    output_dir: Path,
    distributed_context: DistributedContext,
    model_description: dict[str, int | float],
) -> None:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=False)

    metrics_log: list[dict[str, float | int]] = []
    best_valid_loss = float("inf")
    global_step = 0
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
    if distributed_context.is_main_process:
        if args.tensorboard and not monitor.tensorboard_enabled:
            print("TensorBoard logging is disabled because torch.utils.tensorboard is unavailable in this environment.")
        if not monitor.plot_enabled:
            print("Live PNG loss plot is disabled because matplotlib is unavailable in this environment.")

    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            if train_epoch_controller is not None:
                train_epoch_controller.set_epoch(epoch)
            epoch_started = time.perf_counter()
            running_loss = 0.0
            running_batches = 0

            for grouped_queries, products in train_loader:
                _flat_queries, query_group_ids = flatten_grouped_queries(grouped_queries)
                query_group_ids = query_group_ids.to(device=device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    output = forward_dual_encoder(
                        model,
                        queries=grouped_queries,
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
                    epoch=epoch,
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
                "epoch_seconds": time.perf_counter() - epoch_started,
            }
            metrics_log.append(epoch_metrics)
            monitor.log_epoch(
                epoch=epoch,
                global_step=global_step,
                train_loss=train_loss,
                valid_loss=valid_metrics["loss"],
                valid_accuracy_at_1=valid_metrics["accuracy_at_1"],
                epoch_seconds=epoch_metrics["epoch_seconds"],
            )

            main_print(
                distributed_context,
                f"Epoch {epoch}: "
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
        pd.DataFrame(metrics_log).to_parquet(output_dir / "training_metrics.parquet", index=False)
        with (output_dir / "model_description.json").open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "build_spec": asdict(build_spec),
                    "model_description": model_description,
                    "args": vars(args),
                },
                file,
                ensure_ascii=False,
                indent=2,
            )


def prepare_corpus(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = Path(args.data_dir)
    files = iter_source_parquet_files(data_dir)
    print(f"Found {len(files)} source parquet files in {data_dir}")

    feature_frequency, total_rows, usable_rows = build_feature_frequency_from_files(files)
    print(f"Corpus scan complete: total_rows={total_rows}, usable_rows={usable_rows}, unique_features={len(feature_frequency)}")

    processed_df = build_processed_rows(files, feature_frequency, max_features=args.product_max_features)
    train_df, valid_df = split_train_valid(processed_df, train_ratio=args.train_ratio, seed=args.seed)

    frequency_df = pd.DataFrame(
        sorted(
            ({"feature_name": name, "frequency": count} for name, count in feature_frequency.items()),
            key=lambda item: (-item["frequency"], item["feature_name"]),
        )
    )

    train_path = data_dir / "train.parquet"
    valid_path = data_dir / "valid.parquet"
    frequency_path = data_dir / "feature_frequency.parquet"
    train_df.to_parquet(train_path, index=False)
    valid_df.to_parquet(valid_path, index=False)
    frequency_df.to_parquet(frequency_path, index=False)

    print(f"Saved train split: {train_path} ({len(train_df)} rows)")
    print(f"Saved valid split: {valid_path} ({len(valid_df)} rows)")
    print(f"Saved feature frequency: {frequency_path} ({len(frequency_df)} rows)")
    return train_df, valid_df, frequency_df


def load_prepared_corpus(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path = data_dir / "train.parquet"
    valid_path = data_dir / "valid.parquet"
    frequency_path = data_dir / "feature_frequency.parquet"
    if not train_path.exists() or not valid_path.exists() or not frequency_path.exists():
        raise FileNotFoundError("Prepared train/valid/frequency parquet files are missing.")
    return (
        pd.read_parquet(train_path),
        pd.read_parquet(valid_path),
        pd.read_parquet(frequency_path),
    )


def prepare_or_load_corpus(
    args: argparse.Namespace,
    distributed_context: DistributedContext,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = Path(args.data_dir)
    train_path = data_dir / "train.parquet"
    valid_path = data_dir / "valid.parquet"
    frequency_path = data_dir / "feature_frequency.parquet"

    if distributed_context.is_main_process:
        if args.reuse_prepared_splits and train_path.exists() and valid_path.exists() and frequency_path.exists():
            main_print(distributed_context, f"Loading existing prepared splits from {data_dir}")
            train_df, valid_df, frequency_df = load_prepared_corpus(data_dir)
        else:
            train_df, valid_df, frequency_df = prepare_corpus(args)
    else:
        train_df = valid_df = frequency_df = None

    distributed_barrier(distributed_context)

    if not distributed_context.is_main_process:
        train_df, valid_df, frequency_df = load_prepared_corpus(data_dir)

    assert train_df is not None and valid_df is not None and frequency_df is not None
    return train_df, valid_df, frequency_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare parquet corpus and train a query/product dual-encoder with in-batch negatives."
    )
    parser.add_argument("--data-dir", required=True, help="Directory containing data0001.parquet ... data0029.parquet")
    parser.add_argument("--tokenizer-path", default="tokenizer (1).json", help="Path to tokenizer json")
    parser.add_argument("--output-dir", "--save-dir", dest="output_dir", default=None, help="Directory for checkpoints, logs and metrics. Defaults to <data-dir>/training_artifacts")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--reuse-prepared-splits", action=argparse.BooleanOptionalAction, default=True, help="Reuse existing train.parquet, valid.parquet and feature_frequency.parquet when present")

    parser.add_argument("--batch-size", type=int, default=32, help="Product batch size. Each product contributes --positive-queries-per-product query positives.")
    parser.add_argument("--positive-queries-per-product", type=int, default=5, help="How many positive SKU queries to sample for each product in every batch.")
    parser.add_argument("--dynamic-batching", action=argparse.BooleanOptionalAction, default=True, help="Pack batches by approximate example cost so short cards use larger batches and long cards use smaller ones.")
    parser.add_argument("--dynamic-cost-per-device", type=int, default=None, help="Approximate per-device batch cost budget. By default it is derived from average example cost * batch_size.")
    parser.add_argument("--dynamic-cost-headroom", type=float, default=0.9, help="Headroom multiplier used when deriving the per-device dynamic batch budget automatically.")
    parser.add_argument("--dynamic-bucket-multiplier", type=int, default=32, help="How many max-size global batches to group before sorting by cost for dynamic packing.")
    parser.add_argument("--dynamic-chars-per-token", type=float, default=4.0, help="Approximate characters-per-token ratio used to estimate sequence lengths before batching.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="AdamW learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--max-train-steps", type=int, default=None, help="Optional hard cap on optimizer steps")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader workers")
    parser.add_argument("--device", default="auto", help="Device string, for example cuda, cpu, cuda:0")
    parser.add_argument("--prepare-only", action="store_true", help="Only build train/valid/frequency parquet files, skip training")
    parser.add_argument("--symmetric-loss", action=argparse.BooleanOptionalAction, default=True, help="Use symmetric product<->query multi-positive contrastive loss")
    parser.add_argument("--fsdp", action=argparse.BooleanOptionalAction, default=True, help="Use FSDP parameter sharding when launched with torchrun on multiple GPUs")
    parser.add_argument("--fsdp-sync-module-states", action=argparse.BooleanOptionalAction, default=False, help="Synchronize module weights from rank 0 during FSDP init. Disable for scratch training when all ranks start from the same seed.")
    parser.add_argument("--distributed-backend", default="auto", help="Distributed backend: auto, nccl or gloo")
    parser.add_argument("--distributed-timeout-minutes", type=int, default=60, help="Distributed collective timeout in minutes for long model initialization and checkpoint loading.")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True, help="Recompute transformer activations to reduce memory")
    parser.add_argument("--save-optimizer-state", action=argparse.BooleanOptionalAction, default=True, help="Save optimizer state in checkpoints so continue-training can restore the optimizer exactly.")
    parser.add_argument("--export-safetensors", action=argparse.BooleanOptionalAction, default=True, help="Save model weights as .safetensors when the safetensors package is available, otherwise fall back to an external .model.pt file.")
    parser.add_argument("--log-every", type=int, default=10, help="How often to log step-level loss to CSV/TensorBoard")
    parser.add_argument("--plot-every", type=int, default=50, help="How often to refresh the live loss PNG")
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True, help="Write live TensorBoard logs")
    parser.add_argument("--token-type-loss-weight", type=float, default=0.1, help="Weight of auxiliary token-type classification loss for the learned int/str router")

    parser.add_argument("--query-max-length", type=int, default=256, help="Maximum query sequence length")
    parser.add_argument("--product-max-length", type=int, default=1024, help="Maximum product sequence length")
    parser.add_argument("--embedding-dim", type=int, default=1024, help="Shared retrieval embedding dimension")
    parser.add_argument("--temperature", type=float, default=0.05, help="Contrastive loss temperature")

    parser.add_argument("--query-target-parameters", type=int, default=3_000_000_000, help="Target parameter count for query encoder")
    parser.add_argument("--query-num-layers", type=int, default=30, help="Query encoder transformer layers")
    parser.add_argument("--query-token-type-router-layers", type=int, default=4, help="Query encoder token type/router classifier transformer layers")
    parser.add_argument("--query-num-heads", type=int, default=16, help="Query encoder attention heads")
    parser.add_argument("--query-ff-multiplier", type=int, default=4, help="Query encoder FF multiplier")
    parser.add_argument("--query-max-position-embeddings", type=int, default=2048, help="Query encoder max position embeddings")
    parser.add_argument("--query-dropout", type=float, default=0.1, help="Query encoder dropout")

    parser.add_argument("--product-d-model", type=int, default=1024, help="Product encoder hidden size")
    parser.add_argument("--product-num-heads", type=int, default=16, help="Product encoder attention heads")
    parser.add_argument("--product-num-layers", type=int, default=30, help="Product encoder transformer layers")
    parser.add_argument("--product-token-type-router-layers", type=int, default=4, help="Product encoder token type/router classifier transformer layers")
    parser.add_argument("--product-ff-multiplier", type=int, default=4, help="Product encoder FF multiplier")
    parser.add_argument("--product-max-position-embeddings", type=int, default=8194, help="Product encoder max position embeddings")
    parser.add_argument("--product-max-features", type=int, default=DEFAULT_PRODUCT_MAX_FEATURES, help="Maximum number of product features to pass into the encoder")
    parser.add_argument("--product-local-attention-window", type=int, default=128, help="Product encoder local attention radius for Longformer-style sparse attention")
    parser.add_argument("--product-max-routing-clusters", type=int, default=128, help="Maximum routing centroids; each layer uses ceil(sqrt(active_tokens)) clusters capped by this value")
    parser.add_argument("--product-dropout", type=float, default=0.1, help="Product encoder dropout")
    return parser.parse_args()


def configure_model_for_training(
    model: ProductQueryDualEncoder,
    *,
    args: argparse.Namespace,
    distributed_context: DistributedContext,
) -> nn.Module:
    if distributed_context.use_fsdp and args.gradient_checkpointing:
        non_reentrant_wrapper = partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda module: isinstance(module, (DualPathTransformerLayer, TokenTypeTransformerLayer)),
        )
        model.gradient_checkpointing_disable()
    elif args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()

    if distributed_context.use_fsdp:
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={DualPathTransformerLayer},
        )
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
        return FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mixed_precision,
            device_id=distributed_context.local_rank,
            use_orig_params=True,
            sync_module_states=bool(getattr(args, "fsdp_sync_module_states", False)),
            limit_all_gathers=True,
        )

    model.to(distributed_context.device)
    return model


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    distributed_context = setup_distributed(args)
    try:
        train_df, valid_df, _frequency_df = prepare_or_load_corpus(args, distributed_context)
        if args.prepare_only:
            return

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

        output_dir = Path(args.output_dir) if args.output_dir else Path(args.data_dir) / "training_artifacts"
        if distributed_context.is_main_process:
            output_dir.mkdir(parents=True, exist_ok=True)
        distributed_barrier(distributed_context)

        model, build_spec = build_product_query_dual_encoder_from_tokenizer_json(
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
        model_description = model.describe()
        model = configure_model_for_training(
            model,
            args=args,
            distributed_context=distributed_context,
        )
        main_print(distributed_context, f"Using device: {distributed_context.device}")
        if distributed_context.use_fsdp:
            main_print(distributed_context, f"FSDP enabled across {distributed_context.world_size} GPUs")
        if args.dynamic_batching:
            main_print(
                distributed_context,
                f"Dynamic batching enabled: max_examples_per_device={args.batch_size}, "
                f"target_cost_per_device={train_target_batch_cost}, "
                f"chars_per_token={args.dynamic_chars_per_token}",
            )
        if args.export_safetensors and not SAFETENSORS_AVAILABLE:
            main_print(
                distributed_context,
                "safetensors package is unavailable in this environment; model weights will be saved as external .model.pt files instead.",
            )
        main_print(distributed_context, f"Saving artifacts to: {output_dir}")
        if args.tensorboard:
            main_print(distributed_context, f"TensorBoard log dir: {output_dir / 'tensorboard'}")
        main_print(distributed_context, f"Live loss plot: {output_dir / 'live_loss.png'}")
        main_print(distributed_context, f"Model description: {model_description}")
        main_print(distributed_context, f"Build spec: {asdict(build_spec)}")
        main_print(
            distributed_context,
            f"Training examples: train={len(train_dataset)}, valid={len(valid_dataset)}",
        )

        train(
            args=args,
            model=model,
            build_spec=build_spec,
            train_loader=train_loader,
            valid_loader=valid_loader,
            device=distributed_context.device,
            output_dir=output_dir,
            distributed_context=distributed_context,
            model_description=model_description,
        )
    finally:
        cleanup_distributed(distributed_context)


if __name__ == "__main__":
    main()
