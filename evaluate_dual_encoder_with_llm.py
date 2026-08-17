from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import math
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import urljoin, urlparse
from urllib.request import urlretrieve
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd
import torch
import websockets

from dual_encoder_retrieval import ProductQueryDualEncoder, build_product_query_dual_encoder_from_tokenizer_json
from train_dual_encoder_on_parquet import (
    FEATURES_COLUMN,
    TITLE_COLUMN,
    clean_text_value,
    detect_description_column,
    flatten_features_level1,
    iter_source_parquet_files,
    normalize_spaces,
    resolve_device,
    select_top_features,
)

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

try:
    from safetensors.torch import load_file as safetensors_load_file

    SAFETENSORS_AVAILABLE = True
except ImportError:
    safetensors_load_file = None
    SAFETENSORS_AVAILABLE = False


JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", flags=re.IGNORECASE | re.DOTALL)
CELL_REF_RE = re.compile(r"([A-Z]+)(\d+)")
XLSX_NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pkgrel": "http://schemas.openxmlformats.org/package/2006/relationships",
}
DEFAULT_SCHEDULER_WS_URL = "ws://10.233.3.148:9000/ws"
DEFAULT_SCHEDULER_MODEL = "Llama-3_3-Nemotron-Super-49B-v1_5"
DEFAULT_GPU_COUNT = 16
DEFAULT_REQUESTS_PER_GPU = 10
DEFAULT_LLM_MAX_IN_FLIGHT = DEFAULT_GPU_COUNT * DEFAULT_REQUESTS_PER_GPU
DEFAULT_LLM_TEMPERATURE = 0.7
DEFAULT_LLM_MAX_TOKENS = 3000
DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS = 600.0
DEFAULT_SCHEDULER_CONNECT_TIMEOUT_SECONDS = 30.0
CATALOG_VECTOR_DIR_NAME = "catalog_vectors"
CATALOG_EMBEDDING_FILE_NAME = "catalog_embeddings.float16.npy"
CATALOG_MANIFEST_FILE_NAME = "catalog_embeddings_manifest.json"
CATALOG_FEATURE_FREQUENCY_FILE_NAME = "catalog_feature_frequency.parquet"


@dataclass(slots=True)
class EvaluationPosition:
    row_index: int
    position_id: str
    name: str
    description: str
    quantity: str
    unit: str

    @property
    def query_text(self) -> str:
        parts = [self.name.strip()]
        if self.description.strip():
            parts.append(self.description.strip())
        return normalize_spaces(". ".join(part for part in parts if part))


@dataclass(slots=True)
class CandidateRecord:
    catalog_index: int
    source_file: str
    source_row_index: int
    title: str
    description: str
    features: list[dict[str, str]]


@dataclass(slots=True)
class SchedulerTaskHandle:
    session_id: str
    done_future: Any

    async def result(self) -> dict[str, Any]:
        return await self.done_future


POSITION_ID_COLUMN_ALIASES = (
    "номер зз",
    "номер заявки",
    "номер позиции",
)
POSITION_NAME_COLUMN_ALIASES = (
    "наименование позиции",
    "наименование",
    "позиция",
)
POSITION_DESCRIPTION_COLUMN_ALIASES = (
    "описание позиции",
    "описание",
    "техническое описание",
)
POSITION_QUANTITY_COLUMN_ALIASES = (
    "количество позиции",
    "количество",
    "кол-во позиции",
)
POSITION_UNIT_COLUMN_ALIASES = (
    "единица измерения",
    "ед измерения",
    "ед изм",
    "единица изм",
)


POSITION_ID_COLUMN_ALIASES = (
    "номер зз",
    "номер заявки",
    "номер позиции",
)
POSITION_NAME_COLUMN_ALIASES = (
    "наименование позиции",
    "наименование",
    "позиция",
)
POSITION_DESCRIPTION_COLUMN_ALIASES = (
    "описание позиции",
    "описание",
    "техническое описание",
)
POSITION_QUANTITY_COLUMN_ALIASES = (
    "количество позиции",
    "количество",
    "кол-во позиции",
)
POSITION_UNIT_COLUMN_ALIASES = (
    "единица измерения",
    "ед измерения",
    "ед изм",
    "единица изм",
)


class SchedulerClient:
    def __init__(
        self,
        ws_url: str,
        model: str,
        path: str = "/v1/completions",
        method: str = "POST",
        connect_timeout_seconds: float = DEFAULT_SCHEDULER_CONNECT_TIMEOUT_SECONDS,
    ) -> None:
        self.ws_url = ws_url
        self.model = model
        self.path = path
        self.method = method
        self.connect_timeout_seconds = max(1.0, float(connect_timeout_seconds))
        self.ws = None
        self.recv_task = None
        self.submit_lock = None
        self.send_lock = None
        self.reconnect_lock = None
        self.connection_generation = 0
        self.awaiting_ack = None
        self.started_events: dict[str, Any] = {}
        self.done_futures: dict[str, Any] = {}
        self.early_msgs: dict[str, list[dict[str, Any]]] = {}

    async def connect(self) -> None:
        self.submit_lock = asyncio.Lock()
        self.send_lock = asyncio.Lock()
        self.reconnect_lock = asyncio.Lock()
        await self._open_connection()

    async def _open_connection(self) -> None:
        self.awaiting_ack = None
        self.started_events = {}
        self.done_futures = {}
        self.early_msgs = {}
        self.ws = await asyncio.wait_for(
            websockets.connect(
                self.ws_url,
                ping_interval=20,
                ping_timeout=20,
                max_size=None,
                open_timeout=self.connect_timeout_seconds,
            ),
            timeout=self.connect_timeout_seconds + 5.0,
        )
        self.connection_generation += 1
        self.recv_task = asyncio.create_task(self.receiver_loop())

    def fail_pending(self, error: BaseException) -> None:
        if self.awaiting_ack and not self.awaiting_ack.done():
            self.awaiting_ack.set_exception(error)
        for future in list(self.done_futures.values()):
            if not future.done():
                future.set_exception(error)

    async def close(self) -> None:
        if self.recv_task:
            self.recv_task.cancel()
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await asyncio.wait_for(self.recv_task, timeout=5.0)
        if self.ws:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self.ws.close(), timeout=5.0)
        self.ws = None
        self.recv_task = None

    async def reconnect(self, reason: str = "scheduler connection closed", observed_generation: int | None = None) -> None:
        if self.reconnect_lock is None:
            self.reconnect_lock = asyncio.Lock()
        async with self.reconnect_lock:
            if observed_generation is not None and observed_generation != self.connection_generation:
                return
            self.fail_pending(RuntimeError(reason))
            if self.recv_task:
                self.recv_task.cancel()
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    await asyncio.wait_for(self.recv_task, timeout=5.0)
            if self.ws:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(self.ws.close(), timeout=5.0)
            self.ws = None
            self.recv_task = None
            await self._open_connection()

    async def ws_send(self, msg: dict[str, Any]) -> None:
        if not self.ws:
            raise RuntimeError("WebSocket is not connected")
        async with self.send_lock:
            await self.ws.send(json.dumps(msg, ensure_ascii=False))

    def handle_session_msg(self, session_id: str, msg: dict[str, Any]) -> None:
        msg_type = msg.get("type")
        if msg_type == "accepted":
            self.started_events.setdefault(session_id, asyncio.Event()).set()
            return
        if msg_type == "rerouted":
            return
        if msg_type == "done":
            future = self.done_futures.get(session_id)
            if future and not future.done():
                future.set_result(msg.get("result", {}))
            return
        if msg_type == "error":
            future = self.done_futures.get(session_id)
            error_text = msg.get("error") or "job_failed"
            if future and not future.done():
                future.set_exception(RuntimeError(str(error_text)))
            return
        if msg_type == "busy":
            return

    async def receiver_loop(self) -> None:
        if not self.ws:
            raise RuntimeError("WebSocket is not connected")
        try:
            async for raw_msg in self.ws:
                msg = json.loads(raw_msg)
                msg_type = msg.get("type")
                if msg_type in {"received", "busy"}:
                    if self.awaiting_ack and not self.awaiting_ack.done():
                        self.awaiting_ack.set_result(msg)
                    continue
                session_id = msg.get("session_id")
                if not session_id:
                    continue
                if msg_type in {"accepted", "rerouted", "done", "error"}:
                    self.handle_session_msg(session_id, msg)
                else:
                    self.early_msgs.setdefault(session_id, []).append(msg)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.fail_pending(RuntimeError(f"Scheduler WebSocket receiver failed: {exc}"))

    async def submit(self, text: str, prompt: str, temperature: float, max_tokens: int) -> SchedulerTaskHandle:
        if not self.ws or (self.recv_task is not None and self.recv_task.done()):
            raise RuntimeError("WebSocket is not connected")
        async with self.submit_lock:
            payload = {
                "model": self.model,
                "prompt": f"{prompt}\n\n{text}\n",
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
                "stream": False,
            }
            request = {
                "type": "request",
                "path": self.path,
                "method": self.method,
                "payload": payload,
            }
            self.awaiting_ack = asyncio.get_running_loop().create_future()
            await self.ws_send(request)
            ack = await self.awaiting_ack
            session_id = ack["session_id"]
            self.started_events.setdefault(session_id, asyncio.Event())
            done_future = self.done_futures.setdefault(session_id, asyncio.get_running_loop().create_future())
            for early_msg in self.early_msgs.pop(session_id, []):
                self.handle_session_msg(session_id, early_msg)
            return SchedulerTaskHandle(session_id=session_id, done_future=done_future)


def unwrap_llm_response_payload(res: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(res, dict):
        return {}
    nested = res.get("result")
    if isinstance(nested, dict) and "choices" not in res:
        return nested
    return res


def extract_completion_text(res: dict[str, Any] | None) -> str:
    payload = unwrap_llm_response_payload(res)
    choices = payload.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            text = choice.get("text")
            if isinstance(text, str) and text:
                return text
            message = choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and content:
                    return content
            delta = choice.get("delta")
            if isinstance(delta, dict):
                content = delta.get("content")
                if isinstance(content, str) and content:
                    return content
        if choices and isinstance(choices[0], dict):
            first_text = choices[0].get("text")
            if isinstance(first_text, str):
                return first_text
    for key in ("output_text", "text", "content"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return ""


def llm_response_debug_summary(res: dict[str, Any] | None) -> dict[str, Any]:
    payload = unwrap_llm_response_payload(res)
    text = extract_completion_text(res)
    choices = payload.get("choices")
    first_choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    return {
        "raw_type": type(res).__name__,
        "raw_keys": sorted(res.keys()) if isinstance(res, dict) else [],
        "payload_keys": sorted(payload.keys()) if isinstance(payload, dict) else [],
        "choice_count": len(choices) if isinstance(choices, list) else 0,
        "first_choice_keys": sorted(first_choice.keys()) if isinstance(first_choice, dict) else [],
        "finish_reason": first_choice.get("finish_reason") if isinstance(first_choice, dict) else None,
        "usage": payload.get("usage") if isinstance(payload, dict) else None,
        "completion_text_length": len(text),
        "completion_text_preview": text[:300],
    }


def extract_final_text(res: dict[str, Any]) -> str:
    text = extract_completion_text(res)
    if not isinstance(text, str):
        return ""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = text.strip()
    text = re.sub(r"^\s*```(?:\w+)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def sanitize_text(value: str) -> str:
    value = value.replace("\ufeff", "")
    value = value.replace("\u200b", "")
    value = value.replace("\u00a0", " ")
    return value.strip()


def extract_json_only(text: str) -> tuple[str, Any]:
    if not isinstance(text, str):
        raise ValueError("text is not str")
    text = sanitize_text(text)
    fenced_match = JSON_FENCE_RE.search(text)
    if fenced_match:
        candidate = sanitize_text(fenced_match.group(1))
        obj = json.loads(candidate)
        return json.dumps(obj, ensure_ascii=False), obj
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            obj, _ = decoder.raw_decode(text[index:])
            return json.dumps(obj, ensure_ascii=False), obj
        except json.JSONDecodeError:
            continue
    raise ValueError("No valid JSON found in model output")


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"}


def download_to_cache(source: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(urlparse(source).path).name or "downloaded_checkpoint"
    target_path = cache_dir / filename
    if target_path.exists():
        return target_path
    urlretrieve(source, target_path)
    return target_path


CHECKPOINT_ARTIFACT_METADATA_SUFFIXES = (
    ".product_encoder.safetensors",
    ".query_encoder.safetensors",
    ".product_encoder.pt",
    ".query_encoder.pt",
    ".model.safetensors",
    ".model.pt",
    ".safetensors",
)


def metadata_name_from_checkpoint_artifact_name(name: str) -> str | None:
    for suffix in CHECKPOINT_ARTIFACT_METADATA_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)] + ".pt"
    return None


def resolve_metadata_source(checkpoint_source: str, cache_dir: Path) -> tuple[Path, str | Path]:
    source = checkpoint_source.strip()
    if is_url(source):
        metadata_source = metadata_name_from_checkpoint_artifact_name(source)
        if metadata_source is None and source.endswith(".pt"):
            metadata_source = source
        if metadata_source is None:
            raise ValueError(
                "Checkpoint URL must point to .pt metadata, .model.pt, .safetensors, "
                ".query_encoder.* or .product_encoder.* artifact"
            )
        local_metadata = download_to_cache(metadata_source, cache_dir)
        return local_metadata, metadata_source

    path = Path(source)
    if path.is_dir():
        if (path / "best_checkpoint.pt").exists():
            return path / "best_checkpoint.pt", path
        if (path / "last_checkpoint.pt").exists():
            return path / "last_checkpoint.pt", path
        raise FileNotFoundError(f"No best_checkpoint.pt or last_checkpoint.pt found in {path}")
    metadata_name = metadata_name_from_checkpoint_artifact_name(path.name)
    if metadata_name is not None:
        return path.with_name(metadata_name), path.parent
    if path.suffix == ".pt":
        return path, path.parent
    raise ValueError(
        "Checkpoint path must be a directory, .pt metadata file, .model.pt, .safetensors, "
        ".query_encoder.* or .product_encoder.* artifact"
    )


def resolve_artifact_reference(
    artifact_reference: str | None,
    *,
    metadata_source: str | Path,
    metadata_local_path: Path,
    cache_dir: Path,
) -> Path | None:
    if not artifact_reference:
        return None
    if is_url(str(metadata_source)):
        artifact_url = urljoin(str(metadata_source), artifact_reference)
        return download_to_cache(artifact_url, cache_dir)
    artifact_path = Path(artifact_reference)
    if artifact_path.is_absolute():
        return artifact_path
    return metadata_local_path.parent / artifact_path


def load_metadata_payload(checkpoint_source: str, cache_dir: Path) -> tuple[dict[str, Any], Path, str | Path]:
    metadata_local_path, metadata_source = resolve_metadata_source(checkpoint_source, cache_dir)
    payload = torch.load(metadata_local_path, map_location="cpu")
    return payload, metadata_local_path, metadata_source


def build_model_kwargs_from_checkpoint_metadata(
    checkpoint_payload: dict[str, Any],
    *,
    tokenizer_path_override: str | None,
) -> tuple[str, dict[str, Any]]:
    checkpoint_args = dict(checkpoint_payload.get("args", {}) or {})
    build_spec = dict(checkpoint_payload.get("build_spec", {}) or {})
    query_model_size = dict(build_spec.get("query_model_size", {}) or {})
    tokenizer_path = (
        tokenizer_path_override
        or checkpoint_args.get("tokenizer_path")
        or build_spec.get("tokenizer_path")
        or "tokenizer (1).json"
    )
    kwargs = {
        "embedding_dim": int(build_spec.get("embedding_dim", checkpoint_args.get("embedding_dim", 1024))),
        "query_target_parameters": int(query_model_size.get("estimated_parameters", checkpoint_args.get("query_target_parameters", 3_000_000_000))),
        "query_num_layers": int(query_model_size.get("num_layers", checkpoint_args.get("query_num_layers", 30))),
        "query_token_type_router_layers": int(query_model_size.get("token_type_router_layers", checkpoint_args.get("query_token_type_router_layers", 4))),
        "query_num_heads": int(query_model_size.get("num_heads", checkpoint_args.get("query_num_heads", 16))),
        "query_ff_multiplier": int(query_model_size.get("ff_multiplier", checkpoint_args.get("query_ff_multiplier", 4))),
        "query_max_position_embeddings": int(query_model_size.get("max_position_embeddings", checkpoint_args.get("query_max_position_embeddings", 2048))),
        "query_dropout": float(checkpoint_args.get("query_dropout", 0.1)),
        "product_d_model": int(build_spec.get("product_d_model", checkpoint_args.get("product_d_model", 1024))),
        "product_num_heads": int(build_spec.get("product_num_heads", checkpoint_args.get("product_num_heads", 16))),
        "product_num_layers": int(build_spec.get("product_num_layers", checkpoint_args.get("product_num_layers", 30))),
        "product_token_type_router_layers": int(build_spec.get("product_token_type_router_layers", checkpoint_args.get("product_token_type_router_layers", 4))),
        "product_ff_multiplier": int(checkpoint_args.get("product_ff_multiplier", 4)),
        "product_max_position_embeddings": int(build_spec.get("product_max_position_embeddings", checkpoint_args.get("product_max_position_embeddings", 8194))),
        "product_max_features": int(build_spec.get("product_max_features", checkpoint_args.get("product_max_features", 14))),
        "product_local_attention_window": int(build_spec.get("product_local_attention_window", checkpoint_args.get("product_local_attention_window", 128))),
        "product_max_routing_clusters": int(build_spec.get("product_max_routing_clusters", checkpoint_args.get("product_max_routing_clusters", 128))),
        "product_dropout": float(checkpoint_args.get("product_dropout", 0.1)),
        "temperature": float(checkpoint_args.get("temperature", 0.05)),
    }
    return str(tokenizer_path), kwargs


def load_model_state_dict_from_checkpoint(
    checkpoint_payload: dict[str, Any],
    *,
    metadata_local_path: Path,
    metadata_source: str | Path,
    cache_dir: Path,
) -> dict[str, Any]:
    model_state_path = resolve_artifact_reference(
        checkpoint_payload.get("model_state_path"),
        metadata_source=metadata_source,
        metadata_local_path=metadata_local_path,
        cache_dir=cache_dir,
    )
    model_state_format = str(checkpoint_payload.get("model_state_format") or "")
    if model_state_path is not None and model_state_path.exists():
        if model_state_format == "safetensors":
            if not SAFETENSORS_AVAILABLE:
                raise RuntimeError("Weights are stored in .safetensors but the safetensors package is not installed.")
            return safetensors_load_file(str(model_state_path), device="cpu")
        return torch.load(model_state_path, map_location="cpu")
    if "model_state_dict" in checkpoint_payload:
        return checkpoint_payload["model_state_dict"]
    raise FileNotFoundError("Model state dict not found in metadata checkpoint or external weight file.")


def load_dual_encoder_for_eval(
    checkpoint_source: str,
    *,
    tokenizer_path_override: str | None,
    device: torch.device,
    cache_dir: Path,
) -> tuple[ProductQueryDualEncoder, dict[str, Any], str]:
    checkpoint_payload, metadata_local_path, metadata_source = load_metadata_payload(checkpoint_source, cache_dir)
    tokenizer_path, model_kwargs = build_model_kwargs_from_checkpoint_metadata(
        checkpoint_payload,
        tokenizer_path_override=tokenizer_path_override,
    )
    model, _build_spec = build_product_query_dual_encoder_from_tokenizer_json(tokenizer_path, **model_kwargs)
    model_state_dict = load_model_state_dict_from_checkpoint(
        checkpoint_payload,
        metadata_local_path=metadata_local_path,
        metadata_source=metadata_source,
        cache_dir=cache_dir,
    )
    model.load_state_dict(model_state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint_payload, tokenizer_path


def parse_list_argument(raw_value: str, *, cast_fn) -> list[Any]:
    values = []
    for part in raw_value.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(cast_fn(part))
    if not values:
        raise ValueError(f"Failed to parse list from {raw_value!r}")
    return values


def parse_optional_list_argument(raw_value: str | None, *, cast_fn) -> list[Any]:
    if raw_value is None:
        return []
    stripped = raw_value.strip()
    if not stripped:
        return []
    return parse_list_argument(stripped, cast_fn=cast_fn)


def read_excel_positions_with_pandas(path: Path) -> dict[str, pd.DataFrame]:
    return pd.read_excel(path, sheet_name=None, engine="openpyxl")


def excel_column_letters_to_index(letters: str) -> int:
    result = 0
    for char in letters:
        if not ("A" <= char <= "Z"):
            continue
        result = result * 26 + (ord(char) - ord("A") + 1)
    return max(0, result - 1)


def extract_xlsx_string_item_text(item: ET.Element) -> str:
    parts: list[str] = []
    for text_node in item.findall(".//main:t", XLSX_NS):
        parts.append(text_node.text or "")
    return "".join(parts)


def load_xlsx_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    shared_strings_path = "xl/sharedStrings.xml"
    if shared_strings_path not in archive.namelist():
        return []
    root = ET.fromstring(archive.read(shared_strings_path))
    return [extract_xlsx_string_item_text(item) for item in root.findall("main:si", XLSX_NS)]


def extract_xlsx_cell_value(cell: ET.Element, shared_strings: Sequence[str]) -> str:
    cell_type = cell.get("t") or ""
    if cell_type == "inlineStr":
        return "".join(text_node.text or "" for text_node in cell.findall(".//main:t", XLSX_NS))
    value_node = cell.find("main:v", XLSX_NS)
    if value_node is None:
        text_node = cell.find(".//main:t", XLSX_NS)
        return text_node.text or "" if text_node is not None else ""
    raw_value = value_node.text or ""
    if cell_type == "s":
        if not raw_value:
            return ""
        index = int(raw_value)
        return shared_strings[index] if 0 <= index < len(shared_strings) else ""
    if cell_type == "b":
        return "1" if raw_value == "1" else "0"
    return raw_value


def worksheet_root_from_archive(archive: zipfile.ZipFile, target_path: str) -> ET.Element:
    normalized = target_path.lstrip("/")
    if not normalized.startswith("xl/"):
        normalized = f"xl/{normalized}"
    return ET.fromstring(archive.read(normalized))


def build_sheet_dataframe_from_xlsx_root(root: ET.Element, shared_strings: Sequence[str]) -> pd.DataFrame:
    rows = root.findall(".//main:sheetData/main:row", XLSX_NS)
    if not rows:
        return pd.DataFrame(dtype=str)

    parsed_rows: list[dict[int, str]] = []
    max_column_index = -1
    for row in rows:
        parsed_row: dict[int, str] = {}
        current_col = 0
        for cell in row.findall("main:c", XLSX_NS):
            cell_ref = cell.get("r") or ""
            match = CELL_REF_RE.match(cell_ref)
            if match is not None:
                current_col = excel_column_letters_to_index(match.group(1))
            value = extract_xlsx_cell_value(cell, shared_strings)
            parsed_row[current_col] = value
            max_column_index = max(max_column_index, current_col)
            current_col += 1
        parsed_rows.append(parsed_row)

    if not parsed_rows or max_column_index < 0:
        return pd.DataFrame(dtype=str)

    header_row = parsed_rows[0]
    headers: list[str] = []
    for column_index in range(max_column_index + 1):
        header_value = clean_text_value(header_row.get(column_index, ""))
        headers.append(header_value if header_value else f"column_{column_index + 1}")

    records: list[dict[str, str]] = []
    for parsed_row in parsed_rows[1:]:
        record = {
            headers[column_index]: clean_text_value(parsed_row.get(column_index, ""))
            for column_index in range(max_column_index + 1)
        }
        if any(value for value in record.values()):
            records.append(record)
    return pd.DataFrame(records, dtype=str).fillna("")


def read_excel_positions_with_xlsx_xml(path: Path) -> dict[str, pd.DataFrame]:
    with zipfile.ZipFile(path) as archive:
        workbook_root = ET.fromstring(archive.read("xl/workbook.xml"))
        workbook_rels_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        workbook_rels = {
            relationship.get("Id"): relationship.get("Target", "")
            for relationship in workbook_rels_root.findall("pkgrel:Relationship", XLSX_NS)
        }
        shared_strings = load_xlsx_shared_strings(archive)

        sheets: dict[str, pd.DataFrame] = {}
        for sheet in workbook_root.findall("main:sheets/main:sheet", XLSX_NS):
            sheet_name = str(sheet.get("name") or "").strip()
            relationship_id = sheet.get(f"{{{XLSX_NS['rel']}}}id")
            target_path = workbook_rels.get(relationship_id, "")
            if not sheet_name or not target_path:
                continue
            sheet_root = worksheet_root_from_archive(archive, target_path)
            sheets[sheet_name] = build_sheet_dataframe_from_xlsx_root(sheet_root, shared_strings)
        return sheets


def read_excel_positions_with_powershell(path: Path) -> dict[str, pd.DataFrame]:
    if os.name != "nt":
        raise RuntimeError("PowerShell Excel COM fallback is only available on Windows.")
    workbook_path = str(path)
    powershell_script = f"""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$excel = New-Object -ComObject Excel.Application
$excel.Visible = $false
$excel.DisplayAlerts = $false
$wb = $excel.Workbooks.Open('{workbook_path.replace("'", "''")}')
$sheets = @()
function Get-MatrixValue($matrix, [int]$row, [int]$col) {{
  if ($null -eq $matrix) {{
    return ''
  }}
  if ($matrix -is [System.Array]) {{
    return $matrix.GetValue($row, $col)
  }}
  if ($row -eq 1 -and $col -eq 1) {{
    return $matrix
  }}
  return ''
}}
try {{
  foreach ($ws in $wb.Worksheets) {{
    $used = $ws.UsedRange
    $rowCount = $used.Rows.Count
    $colCount = $used.Columns.Count
    if ($rowCount -eq 1 -and $colCount -eq 1 -and [string]::IsNullOrWhiteSpace([string]$used.Cells.Item(1,1).Text)) {{
      continue
    }}
    $values = $used.Value2
    $headers = @()
    for ($col = 1; $col -le $colCount; $col++) {{
      $header = [string](Get-MatrixValue $values 1 $col)
      if ([string]::IsNullOrWhiteSpace($header)) {{
        $header = "column_$col"
      }}
      $headers += $header
    }}
    $rows = @()
    for ($row = 2; $row -le $rowCount; $row++) {{
      $record = [ordered]@{{}}
      $hasValue = $false
      for ($col = 1; $col -le $colCount; $col++) {{
        $value = Get-MatrixValue $values $row $col
        $text = if ($null -eq $value) {{ '' }} else {{ [string]$value }}
        if (-not [string]::IsNullOrWhiteSpace($text)) {{
          $hasValue = $true
        }}
        $record[$headers[$col - 1]] = $text
      }}
      if ($hasValue) {{
        $rows += [PSCustomObject]$record
      }}
    }}
    $sheets += [PSCustomObject]@{{ name = [string]$ws.Name; rows = $rows }}
  }}
}}
finally {{
  $wb.Close($false)
  $excel.Quit()
  [System.Runtime.Interopservices.Marshal]::ReleaseComObject($wb) | Out-Null
  [System.Runtime.Interopservices.Marshal]::ReleaseComObject($excel) | Out-Null
}}
$sheets | ConvertTo-Json -Depth 6 -Compress
"""
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-Command", powershell_script],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    raw = completed.stdout.strip()
    if not raw:
        return {}
    payload = json.loads(raw)
    if isinstance(payload, dict):
        payload = [payload]
    sheets: dict[str, pd.DataFrame] = {}
    for item in payload:
        sheet_name = str(item["name"])
        rows = item.get("rows") or []
        sheets[sheet_name] = pd.DataFrame(rows, dtype=str).fillna("")
    return sheets


def read_positions_workbook(path: Path) -> dict[str, pd.DataFrame]:
    try:
        return read_excel_positions_with_pandas(path)
    except Exception:
        if os.name == "nt":
            try:
                return read_excel_positions_with_powershell(path)
            except Exception:
                return read_excel_positions_with_xlsx_xml(path)
        return read_excel_positions_with_xlsx_xml(path)


def normalize_column_key(value: str) -> str:
    normalized = sanitize_text(str(value))
    normalized = normalize_spaces(normalized).lower().replace("ё", "е")
    normalized = normalized.replace("№", " номер ")
    normalized = normalized.replace("кол-во", "количество")
    normalized = normalized.replace("кол во", "количество")
    normalized = re.sub(r"[^0-9a-zа-я]+", " ", normalized, flags=re.IGNORECASE)
    return normalize_spaces(normalized)


def normalize_column_names(columns: Sequence[str]) -> dict[str, str]:
    return {normalize_column_key(str(column)): str(column) for column in columns}


def find_column_by_aliases(column_map: dict[str, str], aliases: Sequence[str], *, required: bool = True) -> str:
    for alias in aliases:
        normalized_alias = normalize_column_key(alias)
        if normalized_alias in column_map:
            return column_map[normalized_alias]
    if required:
        raise KeyError(
            f"Required Excel column not found. Tried aliases: {list(aliases)}. "
            f"Available normalized columns: {list(column_map.keys())}"
        )
    return ""


def pick_positions_sheet(
    workbook: dict[str, pd.DataFrame],
    *,
    sheet_name: str | None = None,
) -> tuple[str, pd.DataFrame]:
    if sheet_name is not None:
        if sheet_name not in workbook:
            raise KeyError(f"Sheet {sheet_name!r} not found. Available sheets: {list(workbook)}")
        return sheet_name, workbook[sheet_name]

    required = {
        normalize_column_key(POSITION_ID_COLUMN_ALIASES[0]),
        normalize_column_key(POSITION_NAME_COLUMN_ALIASES[0]),
        normalize_column_key(POSITION_DESCRIPTION_COLUMN_ALIASES[0]),
    }
    for name, frame in workbook.items():
        normalized = set(normalize_column_names(frame.columns).keys())
        if required.issubset(normalized):
            return name, frame
    for name, frame in workbook.items():
        if not frame.empty:
            return name, frame
    raise ValueError("Workbook does not contain a non-empty sheet.")


def load_evaluation_positions(path: Path, *, sheet_name: str | None = None) -> list[EvaluationPosition]:
    workbook = read_positions_workbook(path)
    _selected_name, frame = pick_positions_sheet(workbook, sheet_name=sheet_name)
    column_map = normalize_column_names(frame.columns)

    id_column = find_column_by_aliases(column_map, POSITION_ID_COLUMN_ALIASES, required=True)
    name_column = find_column_by_aliases(column_map, POSITION_NAME_COLUMN_ALIASES, required=True)
    description_column = find_column_by_aliases(column_map, POSITION_DESCRIPTION_COLUMN_ALIASES, required=True)
    quantity_column = find_column_by_aliases(column_map, POSITION_QUANTITY_COLUMN_ALIASES, required=False)
    unit_column = find_column_by_aliases(column_map, POSITION_UNIT_COLUMN_ALIASES, required=False)

    positions: list[EvaluationPosition] = []
    for row_index, row in enumerate(frame.to_dict("records")):
        name = clean_text_value(row.get(name_column))
        description = clean_text_value(row.get(description_column))
        if not name and not description:
            continue
        positions.append(
            EvaluationPosition(
                row_index=row_index,
                position_id=clean_text_value(row.get(id_column)),
                name=name,
                description=description,
                quantity=clean_text_value(row.get(quantity_column)) if quantity_column else "",
                unit=clean_text_value(row.get(unit_column)) if unit_column else "",
            )
        )
    if not positions:
        raise ValueError("No evaluation positions found in the workbook.")
    return positions


def iter_catalog_products(
    files: Sequence[Path],
    feature_frequency: dict[str, int],
    *,
    max_features: int,
) -> Iterable[tuple[int, dict[str, Any]]]:
    catalog_index = 0
    for path in files:
        frame = pd.read_parquet(path)
        description_column = detect_description_column(frame.columns.tolist())
        for row_index, row in enumerate(frame.to_dict("records")):
            title = clean_text_value(row.get(TITLE_COLUMN))
            if not title:
                continue
            selected_features = select_top_features(
                row.get(FEATURES_COLUMN),
                feature_frequency,
                max_features=max_features,
                title=title,
            )
            product = {
                "title": title,
                "description": clean_text_value(row.get(description_column)) if description_column else "",
                "features": [{"name": feature["name"], "value": feature["value"]} for feature in selected_features],
                "source_file": path.name,
                "source_row_index": row_index,
            }
            yield catalog_index, product
            catalog_index += 1


def build_catalog_feature_frequency(files: Sequence[Path]) -> tuple[dict[str, int], int]:
    frequency: dict[str, int] = {}
    catalog_size = 0
    for path in files:
        frame = pd.read_parquet(path)
        for row in frame.to_dict("records"):
            title = clean_text_value(row.get(TITLE_COLUMN))
            if not title:
                continue
            catalog_size += 1
            seen_names: set[str] = set()
            for feature in flatten_features_level1(row.get(FEATURES_COLUMN)):
                if not feature.norm_name or feature.norm_name in seen_names:
                    continue
                seen_names.add(feature.norm_name)
                frequency[feature.norm_name] = frequency.get(feature.norm_name, 0) + 1
    return frequency, catalog_size


def save_feature_frequency(output_dir: Path, frequency: dict[str, int]) -> None:
    frame = pd.DataFrame(
        sorted(
            ({"feature_name": name, "frequency": count} for name, count in frequency.items()),
            key=lambda item: (-item["frequency"], item["feature_name"]),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_dir / CATALOG_FEATURE_FREQUENCY_FILE_NAME, index=False)


def load_feature_frequency_from_vector_dir(vector_dir: Path) -> dict[str, int] | None:
    frequency_path = vector_dir / CATALOG_FEATURE_FREQUENCY_FILE_NAME
    if not frequency_path.exists():
        return None
    frame = pd.read_parquet(frequency_path)
    if "feature_name" not in frame.columns or "frequency" not in frame.columns:
        raise ValueError(f"Invalid feature frequency file: {frequency_path}")
    return {str(row["feature_name"]): int(row["frequency"]) for row in frame.to_dict("records")}


def build_source_files_manifest(files: Sequence[Path]) -> list[dict[str, Any]]:
    return [
        {
            "name": path.name,
            "size": int(path.stat().st_size),
            "mtime_ns": int(path.stat().st_mtime_ns),
        }
        for path in files
    ]


def build_catalog_manifest(
    *,
    files: Sequence[Path],
    embedding_dim: int,
    catalog_size: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "data_dir": str(args.data_dir),
        "catalog_size": int(catalog_size),
        "embedding_dim": int(embedding_dim),
        "dtype": "float16",
        "max_features": int(args.product_max_features),
        "product_max_length": int(args.product_max_length),
        "tokenizer_path": str(args.tokenizer_path),
        "checkpoint_source": str(args.checkpoint_path),
        "source_files": build_source_files_manifest(files),
    }


def manifest_matches(existing: dict[str, Any], expected: dict[str, Any]) -> bool:
    return existing == expected


def estimate_product_vectorization_cost(
    product: dict[str, Any],
    *,
    max_length: int,
    chars_per_token: float,
) -> int:
    features = product.get("features") or []
    text_parts = [
        str(product.get("title") or ""),
        str(product.get("description") or ""),
    ]
    for feature in features:
        if not isinstance(feature, dict):
            continue
        text_parts.append(str(feature.get("name") or ""))
        text_parts.append(str(feature.get("value") or ""))
    char_count = sum(len(part) for part in text_parts)
    token_estimate = int(math.ceil(char_count / max(1.0, float(chars_per_token))))
    role_overhead = 8 + 4 * len(features)
    return max(8, min(int(max_length), token_estimate + role_overhead))


def is_cuda_out_of_memory(error: BaseException) -> bool:
    cuda_oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if cuda_oom_type is not None and isinstance(error, cuda_oom_type):
        return True
    return "out of memory" in str(error).lower()


def encode_catalog_embeddings(
    model: ProductQueryDualEncoder,
    *,
    tokenizer_path: str,
    files: Sequence[Path],
    feature_frequency: dict[str, int],
    catalog_size: int,
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> Path:
    embedding_path = output_dir / CATALOG_EMBEDDING_FILE_NAME
    manifest_path = output_dir / CATALOG_MANIFEST_FILE_NAME
    expected_manifest = build_catalog_manifest(
        files=files,
        embedding_dim=model.embedding_dim,
        catalog_size=catalog_size,
        args=args,
    )
    if args.reuse_catalog_cache and embedding_path.exists() and manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_matches(existing_manifest, expected_manifest):
            return embedding_path

    output_dir.mkdir(parents=True, exist_ok=True)
    memmap = np.lib.format.open_memmap(
        embedding_path,
        mode="w+",
        dtype=np.float16,
        shape=(catalog_size, model.embedding_dim),
    )
    use_amp = device.type == "cuda"
    batch_products: list[dict[str, Any]] = []
    batch_indices: list[int] = []
    batch_cost = 0
    dynamic_batching = bool(getattr(args, "dynamic_catalog_batching", True))
    max_batch_size = max(1, int(getattr(args, "catalog_batch_size", 8)))
    token_budget_arg = getattr(args, "catalog_token_budget", None)
    token_budget = int(token_budget_arg) if token_budget_arg else max_batch_size * int(args.product_max_length)
    chars_per_token = float(getattr(args, "catalog_chars_per_token", 4.0))
    progress = tqdm(total=catalog_size, desc="Encode catalog", unit="item", dynamic_ncols=True) if tqdm else None

    def encode_batch(products: Sequence[dict[str, Any]], indices: Sequence[int]) -> None:
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
                memmap[catalog_index] = embeddings[row_offset]
        except RuntimeError as error:
            if len(products) <= 1 or not is_cuda_out_of_memory(error):
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
            midpoint = max(1, len(products) // 2)
            if progress is not None:
                progress.write(f"CUDA OOM while encoding batch_size={len(products)}; retrying as {midpoint}+{len(products) - midpoint}")
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

    for catalog_index, product in iter_catalog_products(files, feature_frequency, max_features=args.product_max_features):
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
    manifest_path.write_text(json.dumps(expected_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return embedding_path


def encode_query_embeddings(
    model: ProductQueryDualEncoder,
    *,
    tokenizer_path: str,
    positions: Sequence[EvaluationPosition],
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    use_amp = device.type == "cuda"
    vectors: list[np.ndarray] = []
    progress = tqdm(total=len(positions), desc="Encode queries", unit="query", dynamic_ncols=True) if tqdm else None
    for start in range(0, len(positions), args.query_batch_size):
        batch = positions[start : start + args.query_batch_size]
        texts = [position.query_text for position in batch]
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                output = model.query_encoder.encode_texts_from_tokenizer_json(
                    texts=texts,
                    tokenizer_path=tokenizer_path,
                    max_length=args.query_max_length,
                    device=device,
                )
        vectors.append(output.normalized_embedding.detach().float().cpu().numpy())
        if progress is not None:
            progress.update(len(batch))
    if progress is not None:
        progress.close()
    return np.concatenate(vectors, axis=0)


def retrieve_top_candidates(
    query_embeddings: np.ndarray,
    embedding_path: Path,
    *,
    top_n: int,
    chunk_size: int,
    query_batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    catalog_embeddings = np.load(embedding_path, mmap_mode="r")
    query_count, embedding_dim = query_embeddings.shape
    top_scores = np.full((query_count, top_n), -np.inf, dtype=np.float32)
    top_indices = np.full((query_count, top_n), -1, dtype=np.int64)
    logsumexp_values = np.full(query_count, -np.inf, dtype=np.float64)
    use_cuda = device.type == "cuda"

    query_progress = tqdm(total=query_count, desc="Retrieve", unit="query", dynamic_ncols=True) if tqdm else None
    for query_start in range(0, query_count, query_batch_size):
        query_end = min(query_start + query_batch_size, query_count)
        query_batch = torch.from_numpy(query_embeddings[query_start:query_end]).to(device=device, dtype=torch.float32)
        batch_top_scores = torch.full((query_batch.size(0), top_n), -torch.inf, device=device, dtype=torch.float32)
        batch_top_indices = torch.full((query_batch.size(0), top_n), -1, device=device, dtype=torch.long)
        running_max = torch.full((query_batch.size(0),), -torch.inf, device=device, dtype=torch.float32)
        running_sum = torch.zeros((query_batch.size(0),), device=device, dtype=torch.float64)

        for catalog_start in range(0, catalog_embeddings.shape[0], chunk_size):
            catalog_chunk = torch.from_numpy(np.asarray(catalog_embeddings[catalog_start : catalog_start + chunk_size], dtype=np.float32)).to(device=device)
            scores = query_batch @ catalog_chunk.transpose(0, 1)
            chunk_max = scores.max(dim=1).values
            new_max = torch.maximum(running_max, chunk_max)
            running_sum = (
                torch.exp((running_max - new_max).to(torch.float64)) * running_sum
                + torch.exp((scores.to(torch.float64) - new_max.unsqueeze(1).to(torch.float64))).sum(dim=1)
            )
            running_max = new_max

            chunk_top_scores, chunk_top_local_indices = torch.topk(scores, k=min(top_n, scores.size(1)), dim=1)
            chunk_top_indices = chunk_top_local_indices + catalog_start

            merged_scores = torch.cat((batch_top_scores, chunk_top_scores), dim=1)
            merged_indices = torch.cat((batch_top_indices, chunk_top_indices), dim=1)
            new_scores, merged_positions = torch.topk(merged_scores, k=top_n, dim=1)
            batch_top_indices = torch.gather(merged_indices, 1, merged_positions)
            batch_top_scores = new_scores

            if use_cuda:
                del catalog_chunk, scores, chunk_top_scores, chunk_top_local_indices, chunk_top_indices, merged_scores, merged_indices, new_scores, merged_positions
                torch.cuda.empty_cache()

        batch_logsumexp = running_max.to(torch.float64) + torch.log(running_sum.clamp_min(1e-40))
        top_scores[query_start:query_end] = batch_top_scores.detach().cpu().numpy()
        top_indices[query_start:query_end] = batch_top_indices.detach().cpu().numpy()
        logsumexp_values[query_start:query_end] = batch_logsumexp.detach().cpu().numpy()
        if query_progress is not None:
            query_progress.update(query_end - query_start)
    if query_progress is not None:
        query_progress.close()
    return top_scores, top_indices, logsumexp_values


def materialize_candidates_by_index(
    files: Sequence[Path],
    target_indices: set[int],
    feature_frequency: dict[str, int],
    *,
    max_features: int,
) -> dict[int, CandidateRecord]:
    materialized: dict[int, CandidateRecord] = {}
    if not target_indices:
        return materialized
    needed_sorted = sorted(target_indices)
    needed_pointer = 0
    next_needed = needed_sorted[needed_pointer]
    catalog_index = 0
    for path in files:
        frame = pd.read_parquet(path)
        description_column = detect_description_column(frame.columns.tolist())
        for row_index, row in enumerate(frame.to_dict("records")):
            title = clean_text_value(row.get(TITLE_COLUMN))
            if not title:
                continue
            if catalog_index == next_needed:
                selected_features = select_top_features(
                    row.get(FEATURES_COLUMN),
                    feature_frequency,
                    max_features=max_features,
                    title=title,
                )
                materialized[catalog_index] = CandidateRecord(
                    catalog_index=catalog_index,
                    source_file=path.name,
                    source_row_index=row_index,
                    title=title,
                    description=clean_text_value(row.get(description_column)) if description_column else "",
                    features=[{"name": feature["name"], "value": feature["value"]} for feature in selected_features],
                )
                needed_pointer += 1
                if needed_pointer >= len(needed_sorted):
                    return materialized
                next_needed = needed_sorted[needed_pointer]
            catalog_index += 1
    return materialized


def softmax_probs_from_top_scores(top_scores: np.ndarray, logsumexp_values: np.ndarray) -> np.ndarray:
    return np.exp(top_scores.astype(np.float64) - logsumexp_values[:, None]).astype(np.float64)


def prefix_size_from_top_p(probabilities: np.ndarray, threshold: float) -> tuple[int, bool]:
    cumulative = np.cumsum(probabilities)
    if cumulative.size == 0:
        return 0, False
    index = int(np.searchsorted(cumulative, threshold, side="left"))
    if index >= cumulative.size:
        return int(cumulative.size), False
    return index + 1, True


def prefix_size_from_score_threshold(scores: np.ndarray, threshold: float) -> int:
    if scores.size == 0:
        return 0
    return int(np.count_nonzero(scores >= threshold))


def prefix_size_from_probability_threshold(probabilities: np.ndarray, threshold: float) -> int:
    if probabilities.size == 0:
        return 0
    return int(np.count_nonzero(probabilities >= threshold))


def compute_binary_metrics(labels: Sequence[int], cutoff: int) -> dict[str, float]:
    effective_cutoff = max(0, min(int(cutoff), len(labels)))
    if effective_cutoff == 0:
        return {
            "k": 0.0,
            "positives_total": float(sum(labels)),
            "precision": 0.0,
            "recall": 0.0,
            "hit_rate": 0.0,
            "mrr": 0.0,
            "rmm": 0.0,
            "map": 0.0,
            "ndcg": 0.0,
            "ngcd": 0.0,
        }
    labels_prefix = [int(value) for value in labels[:effective_cutoff]]
    total_positives = int(sum(int(value) for value in labels))
    prefix_positives = int(sum(labels_prefix))

    precision = prefix_positives / effective_cutoff
    recall = prefix_positives / total_positives if total_positives > 0 else 0.0
    hit_rate = 1.0 if prefix_positives > 0 else 0.0

    mrr = 0.0
    for index, label in enumerate(labels_prefix, start=1):
        if label > 0:
            mrr = 1.0 / index
            break

    precision_sum = 0.0
    seen_positives = 0
    for index, label in enumerate(labels_prefix, start=1):
        if label > 0:
            seen_positives += 1
            precision_sum += seen_positives / index
    average_precision = precision_sum / total_positives if total_positives > 0 else 0.0

    dcg = 0.0
    for index, label in enumerate(labels_prefix, start=1):
        if label > 0:
            dcg += 1.0 / math.log2(index + 1)
    ideal_positives = min(total_positives, effective_cutoff)
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_positives + 1))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return {
        "k": float(effective_cutoff),
        "positives_total": float(total_positives),
        "precision": float(precision),
        "recall": float(recall),
        "hit_rate": float(hit_rate),
        "mrr": float(mrr),
        "rmm": float(mrr),
        "map": float(average_precision),
        "ndcg": float(ndcg),
        "ngcd": float(ndcg),
    }


def aggregate_metric_dicts(metric_dicts: Sequence[dict[str, float]]) -> dict[str, float]:
    if not metric_dicts:
        return {}
    keys = metric_dicts[0].keys()
    return {key: float(sum(item[key] for item in metric_dicts) / len(metric_dicts)) for key in keys}


RELEVANCE_RESPONSE_SCHEMA = '{"labels":[{"rank":1,"relevant":1,"reason":"..."},{"rank":2,"relevant":0,"reason":"..."}]}'
FURNITURE_RESPONSE_SCHEMA = '{"labels":[{"query_index":0,"is_furniture":1,"reason":"..."},{"query_index":1,"is_furniture":0,"reason":"..."}]}'
RELEVANCE_SCHEDULER_PROMPT = "\n".join(
    [
        "Ты эксперт по закупкам и техническому сопоставлению товаров.",
        "Твоя задача: для каждой карточки товара определить, полностью ли она подходит под закупочную позицию.",
        "",
        "Формат ответа:",
        f"- Верни строго один JSON-объект формата: {RELEVANCE_RESPONSE_SCHEMA}",
        "- Не используй markdown, кодовые блоки, комментарии или текст вокруг JSON.",
        "- В labels должен быть один объект на каждый rank из входных данных.",
        "- Не пропускай rank и не добавляй rank, которых нет во входных данных.",
        "- relevant должен быть только 0 или 1.",
        "- reason должен быть короткой причиной решения на русском, 3-12 слов.",
        "",
        "Как ставить relevant:",
        "- relevant = 1 только если карточка товара является полным соответствием закупочной позиции или допустимым эквивалентом.",
        "- relevant = 0 если товар не подходит, подходит частично, не хватает доказательств, есть противоречие или есть сомнение.",
        "",
        "Правила сравнения:",
        "1. Сравни закупочную позицию с каждой карточкой отдельно.",
        "2. Учитывай наименование, описание, количество, единицу измерения и все характеристики карточки.",
        "3. Сначала определи тип товара и назначение. Если тип или назначение отличаются, ставь 0.",
        "4. Если в позиции есть 'или аналог', 'или эквивалент', 'или аналогичный', допускай другой бренд или модель, но только при совпадении назначения и всех критичных характеристик.",
        "5. Если конкретная модель, артикул или бренд указаны без разрешения аналога, считай их обязательными.",
        "6. Критичные характеристики: материал, размеры, форма-фактор, цвет при явном требовании, мощность, напряжение, ток, емкость, диапазоны, точность, класс, стандарт, ГОСТ, совместимость, комплектность, количество предметов в наборе.",
        "7. Числовые требования сравни строго: 'не менее', 'не более', 'от/до', диапазоны, допуски, размеры и единицы измерения.",
        "8. Если карточка не подтверждает критичную характеристику, ставь 0. Нельзя додумывать свойства товара.",
        "9. Если совпадает только общий род товара, но расходятся важные параметры, ставь 0.",
        "10. Если позиция описывает комплект или набор, все обязательные части должны быть подтверждены.",
        "11. Игнорируй цену, скидки, продавца, доставку, рейтинг, внутренние ID и служебные поля, если они не требуются в позиции.",
        "12. При конфликте между title и features карточки ориентируйся на более конкретное техническое поле; при сомнении ставь 0.",
    ]
)
FURNITURE_SCHEDULER_PROMPT = "\n".join(
    [
        "Ты классифицируешь закупочные позиции на мебель и не мебель.",
        "",
        "Формат ответа:",
        f"- Верни строго один JSON-объект формата: {FURNITURE_RESPONSE_SCHEMA}",
        "- Не используй markdown, кодовые блоки, комментарии или текст вокруг JSON.",
        "- В labels должен быть один объект на каждый query_index из входных данных.",
        "- Не пропускай query_index и не добавляй query_index, которых нет во входных данных.",
        "- is_furniture должен быть только 0 или 1.",
        "- reason должен быть короткой причиной решения на русском, 3-12 слов.",
        "",
        "Как ставить is_furniture:",
        "- is_furniture = 1 только если позиция описывает готовый предмет мебели или законченный предмет меблировки для размещения в помещении.",
        "- is_furniture = 0 для инструмента, техники, электроники, расходников, сырья, стройматериалов, фурнитуры, запчастей, комплектующих, декора, светильников и любых немебельных объектов.",
        "",
        "Примеры мебели: стол, стул, кресло, диван, кровать, шкаф, тумба, комод, стеллаж, полка, стойка, гарнитур, кухонный модуль, офисная мебель.",
        "Не мебель: столешница отдельно, фасад, крепеж, направляющая, петля, ручка, мебельная фурнитура, материал для изготовления мебели, инструмент, мультиметр, аккумулятор, кабель, лампа.",
        "Если есть сомнение, ставь 0.",
    ]
)


RELEVANCE_RESPONSE_SCHEMA = '{"labels":[{"rank":1,"relevant":1,"reason":"..."},{"rank":2,"relevant":0,"reason":"..."}]}'
FURNITURE_RESPONSE_SCHEMA = '{"labels":[{"query_index":0,"is_furniture":1,"reason":"..."},{"query_index":1,"is_furniture":0,"reason":"..."}]}'
RELEVANCE_SCHEDULER_PROMPT = "\n".join(
    [
        "Ты эксперт по закупкам и техническому сопоставлению товаров.",
        "Твоя задача: для каждой карточки товара определить, полностью ли она подходит под закупочную позицию.",
        "",
        "Формат ответа:",
        f"- Верни строго один JSON-объект формата: {RELEVANCE_RESPONSE_SCHEMA}",
        "- Не используй markdown, кодовые блоки, комментарии или текст вокруг JSON.",
        "- В labels должен быть один объект на каждый rank из входных данных.",
        "- Не пропускай rank и не добавляй rank, которых нет во входных данных.",
        "- relevant должен быть только 0 или 1.",
        "- reason должен быть короткой причиной решения на русском, 3-12 слов.",
        "",
        "Как ставить relevant:",
        "- relevant = 1 только если карточка товара является полным соответствием закупочной позиции или допустимым эквивалентом.",
        "- relevant = 0 если товар не подходит, подходит частично, не хватает доказательств, есть противоречие или есть сомнение.",
        "",
        "Правила сравнения:",
        "1. Сравни закупочную позицию с каждой карточкой отдельно.",
        "2. Учитывай наименование, описание, количество, единицу измерения и все характеристики карточки.",
        "3. Сначала определи тип товара и назначение. Если тип или назначение отличаются, ставь 0.",
        "4. Если в позиции есть 'или аналог', 'или эквивалент', допускай другой бренд или модель, но только при совпадении назначения и всех критичных характеристик.",
        "5. Если конкретная модель, артикул или бренд указаны без разрешения аналога, считай их обязательными.",
        "6. Критичные характеристики: материал, размеры, форма-фактор, цвет при явном требовании, мощность, напряжение, ток, емкость, диапазоны, точность, класс, стандарт, ГОСТ, совместимость, комплектность, количество предметов в наборе.",
        "7. Числовые требования сравни строго: 'не менее', 'не более', 'от/до', диапазоны, допуски, размеры и единицы измерения.",
        "8. Если карточка не подтверждает критичную характеристику, ставь 0. Нельзя додумывать свойства товара.",
        "9. Если совпадает только общий род товара, но расходятся важные параметры, ставь 0.",
        "10. Если позиция описывает комплект или набор, все обязательные части должны быть подтверждены.",
        "11. Игнорируй цену, скидки, продавца, доставку, рейтинг, внутренние ID и служебные поля, если они не требуются в позиции.",
        "12. При конфликте между title и features карточки ориентируйся на более конкретное техническое поле; при сомнении ставь 0.",
    ]
)
FURNITURE_SCHEDULER_PROMPT = "\n".join(
    [
        "Ты классифицируешь закупочные позиции на мебель и не мебель.",
        "",
        "Формат ответа:",
        f"- Верни строго один JSON-объект формата: {FURNITURE_RESPONSE_SCHEMA}",
        "- Не используй markdown, кодовые блоки, комментарии или текст вокруг JSON.",
        "- В labels должен быть один объект на каждый query_index из входных данных.",
        "- Не пропускай query_index и не добавляй query_index, которых нет во входных данных.",
        "- is_furniture должен быть только 0 или 1.",
        "- reason должен быть короткой причиной решения на русском, 3-12 слов.",
        "",
        "Как ставить is_furniture:",
        "- is_furniture = 1 только если позиция описывает готовый предмет мебели или законченный предмет меблировки для размещения в помещении.",
        "- is_furniture = 0 для инструмента, техники, электроники, расходников, сырья, стройматериалов, фурнитуры, запчастей, комплектующих, декора, светильников и любых немебельных объектов.",
        "",
        "Примеры мебели: стол, стул, кресло, диван, кровать, шкаф, тумба, комод, стеллаж, полка, стойка, гарнитур, кухонный модуль, офисная мебель.",
        "Не мебель: столешница отдельно, фасад, крепеж, направляющая, петля, ручка, мебельная фурнитура, материал для изготовления мебели, инструмент, мультиметр, аккумулятор, кабель, лампа.",
        "Если есть сомнение, ставь 0.",
    ]
)


def build_relevance_prompt_template() -> str:
    return "\n".join(
        [
            RELEVANCE_SCHEDULER_PROMPT,
            "",
            "Входные данные:",
            "Закупочная позиция:",
            "- Номер ЗЗ: {position_id}",
            "- Наименование: {name}",
            "- Описание: {description}",
            "- Количество: {quantity} {unit}",
            "",
            "Карточки товаров:",
            "{rank}. title: {title}",
            "   description: {candidate_description}",
            "   features:",
            "   - {feature_name}: {feature_value}",
        ]
    )


def build_furniture_prompt_template() -> str:
    return "\n".join(
        [
            FURNITURE_SCHEDULER_PROMPT,
            "",
            "Входные данные:",
            "Позиции:",
            "- query_index: {query_index}",
            "- Номер ЗЗ: {position_id}",
            "- Наименование: {name}",
            "- Описание: {description}",
            "- Количество: {quantity} {unit}",
        ]
    )


def write_prompt_artifacts(output_dir: Path) -> None:
    prompts_dir = output_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    prompt_files = {
        "relevance_scheduler_prompt.txt": RELEVANCE_SCHEDULER_PROMPT,
        "relevance_user_prompt_template.txt": build_relevance_prompt_template(),
        "furniture_scheduler_prompt.txt": FURNITURE_SCHEDULER_PROMPT,
        "furniture_user_prompt_template.txt": build_furniture_prompt_template(),
    }
    for file_name, content in prompt_files.items():
        (prompts_dir / file_name).write_text(content, encoding="utf-8")
    (prompts_dir / "prompt_manifest.json").write_text(
        json.dumps(
            {
                "relevance_scheduler_prompt": RELEVANCE_SCHEDULER_PROMPT,
                "relevance_response_schema": RELEVANCE_RESPONSE_SCHEMA,
                "furniture_scheduler_prompt": FURNITURE_SCHEDULER_PROMPT,
                "furniture_response_schema": FURNITURE_RESPONSE_SCHEMA,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def format_features_for_prompt(features: Sequence[dict[str, str]]) -> str:
    if not features:
        return "нет характеристик"
    return "\n".join(f"- {feature['name']}: {feature['value']}" for feature in features)


def build_llm_judgement_prompt_body(
    position: EvaluationPosition,
    candidates: Sequence[tuple[int, CandidateRecord]],
) -> str:
    lines = [
        "Входные данные для сравнения.",
        "Закупочная позиция:",
        f"- Номер ЗЗ: {position.position_id}",
        f"- Наименование: {position.name}",
        f"- Описание: {position.description or 'нет описания'}",
    ]
    if position.quantity or position.unit:
        lines.append(f"- Количество: {position.quantity} {position.unit}".strip())
    lines.append("")
    lines.append("Карточки товаров:")
    for rank, candidate in candidates:
        lines.extend(
            [
                f"{rank}. title: {candidate.title}",
                f"   description: {candidate.description or 'нет описания'}",
                "   features:",
                *[f"   {line}" for line in format_features_for_prompt(candidate.features).splitlines()],
            ]
        )
    return "\n".join(lines)


def build_furniture_classification_prompt_body(
    positions_batch: Sequence[tuple[int, EvaluationPosition]],
) -> str:
    lines = [
        "Входные данные для классификации.",
        "Позиции:",
    ]
    for query_index, position in positions_batch:
        lines.extend(
            [
                f"- query_index: {query_index}",
                f"  Номер ЗЗ: {position.position_id}",
                f"  Наименование: {position.name}",
                f"  Описание: {position.description or 'нет описания'}",
                f"  Количество: {(f'{position.quantity} {position.unit}').strip()}",
            ]
        )
    return "\n".join(lines)


def format_features_for_prompt(features: Sequence[dict[str, str]]) -> str:
    if not features:
        return "нет характеристик"
    return "\n".join(f"- {feature['name']}: {feature['value']}" for feature in features)


def build_llm_judgement_prompt_body(
    position: EvaluationPosition,
    candidates: Sequence[tuple[int, CandidateRecord]],
) -> str:
    lines = [
        "Входные данные для сравнения.",
        "Закупочная позиция:",
        f"- Номер ЗЗ: {position.position_id}",
        f"- Наименование: {position.name}",
        f"- Описание: {position.description or 'нет описания'}",
    ]
    if position.quantity or position.unit:
        lines.append(f"- Количество: {position.quantity} {position.unit}".strip())
    lines.append("")
    lines.append("Карточки товаров:")
    for rank, candidate in candidates:
        lines.extend(
            [
                f"{rank}. title: {candidate.title}",
                f"   description: {candidate.description or 'нет описания'}",
                "   features:",
                *[f"   {line}" for line in format_features_for_prompt(candidate.features).splitlines()],
            ]
        )
    return "\n".join(lines)


def build_furniture_classification_prompt_body(
    positions_batch: Sequence[tuple[int, EvaluationPosition]],
) -> str:
    lines = [
        "Входные данные для классификации.",
        "Позиции:",
    ]
    for query_index, position in positions_batch:
        lines.extend(
            [
                f"- query_index: {query_index}",
                f"  Номер ЗЗ: {position.position_id}",
                f"  Наименование: {position.name}",
                f"  Описание: {position.description or 'нет описания'}",
                f"  Количество: {(f'{position.quantity} {position.unit}').strip()}",
            ]
        )
    return "\n".join(lines)


def parse_llm_binary_labels(raw_res: dict[str, Any], *, expected_ranks: Sequence[int]) -> dict[int, int]:
    _, obj = extract_json_only(extract_final_text(raw_res))
    if not isinstance(obj, dict) or "labels" not in obj or not isinstance(obj["labels"], list):
        raise ValueError("LLM response must be a JSON object with a labels list.")
    labels: dict[int, int] = {}
    for item in obj["labels"]:
        if not isinstance(item, dict):
            raise ValueError("Each label item must be a JSON object.")
        rank = int(item.get("rank"))
        relevant = int(item.get("relevant"))
        if relevant not in {0, 1}:
            raise ValueError("relevant must be 0 or 1.")
        labels[rank] = relevant
    missing = [rank for rank in expected_ranks if rank not in labels]
    if missing:
        raise ValueError(f"LLM response is missing labels for ranks {missing}.")
    return {rank: labels[rank] for rank in expected_ranks}


def parse_llm_furniture_labels(raw_res: dict[str, Any], *, expected_query_indices: Sequence[int]) -> dict[int, int]:
    _, obj = extract_json_only(extract_final_text(raw_res))
    if not isinstance(obj, dict) or "labels" not in obj or not isinstance(obj["labels"], list):
        raise ValueError("LLM furniture response must be a JSON object with a labels list.")
    labels: dict[int, int] = {}
    for item in obj["labels"]:
        if not isinstance(item, dict):
            raise ValueError("Each furniture label item must be a JSON object.")
        query_index = int(item.get("query_index"))
        is_furniture = int(item.get("is_furniture"))
        if is_furniture not in {0, 1}:
            raise ValueError("is_furniture must be 0 or 1.")
        labels[query_index] = is_furniture
    missing = [query_index for query_index in expected_query_indices if query_index not in labels]
    if missing:
        raise ValueError(f"LLM furniture response is missing labels for query_index values {missing}.")
    return {query_index: labels[query_index] for query_index in expected_query_indices}


def raw_llm_text(raw_res: dict[str, Any] | None) -> str:
    if raw_res is None:
        return ""
    try:
        return extract_final_text(raw_res)
    except Exception:
        return str(raw_res)


def raw_llm_completion_text(raw_res: dict[str, Any] | None) -> str:
    if raw_res is None:
        return ""
    try:
        return extract_completion_text(raw_res)
    except Exception:
        return str(raw_res)


def is_scheduler_connection_error(error: BaseException) -> bool:
    if isinstance(error, websockets.exceptions.ConnectionClosed):
        return True
    text = str(error).lower()
    return any(
        marker in text
        for marker in (
            "websocket is not connected",
            "connection closed",
            "connection reset",
            "scheduler websocket receiver failed",
            "no close frame received",
            "sent 1011",
            "received 1011",
        )
    )


async def submit_scheduler_job_with_retries(
    *,
    client: SchedulerClient,
    text: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    parse_fn,
    expected_values: Sequence[int],
    max_attempts: int = 3,
    request_timeout_seconds: float | None = DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS,
) -> tuple[dict[int, int] | None, list[dict[str, Any]], str | None]:
    attempts: list[dict[str, Any]] = []
    current_prompt = prompt
    max_attempts = max(1, int(max_attempts))
    timeout = float(request_timeout_seconds) if request_timeout_seconds and float(request_timeout_seconds) > 0 else None
    for attempt_index in range(1, max_attempts + 1):
        raw_res: dict[str, Any] | None = None
        handle: SchedulerTaskHandle | None = None
        observed_generation = client.connection_generation
        try:
            submit_coro = client.submit(
                text=text,
                prompt=current_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            handle = await asyncio.wait_for(submit_coro, timeout=timeout) if timeout else await submit_coro
            result_coro = handle.result()
            raw_res = await asyncio.wait_for(result_coro, timeout=timeout) if timeout else await result_coro
            parsed = parse_fn(raw_res, expected_values)
            attempts.append(
                {
                    "attempt": attempt_index,
                    "ok": True,
                    "scheduler_prompt": current_prompt,
                    "raw_response": raw_res,
                    "raw_response_debug": llm_response_debug_summary(raw_res),
                    "raw_completion_text": raw_llm_completion_text(raw_res),
                    "raw_text": raw_llm_text(raw_res),
                    "error": None,
                }
            )
            return parsed, attempts, None
        except asyncio.TimeoutError:
            timeout_text = f"{timeout:.1f}" if timeout is not None else "unknown"
            error_text = f"scheduler request timed out after {timeout_text} seconds"
            if handle is not None and not handle.done_future.done():
                handle.done_future.cancel()
            attempts.append(
                {
                    "attempt": attempt_index,
                    "ok": False,
                    "scheduler_prompt": current_prompt,
                    "raw_response": raw_res,
                    "raw_response_debug": llm_response_debug_summary(raw_res),
                    "raw_completion_text": raw_llm_completion_text(raw_res),
                    "raw_text": raw_llm_text(raw_res),
                    "error": error_text,
                }
            )
            if attempt_index < max_attempts:
                with contextlib.suppress(Exception):
                    await client.reconnect(error_text, observed_generation=observed_generation)
                await asyncio.sleep(min(2.0 * attempt_index, 10.0))
            current_prompt = (
                f"{prompt}\n"
                "Предыдущий ответ не удалось получить или распарсить как JSON. "
                "Верни строго один JSON-объект без markdown, комментариев и текста вокруг."
            )
        except Exception as exc:
            error_text = str(exc)
            attempts.append(
                {
                    "attempt": attempt_index,
                    "ok": False,
                    "scheduler_prompt": current_prompt,
                    "raw_response": raw_res,
                    "raw_response_debug": llm_response_debug_summary(raw_res),
                    "raw_completion_text": raw_llm_completion_text(raw_res),
                    "raw_text": raw_llm_text(raw_res),
                    "error": error_text,
                }
            )
            current_prompt = (
                f"{prompt}\n"
                "Предыдущий ответ не удалось распарсить как JSON. "
                "Верни строго один JSON-объект без markdown, комментариев и текста вокруг."
            )
            if attempt_index < max_attempts and is_scheduler_connection_error(exc):
                with contextlib.suppress(Exception):
                    await client.reconnect(error_text, observed_generation=observed_generation)
                await asyncio.sleep(min(2.0 * attempt_index, 10.0))
    return None, attempts, attempts[-1]["error"] if attempts else "unknown_error"


def default_zero_labels(expected_values: Sequence[int]) -> dict[int, int]:
    return {int(value): 0 for value in expected_values}


LLM_LABEL_CACHE_VERSION = "utf8_prompts_v2"


def compact_llm_attempts_for_log(
    attempts: Sequence[dict[str, Any]],
    *,
    include_raw_responses: bool,
) -> list[dict[str, Any]]:
    compact_attempts: list[dict[str, Any]] = []
    for attempt in attempts:
        item = {
            "attempt": int(attempt.get("attempt", 0)),
            "ok": bool(attempt.get("ok", False)),
            "error": attempt.get("error"),
            "raw_response_debug": attempt.get("raw_response_debug"),
        }
        if include_raw_responses:
            item["scheduler_prompt"] = attempt.get("scheduler_prompt")
            item["raw_response"] = attempt.get("raw_response")
            item["raw_completion_text"] = attempt.get("raw_completion_text")
            item["raw_text"] = attempt.get("raw_text")
        else:
            raw_text = str(attempt.get("raw_text") or "")
            if raw_text:
                item["raw_text_preview"] = raw_text[:500]
        compact_attempts.append(item)
    return compact_attempts


def llm_answer_attempts_for_export(attempts: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    exported: list[dict[str, Any]] = []
    for attempt in attempts:
        exported.append(
            {
                "attempt": int(attempt.get("attempt", 0)),
                "ok": bool(attempt.get("ok", False)),
                "answer_text": str(attempt.get("raw_completion_text") or ""),
                "raw_response_debug": attempt.get("raw_response_debug"),
                "error": attempt.get("error"),
            }
        )
    return exported


def write_procurement_expert_answer_log(
    handle,
    *,
    job: dict[str, Any],
    position: EvaluationPosition,
    attempts: Sequence[dict[str, Any]],
    parsed_labels: dict[int, int],
    error_text: str | None,
) -> None:
    for attempt in attempts:
        handle.write(
            "\n".join(
                [
                    "=" * 100,
                    "mode=procurement_expert_relevance",
                    f"query_index={job['query_index']}",
                    f"position_id={position.position_id}",
                    f"position_name={position.name}",
                    f"chunk_start={job['chunk_start']}",
                    f"expected_ranks={job['expected_ranks']}",
                    f"candidate_count={job.get('candidate_count')}",
                    f"prompt_chars={job.get('prompt_chars')}",
                    f"prompt_body_chars={job.get('prompt_body_chars')}",
                    f"full_prompt_chars={job.get('full_prompt_chars')}",
                    f"attempt={int(attempt.get('attempt', 0))}",
                    f"ok={bool(attempt.get('ok', False))}",
                    f"error={attempt.get('error')}",
                    f"final_error={error_text}",
                    f"parsed_labels={json.dumps(parsed_labels, ensure_ascii=False)}",
                    f"raw_response_debug={json.dumps(attempt.get('raw_response_debug') or {}, ensure_ascii=False)}",
                    "-" * 100,
                ]
            )
        )
        handle.write("\n")
        handle.write(str(attempt.get("raw_completion_text") or ""))
        handle.write("\n")
    handle.flush()


def write_procurement_expert_cached_labels_log(
    output_dir: Path,
    *,
    model_name: str,
    labels_per_query: Sequence[Sequence[int]],
) -> Path:
    log_path = output_dir / "procurement_expert_llm_answers.log"
    positive_label_count = int(sum(sum(int(value) for value in labels) for labels in labels_per_query))
    total_label_count = int(sum(len(labels) for labels in labels_per_query))
    positive_query_count = int(sum(1 for labels in labels_per_query if any(int(value) for value in labels)))
    log_path.write_text(
        "\n".join(
            [
                "mode=procurement_expert_relevance",
                f"model_name={model_name}",
                "raw_llm_answers_available=false",
                "reason=LLM was not called in this run because cached labels were reused.",
                "To generate raw LLM answers, rerun evaluation with --no-reuse-llm-labels.",
                f"positive_labels={positive_label_count}/{total_label_count}",
                f"positive_queries={positive_query_count}/{len(labels_per_query)}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return log_path


def try_load_cached_furniture_labels(output_dir: Path, positions: Sequence[EvaluationPosition]) -> list[int] | None:
    cache_path = output_dir / "furniture_query_labels.parquet"
    manifest_path = output_dir / "furniture_query_labels_manifest.json"
    if not cache_path.exists():
        return None
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("prompt_version")) != LLM_LABEL_CACHE_VERSION:
        return None
    if int(manifest.get("positions_count", -1)) != len(positions):
        return None
    frame = pd.read_parquet(cache_path)
    if len(frame) != len(positions):
        return None
    cached_ids = frame["position_id"].astype(str).tolist()
    expected_ids = [position.position_id for position in positions]
    if cached_ids != expected_ids:
        return None
    return [int(value) for value in frame["is_furniture"].astype(int).tolist()]


def top_indices_digest(top_indices: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(top_indices.astype(np.int64, copy=False))
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def try_load_cached_llm_labels(
    output_dir: Path,
    positions: Sequence[EvaluationPosition],
    top_indices: np.ndarray,
    *,
    recover_from_raw_log: bool = False,
) -> list[list[int]] | None:
    labels_path = output_dir / "llm_labels_per_query.parquet"
    manifest_path = output_dir / "llm_labels_manifest.json"
    if labels_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("positions_count", -1)) != len(positions):
            return None
        if list(manifest.get("top_indices_shape", [])) != [int(value) for value in top_indices.shape]:
            return None
        if str(manifest.get("top_indices_sha256")) != top_indices_digest(top_indices):
            return None
        if str(manifest.get("prompt_version")) != LLM_LABEL_CACHE_VERSION:
            return None

        frame = pd.read_parquet(labels_path)
        if len(frame) != len(positions):
            return None
        labels_by_query: dict[int, list[int]] = {}
        for row in frame.to_dict("records"):
            labels_by_query[int(row["query_index"])] = [int(value) for value in json.loads(str(row["labels_json"]))]
        expected_width = int(top_indices.shape[1])
        labels: list[list[int]] = []
        for query_index in range(len(positions)):
            query_labels = labels_by_query.get(query_index)
            if query_labels is None or len(query_labels) != expected_width:
                return None
            labels.append(query_labels)
        return labels

    if not recover_from_raw_log:
        return None

    raw_log_path = output_dir / "llm_judgement_raw.jsonl"
    if not raw_log_path.exists():
        return None
    expected_width = int(top_indices.shape[1])
    labels = [[0] * expected_width for _ in positions]
    seen = [[False] * expected_width for _ in positions]
    try:
        with raw_log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                query_index = int(item["query_index"])
                if query_index < 0 or query_index >= len(positions):
                    return None
                parsed_labels = item.get("parsed_labels") or {}
                for rank_raw, relevant_raw in dict(parsed_labels).items():
                    rank_index = int(rank_raw) - 1
                    if 0 <= rank_index < expected_width:
                        labels[query_index][rank_index] = int(relevant_raw)
                        seen[query_index][rank_index] = True
    except Exception:
        return None
    if not all(all(row) for row in seen):
        return None
    print(f"Recovered cached LLM labels from old raw log: {raw_log_path}", flush=True)
    return labels


def save_llm_labels_cache(
    output_dir: Path,
    labels_per_query: Sequence[Sequence[int]],
    top_indices: np.ndarray,
    *,
    checkpoint_path: str,
) -> None:
    labels_frame = pd.DataFrame(
        [
            {
                "query_index": query_index,
                "labels_json": json.dumps([int(value) for value in labels], ensure_ascii=False),
            }
            for query_index, labels in enumerate(labels_per_query)
        ]
    )
    labels_frame.to_parquet(output_dir / "llm_labels_per_query.parquet", index=False)
    manifest = {
        "checkpoint_path": checkpoint_path,
        "positions_count": len(labels_per_query),
        "top_indices_shape": [int(value) for value in top_indices.shape],
        "top_indices_sha256": top_indices_digest(top_indices),
        "prompt_version": LLM_LABEL_CACHE_VERSION,
    }
    (output_dir / "llm_labels_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


async def classify_positions_furniture_with_scheduler(
    *,
    positions: Sequence[EvaluationPosition],
    output_dir: Path,
    ws_url: str,
    model_name: str,
    llm_chunk_size: int,
    max_in_flight: int,
    temperature: float,
    max_tokens: int,
    max_attempts: int = 3,
    request_timeout_seconds: float = DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS,
    scheduler_connect_timeout_seconds: float = DEFAULT_SCHEDULER_CONNECT_TIMEOUT_SECONDS,
    save_prompt_bodies: bool = False,
    save_raw_responses: bool = False,
) -> tuple[list[int], list[dict[str, Any]]]:
    client = SchedulerClient(ws_url, model=model_name, connect_timeout_seconds=scheduler_connect_timeout_seconds)
    await client.connect()
    labels: list[int] = [0] * len(positions)
    jobs = []
    for chunk_start in range(0, len(positions), llm_chunk_size):
        chunk = [(query_index, positions[query_index]) for query_index in range(chunk_start, min(chunk_start + llm_chunk_size, len(positions)))]
        prompt_body = build_furniture_classification_prompt_body(chunk)
        jobs.append(
            {
                "chunk_start": chunk_start,
                "expected_query_indices": [query_index for query_index, _position in chunk],
                "prompt_chars": len(FURNITURE_SCHEDULER_PROMPT),
                "prompt_body_chars": len(prompt_body),
                "full_prompt_chars": len(FURNITURE_SCHEDULER_PROMPT) + 2 + len(prompt_body) + 1,
                "prompt_body": prompt_body,
            }
        )

    progress = tqdm(total=len(jobs), desc="LLM furniture", unit="chunk", dynamic_ncols=True) if tqdm else None
    raw_log_path = output_dir / "furniture_llm_raw.jsonl"
    raw_log_handle = raw_log_path.open("w", encoding="utf-8")

    async def wrap_job(job: dict[str, Any]) -> tuple[dict[str, Any], dict[int, int] | None, list[dict[str, Any]], str | None]:
        parsed_labels, attempts, error_text = await submit_scheduler_job_with_retries(
            client=client,
            text=job["prompt_body"],
            prompt=FURNITURE_SCHEDULER_PROMPT,
            temperature=temperature,
            max_tokens=max_tokens,
            parse_fn=lambda raw_res, expected_values: parse_llm_furniture_labels(
                raw_res,
                expected_query_indices=expected_values,
            ),
            expected_values=job["expected_query_indices"],
            max_attempts=max_attempts,
            request_timeout_seconds=request_timeout_seconds,
        )
        return job, parsed_labels, attempts, error_text

    try:
        in_flight: set[Any] = set()
        job_iter = iter(jobs)
        max_in_flight = max(1, int(max_in_flight))

        while True:
            while len(in_flight) < max_in_flight:
                try:
                    job = next(job_iter)
                except StopIteration:
                    break
                in_flight.add(asyncio.create_task(wrap_job(job)))
            if not in_flight:
                break
            done, in_flight = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
            for done_task in done:
                job, parsed_labels, attempts, error_text = await done_task
                if parsed_labels is None:
                    parsed_labels = default_zero_labels(job["expected_query_indices"])
                log_item = {
                    "chunk_start": job["chunk_start"],
                    "expected_query_indices": job["expected_query_indices"],
                    "prompt_chars": job.get("prompt_chars"),
                    "prompt_body_chars": job.get("prompt_body_chars"),
                    "full_prompt_chars": job.get("full_prompt_chars"),
                    "attempts": compact_llm_attempts_for_log(attempts, include_raw_responses=save_raw_responses),
                    "error": error_text,
                    "parsed_labels": parsed_labels,
                }
                if save_prompt_bodies:
                    log_item["scheduler_prompt"] = FURNITURE_SCHEDULER_PROMPT
                    log_item["prompt_body"] = job["prompt_body"]
                raw_log_handle.write(json.dumps(log_item, ensure_ascii=False) + "\n")
                for query_index, is_furniture in parsed_labels.items():
                    labels[query_index] = is_furniture
                if progress is not None:
                    progress.update(1)
    finally:
        raw_log_handle.close()
        await client.close()
        if progress is not None:
            progress.close()

    return labels, []


def ensure_furniture_labels(
    *,
    positions: Sequence[EvaluationPosition],
    output_dir: Path,
    ws_url: str,
    model_name: str,
    llm_chunk_size: int,
    max_in_flight: int,
    temperature: float,
    max_tokens: int,
    max_attempts: int = 3,
    request_timeout_seconds: float = DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS,
    scheduler_connect_timeout_seconds: float = DEFAULT_SCHEDULER_CONNECT_TIMEOUT_SECONDS,
    save_prompt_bodies: bool = False,
    save_raw_responses: bool = False,
) -> list[int]:
    cached = try_load_cached_furniture_labels(output_dir, positions)
    if cached is not None:
        return cached
    labels, _raw_logs = asyncio.run(
        classify_positions_furniture_with_scheduler(
            positions=positions,
            output_dir=output_dir,
            ws_url=ws_url,
            model_name=model_name,
            llm_chunk_size=llm_chunk_size,
            max_in_flight=max_in_flight,
            temperature=temperature,
            max_tokens=max_tokens,
            max_attempts=max_attempts,
            request_timeout_seconds=request_timeout_seconds,
            scheduler_connect_timeout_seconds=scheduler_connect_timeout_seconds,
            save_prompt_bodies=save_prompt_bodies,
            save_raw_responses=save_raw_responses,
        )
    )
    frame = pd.DataFrame(
        [
            {
                "query_index": query_index,
                "position_id": positions[query_index].position_id,
                "name": positions[query_index].name,
                "description": positions[query_index].description,
                "is_furniture": int(labels[query_index]),
            }
            for query_index in range(len(positions))
        ]
    )
    frame.to_parquet(output_dir / "furniture_query_labels.parquet", index=False)
    (output_dir / "furniture_query_labels_manifest.json").write_text(
        json.dumps(
            {
                "positions_count": len(positions),
                "prompt_version": LLM_LABEL_CACHE_VERSION,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return labels


async def judge_candidates_with_scheduler(
    *,
    positions: Sequence[EvaluationPosition],
    candidate_scores: np.ndarray,
    candidate_indices: np.ndarray,
    candidate_probabilities: np.ndarray,
    materialized_candidates: dict[int, CandidateRecord],
    output_dir: Path,
    ws_url: str,
    model_name: str,
    max_candidates_per_query: int,
    llm_chunk_size: int,
    max_in_flight: int,
    temperature: float,
    max_tokens: int,
    max_attempts: int = 3,
    request_timeout_seconds: float = DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS,
    scheduler_connect_timeout_seconds: float = DEFAULT_SCHEDULER_CONNECT_TIMEOUT_SECONDS,
    save_prompt_bodies: bool = False,
    save_raw_responses: bool = False,
) -> tuple[list[list[int]], list[dict[str, Any]]]:
    del candidate_scores, candidate_probabilities
    client = SchedulerClient(ws_url, model=model_name, connect_timeout_seconds=scheduler_connect_timeout_seconds)
    await client.connect()
    labels_per_query: list[list[int]] = [[0] * min(max_candidates_per_query, candidate_indices.shape[1]) for _ in positions]
    jobs = []
    for query_index, position in enumerate(positions):
        query_candidate_indices = [int(index) for index in candidate_indices[query_index, :max_candidates_per_query].tolist() if int(index) >= 0]
        for chunk_start in range(0, len(query_candidate_indices), llm_chunk_size):
            chunk_indices = query_candidate_indices[chunk_start : chunk_start + llm_chunk_size]
            candidates = [(chunk_start + offset + 1, materialized_candidates[index]) for offset, index in enumerate(chunk_indices)]
            prompt_body = build_llm_judgement_prompt_body(position, candidates)
            jobs.append(
                {
                    "query_index": query_index,
                    "chunk_start": chunk_start,
                    "expected_ranks": [rank for rank, _candidate in candidates],
                    "candidate_count": len(candidates),
                    "prompt_chars": len(RELEVANCE_SCHEDULER_PROMPT),
                    "prompt_body_chars": len(prompt_body),
                    "full_prompt_chars": len(RELEVANCE_SCHEDULER_PROMPT) + 2 + len(prompt_body) + 1,
                    "prompt_body": prompt_body,
                }
            )

    progress = tqdm(total=len(jobs), desc="LLM judge", unit="chunk", dynamic_ncols=True) if tqdm else None
    raw_log_path = output_dir / "llm_judgement_raw.jsonl"
    raw_log_handle = raw_log_path.open("w", encoding="utf-8")
    procurement_answers_path = output_dir / "procurement_expert_llm_answers.jsonl"
    procurement_answers_handle = procurement_answers_path.open("w", encoding="utf-8")
    procurement_answers_log_path = output_dir / "procurement_expert_llm_answers.log"
    procurement_answers_log_handle = procurement_answers_log_path.open("w", encoding="utf-8")

    async def wrap_job(job: dict[str, Any]) -> tuple[dict[str, Any], dict[int, int] | None, list[dict[str, Any]], str | None]:
        parsed_labels, attempts, error_text = await submit_scheduler_job_with_retries(
            client=client,
            text=job["prompt_body"],
            prompt=RELEVANCE_SCHEDULER_PROMPT,
            temperature=temperature,
            max_tokens=max_tokens,
            parse_fn=lambda raw_res, expected_values: parse_llm_binary_labels(
                raw_res,
                expected_ranks=expected_values,
            ),
            expected_values=job["expected_ranks"],
            max_attempts=max_attempts,
            request_timeout_seconds=request_timeout_seconds,
        )
        return job, parsed_labels, attempts, error_text

    try:
        in_flight: set[Any] = set()
        job_iter = iter(jobs)
        max_in_flight = max(1, int(max_in_flight))

        while True:
            while len(in_flight) < max_in_flight:
                try:
                    job = next(job_iter)
                except StopIteration:
                    break
                in_flight.add(asyncio.create_task(wrap_job(job)))
            if not in_flight:
                break
            done, in_flight = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
            for done_task in done:
                job, parsed_labels, attempts, error_text = await done_task
                if parsed_labels is None:
                    parsed_labels = default_zero_labels(job["expected_ranks"])
                log_item = {
                    "query_index": job["query_index"],
                    "chunk_start": job["chunk_start"],
                    "expected_ranks": job["expected_ranks"],
                    "candidate_count": job.get("candidate_count"),
                    "prompt_chars": job.get("prompt_chars"),
                    "prompt_body_chars": job.get("prompt_body_chars"),
                    "full_prompt_chars": job.get("full_prompt_chars"),
                    "attempts": compact_llm_attempts_for_log(attempts, include_raw_responses=save_raw_responses),
                    "error": error_text,
                    "parsed_labels": parsed_labels,
                }
                if save_prompt_bodies:
                    log_item["scheduler_prompt"] = RELEVANCE_SCHEDULER_PROMPT
                    log_item["prompt_body"] = job["prompt_body"]
                raw_log_handle.write(json.dumps(log_item, ensure_ascii=False) + "\n")
                position = positions[job["query_index"]]
                procurement_answers_handle.write(
                    json.dumps(
                        {
                            "mode": "procurement_expert_relevance",
                            "query_index": job["query_index"],
                            "position_id": position.position_id,
                            "position_name": position.name,
                            "chunk_start": job["chunk_start"],
                            "expected_ranks": job["expected_ranks"],
                            "candidate_count": job.get("candidate_count"),
                            "prompt_chars": job.get("prompt_chars"),
                            "prompt_body_chars": job.get("prompt_body_chars"),
                            "full_prompt_chars": job.get("full_prompt_chars"),
                            "attempts": llm_answer_attempts_for_export(attempts),
                            "error": error_text,
                            "parsed_labels": parsed_labels,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                write_procurement_expert_answer_log(
                    procurement_answers_log_handle,
                    job=job,
                    position=position,
                    attempts=attempts,
                    parsed_labels=parsed_labels,
                    error_text=error_text,
                )
                for rank, relevant in parsed_labels.items():
                    labels_per_query[job["query_index"]][rank - 1] = relevant
                if progress is not None:
                    progress.update(1)
    finally:
        raw_log_handle.close()
        procurement_answers_handle.close()
        procurement_answers_log_handle.close()
        await client.close()
        if progress is not None:
            progress.close()

    return labels_per_query, []


def write_rankings_output(
    *,
    output_dir: Path,
    positions: Sequence[EvaluationPosition],
    top_scores: np.ndarray,
    top_indices: np.ndarray,
    top_probabilities: np.ndarray,
    labels_per_query: Sequence[Sequence[int]],
    materialized_candidates: dict[int, CandidateRecord],
    furniture_labels: Sequence[int] | None = None,
    include_text: bool = False,
    chunk_size: int = 100_000,
) -> None:
    ranking_path = output_dir / "retrieval_rankings.parquet"
    rows: list[dict[str, Any]] = []
    writer = None
    part_index = 0
    fallback_parts_dir = output_dir / "retrieval_rankings_parts"
    use_pyarrow_writer = True
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        pa = None
        pq = None
        use_pyarrow_writer = False
        fallback_parts_dir.mkdir(parents=True, exist_ok=True)

    total_rows = int(np.count_nonzero(top_indices >= 0))
    progress = tqdm(total=total_rows, desc="Write rankings", unit="row", dynamic_ncols=True) if tqdm else None

    def flush_rows() -> None:
        nonlocal rows, writer, part_index
        if not rows:
            return
        frame = pd.DataFrame(rows)
        if use_pyarrow_writer:
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(ranking_path, table.schema)
            writer.write_table(table)
        else:
            frame.to_parquet(fallback_parts_dir / f"part-{part_index:05d}.parquet", index=False)
            part_index += 1
        if progress is not None:
            progress.update(len(rows))
        rows = []

    for query_index, position in enumerate(positions):
        for rank in range(top_indices.shape[1]):
            catalog_index = int(top_indices[query_index, rank])
            if catalog_index < 0:
                continue
            candidate = materialized_candidates.get(catalog_index)
            row = {
                "query_index": query_index,
                "position_id": position.position_id,
                "is_furniture_query": int(furniture_labels[query_index]) if furniture_labels is not None else -1,
                "rank": rank + 1,
                "catalog_index": catalog_index,
                "score": float(top_scores[query_index, rank]),
                "softmax_prob": float(top_probabilities[query_index, rank]),
                "label": int(labels_per_query[query_index][rank]) if rank < len(labels_per_query[query_index]) else 0,
                "source_file": candidate.source_file if candidate else "",
                "source_row_index": candidate.source_row_index if candidate else -1,
            }
            if include_text:
                row.update(
                    {
                        "title": candidate.title if candidate else "",
                        "description": candidate.description if candidate else "",
                        "features_json": json.dumps(candidate.features, ensure_ascii=False) if candidate else "[]",
                    }
                )
            rows.append(row)
            if len(rows) >= max(1, int(chunk_size)):
                flush_rows()
    flush_rows()
    if writer is not None:
        writer.close()
    if progress is not None:
        progress.close()


def write_query_match_files(
    *,
    output_dir: Path,
    model_name: str,
    checkpoint_path: str,
    positions: Sequence[EvaluationPosition],
    top_scores: np.ndarray,
    top_indices: np.ndarray,
    top_probabilities: np.ndarray,
    labels_per_query: Sequence[Sequence[int]],
    materialized_candidates: dict[int, CandidateRecord],
    furniture_labels: Sequence[int] | None = None,
) -> dict[str, str]:
    positive_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []

    for query_index, position in enumerate(positions):
        best_positive_row: dict[str, Any] | None = None
        for rank in range(top_indices.shape[1]):
            catalog_index = int(top_indices[query_index, rank])
            if catalog_index < 0:
                continue
            label = int(labels_per_query[query_index][rank]) if rank < len(labels_per_query[query_index]) else 0
            if label != 1:
                continue
            candidate = materialized_candidates.get(catalog_index)
            row = {
                "model_name": model_name,
                "checkpoint_path": checkpoint_path,
                "query_index": query_index,
                "position_id": position.position_id,
                "query_text": position.query_text,
                "is_furniture_query": int(furniture_labels[query_index]) if furniture_labels is not None else -1,
                "rank": rank + 1,
                "catalog_index": catalog_index,
                "score": float(top_scores[query_index, rank]),
                "softmax_prob": float(top_probabilities[query_index, rank]),
                "matched_source_file": candidate.source_file if candidate else "",
                "matched_source_row_index": candidate.source_row_index if candidate else -1,
                "matched_title": candidate.title if candidate else "",
                "matched_description": candidate.description if candidate else "",
                "matched_features_json": json.dumps(candidate.features, ensure_ascii=False) if candidate else "[]",
            }
            positive_rows.append(row)
            if best_positive_row is None:
                best_positive_row = row

        if best_positive_row is None:
            best_rows.append(
                {
                    "model_name": model_name,
                    "checkpoint_path": checkpoint_path,
                    "query_index": query_index,
                    "position_id": position.position_id,
                    "query_text": position.query_text,
                    "is_furniture_query": int(furniture_labels[query_index]) if furniture_labels is not None else -1,
                    "has_match": 0,
                    "rank": -1,
                    "catalog_index": -1,
                    "score": float("nan"),
                    "softmax_prob": float("nan"),
                    "matched_source_file": "",
                    "matched_source_row_index": -1,
                    "matched_title": "",
                    "matched_description": "",
                    "matched_features_json": "[]",
                }
            )
        else:
            best_row = dict(best_positive_row)
            best_row["has_match"] = 1
            best_rows.append(best_row)

    positive_frame = pd.DataFrame(positive_rows)
    best_frame = pd.DataFrame(best_rows)
    output_columns = [
        "model_name",
        "checkpoint_path",
        "query_index",
        "position_id",
        "query_text",
        "is_furniture_query",
        "rank",
        "catalog_index",
        "score",
        "softmax_prob",
        "matched_source_file",
        "matched_source_row_index",
        "matched_title",
        "matched_description",
        "matched_features_json",
    ]
    best_columns = output_columns[:6] + ["has_match"] + output_columns[6:]
    positive_frame = positive_frame.reindex(columns=output_columns)
    best_frame = best_frame.reindex(columns=best_columns)

    positive_csv_path = output_dir / "query_to_matching_files.csv"
    positive_parquet_path = output_dir / "query_to_matching_files.parquet"
    best_csv_path = output_dir / "query_to_best_matching_file.csv"
    best_parquet_path = output_dir / "query_to_best_matching_file.parquet"
    txt_path = output_dir / "query_to_best_matching_file.txt"

    positive_frame.to_csv(positive_csv_path, index=False)
    positive_frame.to_parquet(positive_parquet_path, index=False)
    best_frame.to_csv(best_csv_path, index=False)
    best_frame.to_parquet(best_parquet_path, index=False)
    with txt_path.open("w", encoding="utf-8") as file:
        file.write("query_text\tmatched_source_file\tmatched_source_row_index\tmatched_title\trank\tscore\tsoftmax_prob\n")
        for row in best_rows:
            file.write(
                "\t".join(
                    [
                        str(row["query_text"]).replace("\t", " ").replace("\n", " "),
                        str(row["matched_source_file"]),
                        str(row["matched_source_row_index"]),
                        str(row["matched_title"]).replace("\t", " ").replace("\n", " "),
                        str(row["rank"]),
                        "" if pd.isna(row["score"]) else f"{float(row['score']):.8f}",
                        "" if pd.isna(row["softmax_prob"]) else f"{float(row['softmax_prob']):.12g}",
                    ]
                )
                + "\n"
            )

    return {
        "query_to_matching_files_csv": str(positive_csv_path),
        "query_to_matching_files_parquet": str(positive_parquet_path),
        "query_to_best_matching_file_csv": str(best_csv_path),
        "query_to_best_matching_file_parquet": str(best_parquet_path),
        "query_to_best_matching_file_txt": str(txt_path),
    }


def evaluate_metrics(
    *,
    positions: Sequence[EvaluationPosition],
    top_scores: np.ndarray,
    top_probabilities: np.ndarray,
    labels_per_query: Sequence[Sequence[int]],
    top_k_values: Sequence[int],
    top_p_values: Sequence[float],
    score_threshold_values: Sequence[float],
    probability_threshold_values: Sequence[float],
    furniture_labels: Sequence[int] | None = None,
    query_indices: Sequence[int] | None = None,
    desc: str = "Metrics",
) -> dict[str, Any]:
    per_query_rows: list[dict[str, Any]] = []
    aggregate_top_k: dict[str, dict[str, float]] = {}
    aggregate_top_p: dict[str, dict[str, float]] = {}
    aggregate_score_threshold: dict[str, dict[str, float]] = {}
    aggregate_probability_threshold: dict[str, dict[str, float]] = {}

    top_k_metrics_collection: dict[int, list[dict[str, float]]] = {int(k): [] for k in top_k_values}
    top_p_metrics_collection: dict[float, list[dict[str, float]]] = {float(p): [] for p in top_p_values}
    top_p_cutoff_sizes: dict[float, list[int]] = {float(p): [] for p in top_p_values}
    top_p_reached_counts: dict[float, int] = {float(p): 0 for p in top_p_values}
    score_threshold_metrics_collection: dict[float, list[dict[str, float]]] = {float(v): [] for v in score_threshold_values}
    probability_threshold_metrics_collection: dict[float, list[dict[str, float]]] = {float(v): [] for v in probability_threshold_values}
    score_threshold_cutoff_sizes: dict[float, list[int]] = {float(v): [] for v in score_threshold_values}
    probability_threshold_cutoff_sizes: dict[float, list[int]] = {float(v): [] for v in probability_threshold_values}

    selected_query_indices = list(query_indices) if query_indices is not None else list(range(len(positions)))
    progress = tqdm(total=len(selected_query_indices), desc=desc, unit="query", dynamic_ncols=True) if tqdm else None

    try:
        for query_index in selected_query_indices:
            position = positions[query_index]
            labels = [int(value) for value in labels_per_query[query_index]]
            row: dict[str, Any] = {
                "query_index": query_index,
                "position_id": position.position_id,
                "query_text": position.query_text,
                "relevant_in_pool": int(sum(labels)),
            }
            if furniture_labels is not None:
                row["is_furniture_query"] = int(furniture_labels[query_index])
            for top_k in top_k_values:
                metrics = compute_binary_metrics(labels, int(top_k))
                top_k_metrics_collection[int(top_k)].append(metrics)
                for key, value in metrics.items():
                    if key == "k":
                        continue
                    row[f"top_k_{top_k}_{key}"] = value

            for top_p in top_p_values:
                cutoff, reached = prefix_size_from_top_p(top_probabilities[query_index], float(top_p))
                metrics = compute_binary_metrics(labels, cutoff)
                top_p_metrics_collection[float(top_p)].append(metrics)
                top_p_cutoff_sizes[float(top_p)].append(cutoff)
                top_p_reached_counts[float(top_p)] += 1 if reached else 0
                row[f"top_p_{top_p}_cutoff"] = cutoff
                row[f"top_p_{top_p}_reached"] = int(reached)
                for key, value in metrics.items():
                    if key == "k":
                        continue
                    row[f"top_p_{top_p}_{key}"] = value

            for threshold in score_threshold_values:
                cutoff = prefix_size_from_score_threshold(top_scores[query_index], float(threshold))
                metrics = compute_binary_metrics(labels, cutoff)
                score_threshold_metrics_collection[float(threshold)].append(metrics)
                score_threshold_cutoff_sizes[float(threshold)].append(cutoff)
                row[f"score_threshold_{threshold}_cutoff"] = cutoff
                for key, value in metrics.items():
                    if key == "k":
                        continue
                    row[f"score_threshold_{threshold}_{key}"] = value

            for threshold in probability_threshold_values:
                cutoff = prefix_size_from_probability_threshold(top_probabilities[query_index], float(threshold))
                metrics = compute_binary_metrics(labels, cutoff)
                probability_threshold_metrics_collection[float(threshold)].append(metrics)
                probability_threshold_cutoff_sizes[float(threshold)].append(cutoff)
                row[f"probability_threshold_{threshold}_cutoff"] = cutoff
                for key, value in metrics.items():
                    if key == "k":
                        continue
                    row[f"probability_threshold_{threshold}_{key}"] = value
            per_query_rows.append(row)
            if progress is not None:
                progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    for top_k in top_k_values:
        aggregate_top_k[str(top_k)] = aggregate_metric_dicts(top_k_metrics_collection[int(top_k)])

    for top_p in top_p_values:
        aggregate_metrics = aggregate_metric_dicts(top_p_metrics_collection[float(top_p)])
        aggregate_metrics["avg_cutoff"] = float(sum(top_p_cutoff_sizes[float(top_p)]) / max(len(top_p_cutoff_sizes[float(top_p)]), 1))
        aggregate_metrics["reached_fraction"] = float(top_p_reached_counts[float(top_p)] / max(len(selected_query_indices), 1))
        aggregate_top_p[str(top_p)] = aggregate_metrics

    for threshold in score_threshold_values:
        aggregate_metrics = aggregate_metric_dicts(score_threshold_metrics_collection[float(threshold)])
        aggregate_metrics["avg_cutoff"] = float(sum(score_threshold_cutoff_sizes[float(threshold)]) / max(len(score_threshold_cutoff_sizes[float(threshold)]), 1))
        aggregate_metrics["non_empty_fraction"] = float(
            sum(1 for cutoff in score_threshold_cutoff_sizes[float(threshold)] if cutoff > 0) / max(len(selected_query_indices), 1)
        )
        aggregate_score_threshold[str(threshold)] = aggregate_metrics

    for threshold in probability_threshold_values:
        aggregate_metrics = aggregate_metric_dicts(probability_threshold_metrics_collection[float(threshold)])
        aggregate_metrics["avg_cutoff"] = float(
            sum(probability_threshold_cutoff_sizes[float(threshold)]) / max(len(probability_threshold_cutoff_sizes[float(threshold)]), 1)
        )
        aggregate_metrics["non_empty_fraction"] = float(
            sum(1 for cutoff in probability_threshold_cutoff_sizes[float(threshold)] if cutoff > 0) / max(len(selected_query_indices), 1)
        )
        aggregate_probability_threshold[str(threshold)] = aggregate_metrics

    return {
        "summary": {
            "query_count": len(selected_query_indices),
            "top_k": aggregate_top_k,
            "top_p": aggregate_top_p,
            "score_threshold": aggregate_score_threshold,
            "probability_threshold": aggregate_probability_threshold,
        },
        "per_query": per_query_rows,
    }


def normalize_model_output_name(value: str) -> str:
    candidate = value.strip()
    if is_url(candidate):
        parsed = urlparse(candidate)
        candidate = Path(parsed.path).name or "model"
    else:
        path = Path(candidate)
        if path.name in {"last_checkpoint.pt", "best_checkpoint.pt"} and path.parent.name:
            candidate = path.parent.name
        else:
            candidate = path.name if path.suffix or path.exists() else candidate
    for suffix in (".safetensors", ".model.pt", ".pt"):
        if candidate.endswith(suffix):
            candidate = candidate[: -len(suffix)]
            break
    candidate = re.sub(r"[^0-9A-Za-z._-]+", "_", candidate).strip("._-")
    return candidate or "model"


def build_unique_model_output_names(checkpoint_paths: Sequence[str]) -> dict[str, str]:
    counts: dict[str, int] = {}
    names: dict[str, str] = {}
    for checkpoint_path in checkpoint_paths:
        base_name = normalize_model_output_name(checkpoint_path)
        counts[base_name] = counts.get(base_name, 0) + 1
        suffix = counts[base_name]
        names[checkpoint_path] = base_name if suffix == 1 else f"{base_name}_{suffix}"
    return names


def resolve_catalog_vector_dir(
    checkpoint_source: str,
    *,
    model_output_name: str | None = None,
    catalog_vector_root: Path | None = None,
) -> Path:
    if catalog_vector_root is not None:
        if model_output_name:
            return catalog_vector_root / model_output_name
        return catalog_vector_root
    if is_url(checkpoint_source):
        raise ValueError("URL checkpoints need --catalog-vector-root because vectors cannot be stored next to a remote model.")

    path = Path(checkpoint_source).expanduser()
    if path.is_dir():
        return path / CATALOG_VECTOR_DIR_NAME
    if path.suffix or path.name.endswith(".model.pt"):
        return path.parent / CATALOG_VECTOR_DIR_NAME
    return path / CATALOG_VECTOR_DIR_NAME


def load_catalog_vector_manifest(vector_dir: Path) -> dict[str, Any]:
    manifest_path = vector_dir / CATALOG_MANIFEST_FILE_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Catalog vector manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def require_catalog_embedding_path(
    *,
    vector_dir: Path,
    files: Sequence[Path],
    embedding_dim: int,
    tokenizer_path: str,
    checkpoint_path: str,
    product_max_features: int,
    product_max_length: int,
    allow_stale: bool,
) -> tuple[Path, dict[str, Any]]:
    embedding_path = vector_dir / CATALOG_EMBEDDING_FILE_NAME
    if not embedding_path.exists():
        raise FileNotFoundError(
            f"Catalog embeddings not found: {embedding_path}. "
            "Run vectorize_dual_encoder_catalog.py for this checkpoint first."
        )
    manifest = load_catalog_vector_manifest(vector_dir)
    problems: list[str] = []
    if int(manifest.get("embedding_dim", -1)) != int(embedding_dim):
        problems.append(f"embedding_dim manifest={manifest.get('embedding_dim')} current={embedding_dim}")
    if int(manifest.get("max_features", -1)) != int(product_max_features):
        problems.append(f"product_max_features manifest={manifest.get('max_features')} current={product_max_features}")
    if int(manifest.get("product_max_length", -1)) != int(product_max_length):
        problems.append(f"product_max_length manifest={manifest.get('product_max_length')} current={product_max_length}")
    if str(manifest.get("tokenizer_path")) != str(tokenizer_path):
        problems.append(f"tokenizer_path manifest={manifest.get('tokenizer_path')} current={tokenizer_path}")
    if str(manifest.get("checkpoint_source")) != str(checkpoint_path):
        problems.append(f"checkpoint_source manifest={manifest.get('checkpoint_source')} current={checkpoint_path}")
    current_sources = build_source_files_manifest(files)
    if manifest.get("source_files") != current_sources:
        problems.append("source parquet files changed or were read from another data-dir")

    embeddings = np.load(embedding_path, mmap_mode="r")
    if len(embeddings.shape) != 2:
        problems.append(f"embedding array must be 2D, got shape={embeddings.shape}")
    else:
        if int(embeddings.shape[0]) != int(manifest.get("catalog_size", -1)):
            problems.append(f"catalog_size manifest={manifest.get('catalog_size')} embedding_rows={embeddings.shape[0]}")
        if int(embeddings.shape[1]) != int(embedding_dim):
            problems.append(f"embedding columns={embeddings.shape[1]} current_dim={embedding_dim}")
    del embeddings

    if problems and not allow_stale:
        details = "; ".join(problems)
        raise ValueError(
            f"Catalog vector cache in {vector_dir} does not match the current evaluation inputs: {details}. "
            "Re-run vectorize_dual_encoder_catalog.py or pass --allow-stale-catalog-vectors if this is intentional."
        )
    if problems:
        print(f"WARNING: using stale catalog vectors from {vector_dir}: {'; '.join(problems)}", flush=True)
    return embedding_path, manifest


def build_threshold_report_rows(
    *,
    model_name: str,
    checkpoint_path: str,
    scope_name: str,
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    query_count = int(summary.get("query_count", 0))
    for threshold_type in ("top_k", "top_p", "score_threshold", "probability_threshold"):
        bucket = dict(summary.get(threshold_type, {}) or {})
        for threshold_value, metrics in bucket.items():
            row = {
                "model_name": model_name,
                "checkpoint_path": checkpoint_path,
                "scope": scope_name,
                "threshold_type": threshold_type,
                "threshold_value": str(threshold_value),
                "query_count": query_count,
            }
            row.update({key: float(value) for key, value in metrics.items()})
            rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a dual-encoder against the full parquet catalog using LLM binary judgements.")
    parser.add_argument("--data-dir", required=True, help="Directory containing data0001.parquet ... data0029.parquet")
    parser.add_argument("--checkpoint-path", nargs="+", required=True, help="One or more checkpoint directories or .pt/.safetensors/.model.pt paths")
    parser.add_argument("--tokenizer-path", default=None, help="Optional tokenizer override. By default it is restored from the checkpoint metadata.")
    parser.add_argument("--positions-xlsx", required=True, help="Path to the Excel file with procurement positions.")
    parser.add_argument("--positions-sheet", default=None, help="Optional Excel sheet name override.")
    parser.add_argument("--output-dir", "--save-dir", dest="output_dir", required=True, help="Directory for rankings, LLM logs and metrics.")
    parser.add_argument("--device", default="auto", help="Device string, for example cuda, cuda:0 or cpu")
    parser.add_argument("--catalog-vector-root", "--vector-dir", dest="catalog_vector_root", default=None, help="Optional root with precomputed catalog vectors. By default vectors are read from <checkpoint_dir>/catalog_vectors.")
    parser.add_argument("--allow-stale-catalog-vectors", action="store_true", help="Use precomputed catalog vectors even if their manifest does not match the current inputs.")
    parser.add_argument("--query-batch-size", type=int, default=32, help="Query encoding batch size")
    parser.add_argument("--catalog-similarity-chunk-size", type=int, default=65536, help="Number of catalog embeddings per retrieval scan chunk")
    parser.add_argument("--retrieval-query-batch-size", type=int, default=16, help="Number of evaluation queries processed together during retrieval scan")
    parser.add_argument("--retrieval-pool-size", type=int, default=100, help="How many top catalog items to keep per query before LLM judgement")
    parser.add_argument("--top-k", default="1,3,5,10,20,50,100", help="Comma-separated top-k cutoffs")
    parser.add_argument("--top-p", default="0.5,0.8,0.9,0.95", help="Comma-separated top-p thresholds based on exact softmax over full-catalog scores")
    parser.add_argument("--score-thresholds", default="", help="Optional comma-separated score thresholds over ranked similarities")
    parser.add_argument("--probability-thresholds", default="", help="Optional comma-separated softmax probability thresholds over ranked candidates")
    parser.add_argument("--scheduler-ws-url", default=DEFAULT_SCHEDULER_WS_URL, help="Scheduler WebSocket URL")
    parser.add_argument("--scheduler-model", default=DEFAULT_SCHEDULER_MODEL, help="Scheduler model name")
    parser.add_argument("--llm-chunk-size", type=int, default=20, help="How many retrieved candidates to judge per single LLM request")
    parser.add_argument("--furniture-llm-chunk-size", type=int, default=40, help="How many procurement positions to classify as furniture/non-furniture per single LLM request")
    parser.add_argument("--llm-max-in-flight", type=int, default=DEFAULT_LLM_MAX_IN_FLIGHT, help="Maximum number of concurrent scheduler requests for relevance judgement. The default is GPU_COUNT*REQUESTS_PER_GPU=16*10=160.")
    parser.add_argument("--furniture-llm-max-in-flight", type=int, default=DEFAULT_LLM_MAX_IN_FLIGHT, help="Maximum number of concurrent scheduler requests for furniture/non-furniture classification.")
    parser.add_argument("--llm-temperature", type=float, default=DEFAULT_LLM_TEMPERATURE, help="LLM temperature. The default matches data_cleaner TEMPERATURE=0.7.")
    parser.add_argument("--llm-max-tokens", type=int, default=DEFAULT_LLM_MAX_TOKENS, help="LLM max_tokens. The default matches data_cleaner MAX_TOKENS=3000.")
    parser.add_argument("--llm-max-attempts", type=int, default=3, help="Maximum scheduler attempts per LLM chunk before defaulting labels to zero.")
    parser.add_argument("--llm-request-timeout-seconds", type=float, default=DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS, help="Timeout for one scheduler submit/result wait. Set 0 to disable.")
    parser.add_argument("--scheduler-connect-timeout-seconds", type=float, default=DEFAULT_SCHEDULER_CONNECT_TIMEOUT_SECONDS, help="Timeout for scheduler WebSocket connect/reconnect/close operations.")
    parser.add_argument("--save-llm-prompt-bodies", action=argparse.BooleanOptionalAction, default=False, help="Store full per-request prompt bodies in raw LLM jsonl logs. Disabled by default because this can create huge files.")
    parser.add_argument("--save-llm-raw-responses", action=argparse.BooleanOptionalAction, default=False, help="Store full raw scheduler responses in raw LLM jsonl logs. Disabled by default because this can create huge files.")
    parser.add_argument("--reuse-llm-labels", action=argparse.BooleanOptionalAction, default=True, help="Reuse cached LLM relevance labels when top_indices checksum matches.")
    parser.add_argument("--recover-llm-labels-from-raw-log", action=argparse.BooleanOptionalAction, default=False, help="Recover relevance labels from llm_judgement_raw.jsonl when the parquet label cache is absent. Disabled by default because old raw logs may come from obsolete prompts.")
    parser.add_argument("--save-retrieval-rankings", action=argparse.BooleanOptionalAction, default=True, help="Write retrieval_rankings.parquet after LLM judgement.")
    parser.add_argument("--rankings-include-text", action=argparse.BooleanOptionalAction, default=False, help="Include title, description and features_json in retrieval_rankings.parquet. Disabled by default for speed and disk usage.")
    parser.add_argument("--rankings-chunk-size", type=int, default=100_000, help="Rows per chunk while writing retrieval_rankings.parquet.")
    parser.add_argument("--save-match-files", action=argparse.BooleanOptionalAction, default=True, help="Write query_to_matching_files.* and query_to_best_matching_file.* after LLM judgement.")
    parser.add_argument("--product-max-features", type=int, default=14, help="How many product features to pass into the product encoder and to the LLM")
    parser.add_argument("--query-max-length", type=int, default=256, help="Maximum query sequence length for the query encoder")
    parser.add_argument("--product-max-length", type=int, default=1024, help="Maximum product sequence length for the product encoder")
    parser.add_argument("--cache-dir", default=None, help="Optional cache directory for downloaded checkpoints and workbook conversion")
    return parser.parse_args()


def evaluate_single_checkpoint(
    *,
    args: argparse.Namespace,
    checkpoint_path: str,
    model_output_name: str,
    model_output_dir: Path,
    shared_output_dir: Path,
    cache_dir: Path,
    positions: Sequence[EvaluationPosition],
    furniture_labels: Sequence[int],
    files: Sequence[Path],
    device: torch.device,
    top_k_values: Sequence[int],
    top_p_values: Sequence[float],
    score_threshold_values: Sequence[float],
    probability_threshold_values: Sequence[float],
) -> dict[str, Any]:
    model_output_dir.mkdir(parents=True, exist_ok=True)
    write_prompt_artifacts(model_output_dir)

    model_cache_dir = cache_dir / model_output_name
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint_payload, tokenizer_path = load_dual_encoder_for_eval(
        checkpoint_path,
        tokenizer_path_override=args.tokenizer_path,
        device=device,
        cache_dir=model_cache_dir,
    )
    encoder_max_features = int(getattr(model.product_encoder, "max_features", args.product_max_features))
    effective_product_max_features = min(int(args.product_max_features), encoder_max_features)
    if effective_product_max_features <= 0:
        raise ValueError("--product-max-features must be positive.")
    if effective_product_max_features != int(args.product_max_features):
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

    catalog_vector_root = Path(args.catalog_vector_root) if args.catalog_vector_root else None
    catalog_vector_dir = resolve_catalog_vector_dir(
        checkpoint_path,
        model_output_name=model_output_name,
        catalog_vector_root=catalog_vector_root,
    )
    embedding_path, catalog_manifest = require_catalog_embedding_path(
        vector_dir=catalog_vector_dir,
        files=files,
        embedding_dim=model.embedding_dim,
        tokenizer_path=tokenizer_path,
        checkpoint_path=checkpoint_path,
        product_max_features=effective_product_max_features,
        product_max_length=args.product_max_length,
        allow_stale=args.allow_stale_catalog_vectors,
    )
    feature_frequency = load_feature_frequency_from_vector_dir(catalog_vector_dir)
    if feature_frequency is None:
        raise FileNotFoundError(
            f"Feature frequency file not found in {catalog_vector_dir}. "
            "Run vectorize_dual_encoder_catalog.py for this checkpoint first."
        )
    if "catalog_size" in catalog_manifest:
        catalog_size = int(catalog_manifest["catalog_size"])
    else:
        catalog_size = int(np.load(embedding_path, mmap_mode="r").shape[0])

    query_embeddings = encode_query_embeddings(
        model,
        tokenizer_path=tokenizer_path,
        positions=positions,
        args=model_args,
        device=device,
    )
    top_scores, top_indices, logsumexp_values = retrieve_top_candidates(
        query_embeddings,
        embedding_path,
        top_n=args.retrieval_pool_size,
        chunk_size=args.catalog_similarity_chunk_size,
        query_batch_size=args.retrieval_query_batch_size,
        device=device,
    )
    top_probabilities = softmax_probs_from_top_scores(top_scores, logsumexp_values)

    needed_indices = {int(index) for index in top_indices.reshape(-1).tolist() if int(index) >= 0}
    materialized_candidates = materialize_candidates_by_index(
        files,
        needed_indices,
        feature_frequency,
        max_features=effective_product_max_features,
    )

    labels_per_query = (
        try_load_cached_llm_labels(
            model_output_dir,
            positions,
            top_indices,
            recover_from_raw_log=args.recover_llm_labels_from_raw_log,
        )
        if args.reuse_llm_labels
        else None
    )
    if labels_per_query is None:
        labels_per_query, _raw_logs = asyncio.run(
            judge_candidates_with_scheduler(
                positions=positions,
                candidate_scores=top_scores,
                candidate_indices=top_indices,
                candidate_probabilities=top_probabilities,
                materialized_candidates=materialized_candidates,
                output_dir=model_output_dir,
                ws_url=args.scheduler_ws_url,
                model_name=args.scheduler_model,
                max_candidates_per_query=args.retrieval_pool_size,
                llm_chunk_size=args.llm_chunk_size,
                max_in_flight=args.llm_max_in_flight,
                temperature=args.llm_temperature,
                max_tokens=args.llm_max_tokens,
                max_attempts=args.llm_max_attempts,
                request_timeout_seconds=args.llm_request_timeout_seconds,
                scheduler_connect_timeout_seconds=args.scheduler_connect_timeout_seconds,
                save_prompt_bodies=args.save_llm_prompt_bodies,
                save_raw_responses=args.save_llm_raw_responses,
            )
        )
        save_llm_labels_cache(
            model_output_dir,
            labels_per_query,
            top_indices,
            checkpoint_path=checkpoint_path,
        )
    else:
        print(f"Reusing cached LLM labels: {model_output_dir / 'llm_labels_per_query.parquet'}", flush=True)
        if not (model_output_dir / "llm_labels_manifest.json").exists():
            save_llm_labels_cache(
                model_output_dir,
                labels_per_query,
                top_indices,
                checkpoint_path=checkpoint_path,
            )
        cached_log_path = write_procurement_expert_cached_labels_log(
            model_output_dir,
            model_name=model_output_name,
            labels_per_query=labels_per_query,
        )
        print(f"Procurement expert LLM answers log was created from cache state: {cached_log_path}", flush=True)
    positive_label_count = int(sum(sum(int(value) for value in labels) for labels in labels_per_query))
    total_label_count = int(sum(len(labels) for labels in labels_per_query))
    positive_query_count = int(sum(1 for labels in labels_per_query if any(int(value) for value in labels)))
    label_summary = {
        "positive_labels": positive_label_count,
        "total_labels": total_label_count,
        "positive_queries": positive_query_count,
        "total_queries": len(labels_per_query),
        "positive_label_rate": float(positive_label_count / max(total_label_count, 1)),
        "positive_query_rate": float(positive_query_count / max(len(labels_per_query), 1)),
        "prompt_version": LLM_LABEL_CACHE_VERSION,
    }
    (model_output_dir / "llm_label_summary.json").write_text(
        json.dumps(label_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        "LLM labels: "
        f"positive_labels={positive_label_count}/{total_label_count}, "
        f"positive_queries={positive_query_count}/{len(labels_per_query)}",
        flush=True,
    )
    if positive_label_count == 0:
        print(
            "WARNING: LLM returned zero positive relevance labels. "
            "Metrics will be zero; inspect llm_judgement_raw.jsonl and consider rerunning with --save-llm-raw-responses.",
            flush=True,
        )

    match_file_paths: dict[str, str] = {}
    if args.save_match_files:
        match_file_paths = write_query_match_files(
            output_dir=model_output_dir,
            model_name=model_output_name,
            checkpoint_path=checkpoint_path,
            positions=positions,
            top_scores=top_scores,
            top_indices=top_indices,
            top_probabilities=top_probabilities,
            labels_per_query=labels_per_query,
            materialized_candidates=materialized_candidates,
            furniture_labels=furniture_labels,
        )
        print(
            f"Match file saved: {match_file_paths['query_to_best_matching_file_csv']}",
            flush=True,
        )

    if args.save_retrieval_rankings:
        write_rankings_output(
            output_dir=model_output_dir,
            positions=positions,
            top_scores=top_scores,
            top_indices=top_indices,
            top_probabilities=top_probabilities,
            labels_per_query=labels_per_query,
            materialized_candidates=materialized_candidates,
            furniture_labels=furniture_labels,
            include_text=args.rankings_include_text,
            chunk_size=args.rankings_chunk_size,
        )

    all_metrics = evaluate_metrics(
        positions=positions,
        top_scores=top_scores,
        top_probabilities=top_probabilities,
        labels_per_query=labels_per_query,
        top_k_values=top_k_values,
        top_p_values=top_p_values,
        score_threshold_values=score_threshold_values,
        probability_threshold_values=probability_threshold_values,
        furniture_labels=furniture_labels,
        query_indices=None,
        desc=f"Metrics all {model_output_name}",
    )
    furniture_query_indices = [index for index, label in enumerate(furniture_labels) if int(label) == 1]
    furniture_metrics = evaluate_metrics(
        positions=positions,
        top_scores=top_scores,
        top_probabilities=top_probabilities,
        labels_per_query=labels_per_query,
        top_k_values=top_k_values,
        top_p_values=top_p_values,
        score_threshold_values=score_threshold_values,
        probability_threshold_values=probability_threshold_values,
        furniture_labels=furniture_labels,
        query_indices=furniture_query_indices,
        desc=f"Metrics furniture {model_output_name}",
    )

    aggregate_summary = {
        "thresholds": {
            "top_k": [int(value) for value in top_k_values],
            "top_p": [float(value) for value in top_p_values],
            "score_threshold": [float(value) for value in score_threshold_values],
            "probability_threshold": [float(value) for value in probability_threshold_values],
        },
        "all_queries": all_metrics["summary"],
        "furniture_only": furniture_metrics["summary"],
    }
    (model_output_dir / "aggregate_metrics.json").write_text(
        json.dumps(aggregate_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    per_query_all_frame = pd.DataFrame(all_metrics["per_query"])
    per_query_all_frame.to_parquet(model_output_dir / "per_query_metrics.parquet", index=False)
    per_query_all_frame.to_parquet(model_output_dir / "per_query_metrics_all.parquet", index=False)
    pd.DataFrame(furniture_metrics["per_query"]).to_parquet(model_output_dir / "per_query_metrics_furniture.parquet", index=False)

    aggregate_rows = build_threshold_report_rows(
        model_name=model_output_name,
        checkpoint_path=checkpoint_path,
        scope_name="all_queries",
        summary=all_metrics["summary"],
    )
    aggregate_rows.extend(
        build_threshold_report_rows(
            model_name=model_output_name,
            checkpoint_path=checkpoint_path,
            scope_name="furniture_only",
            summary=furniture_metrics["summary"],
        )
    )
    aggregate_frame = pd.DataFrame(aggregate_rows)
    aggregate_frame.to_parquet(model_output_dir / "aggregate_metrics_by_threshold.parquet", index=False)
    aggregate_frame.to_csv(model_output_dir / "aggregate_metrics_by_threshold.csv", index=False)

    query_frame = pd.DataFrame(
        [
            {
                "query_index": query_index,
                "position_id": position.position_id,
                "name": position.name,
                "description": position.description,
                "quantity": position.quantity,
                "unit": position.unit,
                "query_text": position.query_text,
                "is_furniture_query": int(furniture_labels[query_index]),
            }
            for query_index, position in enumerate(positions)
        ]
    )
    query_frame.to_parquet(model_output_dir / "evaluation_positions.parquet", index=False)

    run_metadata = {
        "model_name": model_output_name,
        "positions_xlsx": str(args.positions_xlsx),
        "positions_count": len(positions),
        "furniture_positions_count": int(sum(int(label) for label in furniture_labels)),
        "catalog_size": catalog_size,
        "checkpoint_path": checkpoint_path,
        "tokenizer_path": tokenizer_path,
        "catalog_vector_dir": str(catalog_vector_dir),
        "catalog_embedding_path": str(embedding_path),
        "catalog_manifest": catalog_manifest,
        "device": str(device),
        "retrieval_pool_size": args.retrieval_pool_size,
        "top_k": [int(value) for value in top_k_values],
        "top_p": [float(value) for value in top_p_values],
        "score_thresholds": [float(value) for value in score_threshold_values],
        "probability_thresholds": [float(value) for value in probability_threshold_values],
        "scheduler_ws_url": args.scheduler_ws_url,
        "scheduler_model": args.scheduler_model,
        "llm_chunk_size": int(args.llm_chunk_size),
        "llm_max_in_flight": int(args.llm_max_in_flight),
        "llm_max_attempts": int(args.llm_max_attempts),
        "llm_request_timeout_seconds": float(args.llm_request_timeout_seconds),
        "scheduler_connect_timeout_seconds": float(args.scheduler_connect_timeout_seconds),
        "reuse_llm_labels": bool(args.reuse_llm_labels),
        "recover_llm_labels_from_raw_log": bool(args.recover_llm_labels_from_raw_log),
        "llm_label_cache_version": LLM_LABEL_CACHE_VERSION,
        "llm_label_summary": label_summary,
        "procurement_expert_llm_answers_path": str(model_output_dir / "procurement_expert_llm_answers.jsonl"),
        "procurement_expert_llm_answers_log_path": str(model_output_dir / "procurement_expert_llm_answers.log"),
        "save_llm_prompt_bodies": bool(args.save_llm_prompt_bodies),
        "save_llm_raw_responses": bool(args.save_llm_raw_responses),
        "save_retrieval_rankings": bool(args.save_retrieval_rankings),
        "rankings_include_text": bool(args.rankings_include_text),
        "rankings_chunk_size": int(args.rankings_chunk_size),
        "save_match_files": bool(args.save_match_files),
        "match_file_paths": match_file_paths,
        "product_max_features": int(effective_product_max_features),
        "furniture_llm_chunk_size": int(args.furniture_llm_chunk_size),
        "furniture_llm_max_in_flight": int(args.furniture_llm_max_in_flight),
        "safetensors_available": SAFETENSORS_AVAILABLE,
        "checkpoint_metadata_epoch": int(checkpoint_payload.get("epoch", 0)),
        "checkpoint_metadata_global_step": int(checkpoint_payload.get("global_step", 0)),
        "shared_output_dir": str(shared_output_dir),
        "prompt_dir": str(shared_output_dir / "prompts"),
    }
    (model_output_dir / "evaluation_run_metadata.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "model_name": model_output_name,
        "checkpoint_path": checkpoint_path,
        "output_dir": str(model_output_dir),
        "catalog_vector_dir": str(catalog_vector_dir),
        "catalog_size": catalog_size,
        "procurement_expert_llm_answers_log_path": str(model_output_dir / "procurement_expert_llm_answers.log"),
        "match_file_paths": match_file_paths,
        "summary": aggregate_summary,
        "aggregate_rows": aggregate_rows,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shared_output_dir = output_dir / "shared"
    shared_output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    top_k_values = parse_list_argument(args.top_k, cast_fn=int)
    top_p_values = parse_list_argument(args.top_p, cast_fn=float)
    score_threshold_values = parse_optional_list_argument(args.score_thresholds, cast_fn=float)
    probability_threshold_values = parse_optional_list_argument(args.probability_thresholds, cast_fn=float)
    if max(top_k_values) > args.retrieval_pool_size:
        raise ValueError("retrieval_pool_size must be >= max(top-k) because LLM labels are collected only for the retrieval pool.")

    checkpoint_paths = [str(path) for path in args.checkpoint_path]
    model_output_names = build_unique_model_output_names(checkpoint_paths)
    positions = load_evaluation_positions(Path(args.positions_xlsx), sheet_name=args.positions_sheet)
    device = resolve_device(args.device)
    write_prompt_artifacts(shared_output_dir)

    shared_query_frame = pd.DataFrame(
        [
            {
                "query_index": query_index,
                "position_id": position.position_id,
                "name": position.name,
                "description": position.description,
                "quantity": position.quantity,
                "unit": position.unit,
                "query_text": position.query_text,
            }
            for query_index, position in enumerate(positions)
        ]
    )
    shared_query_frame.to_parquet(shared_output_dir / "evaluation_positions.parquet", index=False)

    furniture_labels = ensure_furniture_labels(
        positions=positions,
        output_dir=shared_output_dir,
        ws_url=args.scheduler_ws_url,
        model_name=args.scheduler_model,
        llm_chunk_size=args.furniture_llm_chunk_size,
        max_in_flight=args.furniture_llm_max_in_flight,
        temperature=args.llm_temperature,
        max_tokens=args.llm_max_tokens,
        max_attempts=args.llm_max_attempts,
        request_timeout_seconds=args.llm_request_timeout_seconds,
        scheduler_connect_timeout_seconds=args.scheduler_connect_timeout_seconds,
        save_prompt_bodies=args.save_llm_prompt_bodies,
        save_raw_responses=args.save_llm_raw_responses,
    )

    shared_query_frame = shared_query_frame.copy()
    shared_query_frame["is_furniture_query"] = [int(value) for value in furniture_labels]
    shared_query_frame.to_parquet(shared_output_dir / "evaluation_positions_with_furniture.parquet", index=False)

    single_model = len(checkpoint_paths) == 1
    files = iter_source_parquet_files(Path(args.data_dir))
    all_model_results: list[dict[str, Any]] = []
    combined_aggregate_rows: list[dict[str, Any]] = []
    for checkpoint_path in checkpoint_paths:
        model_output_name = model_output_names[checkpoint_path]
        model_output_dir = output_dir if single_model else output_dir / "models" / model_output_name
        model_result = evaluate_single_checkpoint(
            args=args,
            checkpoint_path=checkpoint_path,
            model_output_name=model_output_name,
            model_output_dir=model_output_dir,
            shared_output_dir=shared_output_dir,
            cache_dir=cache_dir,
            positions=positions,
            furniture_labels=furniture_labels,
            files=files,
            device=device,
            top_k_values=top_k_values,
            top_p_values=top_p_values,
            score_threshold_values=score_threshold_values,
            probability_threshold_values=probability_threshold_values,
        )
        all_model_results.append(model_result)
        combined_aggregate_rows.extend(model_result["aggregate_rows"])
        print(
            f"Procurement expert LLM answers log for {model_result['model_name']}: "
            f"{model_result['procurement_expert_llm_answers_log_path']}",
            flush=True,
        )

    comparison_frame = pd.DataFrame(combined_aggregate_rows)
    if not comparison_frame.empty:
        comparison_frame.to_parquet(output_dir / "model_comparison_metrics.parquet", index=False)
        comparison_frame.to_csv(output_dir / "model_comparison_metrics.csv", index=False)

    overview_payload = {
        "positions_xlsx": str(args.positions_xlsx),
        "positions_count": len(positions),
        "furniture_positions_count": int(sum(int(label) for label in furniture_labels)),
        "catalog_size": int(all_model_results[0]["catalog_size"]) if all_model_results else 0,
        "checkpoint_paths": checkpoint_paths,
        "model_outputs": [
            {
                "model_name": result["model_name"],
                "checkpoint_path": result["checkpoint_path"],
                "output_dir": result["output_dir"],
                "catalog_vector_dir": result["catalog_vector_dir"],
                "catalog_size": result["catalog_size"],
                "procurement_expert_llm_answers_log_path": result["procurement_expert_llm_answers_log_path"],
                "match_file_paths": result.get("match_file_paths", {}),
                "summary": result["summary"],
            }
            for result in all_model_results
        ],
        "shared_output_dir": str(shared_output_dir),
        "scheduler_ws_url": args.scheduler_ws_url,
        "scheduler_model": args.scheduler_model,
        "llm_max_in_flight": int(args.llm_max_in_flight),
        "llm_max_attempts": int(args.llm_max_attempts),
        "llm_request_timeout_seconds": float(args.llm_request_timeout_seconds),
        "scheduler_connect_timeout_seconds": float(args.scheduler_connect_timeout_seconds),
        "furniture_llm_max_in_flight": int(args.furniture_llm_max_in_flight),
    }
    (output_dir / "evaluation_overview.json").write_text(
        json.dumps(overview_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Procurement expert LLM answers log paths:", flush=True)
    for result in all_model_results:
        print(
            f"- {result['model_name']}: {result['procurement_expert_llm_answers_log_path']}",
            flush=True,
        )
    print("Match file paths:", flush=True)
    for result in all_model_results:
        match_paths = result.get("match_file_paths", {}) or {}
        if match_paths:
            print(
                f"- {result['model_name']}: {match_paths.get('query_to_best_matching_file_csv')}",
                flush=True,
            )


if __name__ == "__main__":
    main()
