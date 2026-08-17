from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from hybrid_dual_path_model import (
    DualPathTransformerLayer,
    AttentionPooling,
    LearnedTokenTypeRouter,
    MergedTokenSpan,
    NumericFeatureEncoder,
    build_merge_join_weak_targets,
    build_soft_numeric_span_batch,
    build_token_type_weak_targets,
    count_parameters,
    load_tokenizer_from_json,
    NUMERIC_BAD_SPAN_LOSS_WEIGHT,
    NUMERIC_SPAN_POSTERIOR_TEMPERATURE,
)


ROLE_CLS = 0
ROLE_TITLE = 1
ROLE_DESCRIPTION = 2
ROLE_FEATURE_NAME = 3
ROLE_FEATURE_VALUE = 4
ROLE_SEPARATOR = 5
NUM_ROLES = 6


@dataclass(slots=True)
class ProductFeature:
    name: str
    value: str | int | float | None


@dataclass(slots=True)
class ProductCard:
    title: str
    description: str = ""
    features: list[ProductFeature] = field(default_factory=list)


@dataclass(slots=True)
class PreparedProductCardBatch:
    input_ids: Tensor
    attention_mask: Tensor
    role_ids: Tensor
    slot_ids: Tensor
    token_texts: list[list[str]]
    merged_tokens: list[list[MergedTokenSpan]]
    numeric_mask: Tensor | None = None
    numeric_values: Tensor | None = None


@dataclass(slots=True)
class ProductCardEncoderOutput:
    last_hidden_state: Tensor
    cls_state: Tensor
    retrieval_embedding: Tensor
    normalized_embedding: Tensor
    numeric_mask: Tensor
    numeric_values: Tensor
    role_ids: Tensor
    slot_ids: Tensor
    token_type_loss: Tensor | None = None
    token_type_logits: Tensor | None = None
    token_type_probabilities: Tensor | None = None
    merge_boundary_loss: Tensor | None = None
    merge_join_logits: Tensor | None = None
    merge_join_probabilities: Tensor | None = None
    merged_token_texts: list[list[str]] | None = None


def _normalize_feature(feature: ProductFeature | Mapping[str, Any]) -> ProductFeature:
    if isinstance(feature, ProductFeature):
        return feature
    if isinstance(feature, Mapping):
        return ProductFeature(
            name=str(feature.get("name", "") or ""),
            value=feature.get("value"),
        )
    raise TypeError(f"Unsupported feature type: {type(feature)!r}")


def _normalize_card(card: ProductCard | Mapping[str, Any]) -> ProductCard:
    if isinstance(card, ProductCard):
        return card
    if isinstance(card, Mapping):
        raw_features = card.get("features") or card.get("attributes") or []
        features = [_normalize_feature(feature) for feature in raw_features]
        return ProductCard(
            title=str(card.get("title", "") or ""),
            description=str(card.get("description", "") or card.get("text", "") or ""),
            features=features,
        )
    raise TypeError(f"Unsupported card type: {type(card)!r}")


def _stringify_feature_value(value: str | int | float | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _tokenize_segment_without_specials(tokenizer: Any, text: str) -> tuple[list[int], list[str], list[tuple[int, int]]]:
    encoding = tokenizer.encode(text)
    token_ids: list[int] = []
    tokens: list[str] = []
    offsets: list[tuple[int, int]] = []

    for token_id, token, offset in zip(encoding.ids, encoding.tokens, encoding.offsets, strict=False):
        start, end = offset
        if end <= start:
            continue
        token_ids.append(token_id)
        tokens.append(token)
        offsets.append((start, end))
    return token_ids, tokens, offsets


def _tokenize_segment_tokens(tokenizer: Any, text: str) -> list[MergedTokenSpan]:
    if not text.strip():
        return []
    token_ids, tokens, offsets = _tokenize_segment_without_specials(tokenizer, text)
    if not token_ids:
        return []
    return [
        MergedTokenSpan(
            token_type="RAW",
            text=token,
            token_id=token_id,
            source_ids=(token_id,),
            source_tokens=(token,),
            start=index,
            end=index + 1,
            numeric_value=None,
            source_positions=(index,),
        )
        for index, (token_id, token, _offset) in enumerate(zip(token_ids, tokens, offsets, strict=False))
    ]


def _maybe_add_separator(
    merged_tokens: list[MergedTokenSpan],
    role_ids: list[int],
    slot_ids: list[int],
    *,
    eos_token_id: int | None,
    slot_id: int,
    segment_has_tokens: bool,
) -> None:
    if eos_token_id is None or not segment_has_tokens:
        return
    merged_tokens.append(
        MergedTokenSpan(
            token_type="STR",
            text="<sep>",
            token_id=eos_token_id,
            source_ids=(eos_token_id,),
            source_tokens=("<sep>",),
            start=0,
            end=0,
            numeric_value=None,
        )
    )
    role_ids.append(ROLE_SEPARATOR)
    slot_ids.append(slot_id)


def prepare_product_card_batch_from_tokenizer(
    cards: Sequence[ProductCard | Mapping[str, Any]],
    tokenizer: Any,
    *,
    max_features: int = 14,
    max_length: int | None = 8194,
    pad_token_id: int = 1,
    bos_token_id: int | None = None,
    eos_token_id: int | None = None,
    numeric_token_id: int | None = None,
    device: torch.device | str | None = None,
) -> PreparedProductCardBatch:
    if max_features <= 0:
        raise ValueError("max_features must be positive.")

    if bos_token_id is None and hasattr(tokenizer, "token_to_id"):
        bos_token_id = tokenizer.token_to_id("<s>")
        if bos_token_id is None:
            bos_token_id = tokenizer.token_to_id("<bos>")
    if eos_token_id is None and hasattr(tokenizer, "token_to_id"):
        eos_token_id = tokenizer.token_to_id("</s>")
        if eos_token_id is None:
            eos_token_id = tokenizer.token_to_id("<eos>")
    if pad_token_id is None and hasattr(tokenizer, "token_to_id"):
        pad_token_id = tokenizer.token_to_id("<pad>")
        if pad_token_id is None:
            raise ValueError("pad_token_id is required when tokenizer has no <pad> token.")

    merged_batch: list[list[MergedTokenSpan]] = []
    role_batch: list[list[int]] = []
    slot_batch: list[list[int]] = []
    token_texts: list[list[str]] = []

    for raw_card in cards:
        card = _normalize_card(raw_card)
        merged_tokens: list[MergedTokenSpan] = []
        role_ids: list[int] = []
        slot_ids: list[int] = []

        if bos_token_id is not None:
            merged_tokens.append(
                MergedTokenSpan(
                    token_type="STR",
                    text="<cls>",
                    token_id=bos_token_id,
                    source_ids=(bos_token_id,),
                    source_tokens=("<cls>",),
                    start=0,
                    end=0,
                    numeric_value=None,
                )
            )
            role_ids.append(ROLE_CLS)
            slot_ids.append(0)

        title_tokens = _tokenize_segment_tokens(tokenizer, card.title)
        merged_tokens.extend(title_tokens)
        role_ids.extend([ROLE_TITLE] * len(title_tokens))
        slot_ids.extend([0] * len(title_tokens))
        _maybe_add_separator(
            merged_tokens,
            role_ids,
            slot_ids,
            eos_token_id=eos_token_id,
            slot_id=0,
            segment_has_tokens=bool(title_tokens),
        )

        description_tokens = _tokenize_segment_tokens(tokenizer, card.description)
        merged_tokens.extend(description_tokens)
        role_ids.extend([ROLE_DESCRIPTION] * len(description_tokens))
        slot_ids.extend([0] * len(description_tokens))
        _maybe_add_separator(
            merged_tokens,
            role_ids,
            slot_ids,
            eos_token_id=eos_token_id,
            slot_id=0,
            segment_has_tokens=bool(description_tokens),
        )

        for feature_index, feature in enumerate(card.features[:max_features], start=1):
            feature_name_tokens = _tokenize_segment_tokens(tokenizer, feature.name)
            merged_tokens.extend(feature_name_tokens)
            role_ids.extend([ROLE_FEATURE_NAME] * len(feature_name_tokens))
            slot_ids.extend([feature_index] * len(feature_name_tokens))
            _maybe_add_separator(
                merged_tokens,
                role_ids,
                slot_ids,
                eos_token_id=eos_token_id,
                slot_id=feature_index,
                segment_has_tokens=bool(feature_name_tokens),
            )

            value_text = _stringify_feature_value(feature.value)
            feature_value_tokens = _tokenize_segment_tokens(tokenizer, value_text)
            merged_tokens.extend(feature_value_tokens)
            role_ids.extend([ROLE_FEATURE_VALUE] * len(feature_value_tokens))
            slot_ids.extend([feature_index] * len(feature_value_tokens))
            _maybe_add_separator(
                merged_tokens,
                role_ids,
                slot_ids,
                eos_token_id=eos_token_id,
                slot_id=feature_index,
                segment_has_tokens=bool(feature_value_tokens),
            )

        if max_length is not None:
            merged_tokens = merged_tokens[:max_length]
            role_ids = role_ids[:max_length]
            slot_ids = slot_ids[:max_length]

        if not merged_tokens:
            raise ValueError("Product card produced an empty token sequence.")

        merged_batch.append(merged_tokens)
        role_batch.append(role_ids)
        slot_batch.append(slot_ids)
        token_texts.append([token.text for token in merged_tokens])

    batch_size = len(merged_batch)
    seq_len = max((len(tokens) for tokens in merged_batch), default=0)
    input_ids = torch.full((batch_size, seq_len), fill_value=pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
    role_ids_tensor = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
    slot_ids_tensor = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)

    for batch_index, merged_tokens in enumerate(merged_batch):
        for token_index, merged_token in enumerate(merged_tokens):
            input_ids[batch_index, token_index] = merged_token.token_id
            attention_mask[batch_index, token_index] = True
            role_ids_tensor[batch_index, token_index] = role_batch[batch_index][token_index]
            slot_ids_tensor[batch_index, token_index] = slot_batch[batch_index][token_index]

    return PreparedProductCardBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        role_ids=role_ids_tensor,
        slot_ids=slot_ids_tensor,
        token_texts=token_texts,
        merged_tokens=merged_batch,
    )


def prepare_product_card_batch_from_tokenizer_json(
    cards: Sequence[ProductCard | Mapping[str, Any]],
    tokenizer_path: str | Path,
    *,
    max_features: int = 14,
    max_length: int | None = 8194,
    pad_token_id: int = 1,
    bos_token_id: int | None = None,
    eos_token_id: int | None = None,
    numeric_token_id: int | None = None,
    device: torch.device | str | None = None,
) -> PreparedProductCardBatch:
    tokenizer = load_tokenizer_from_json(tokenizer_path)
    return prepare_product_card_batch_from_tokenizer(
        cards=cards,
        tokenizer=tokenizer,
        max_features=max_features,
        max_length=max_length,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        numeric_token_id=numeric_token_id,
        device=device,
    )


class BgeM3ProductCardEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        *,
        d_model: int = 1024,
        num_heads: int = 16,
        num_layers: int = 24,
        token_type_router_layers: int = 4,
        ff_multiplier: int = 4,
        max_position_embeddings: int = 8194,
        max_features: int = 14,
        local_attention_window: int = 128,
        max_routing_clusters: int = 128,
        dropout: float = 0.1,
        pad_token_id: int = 1,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
        id_to_token: Sequence[str] | None = None,
        retrieval_projection_dim: int | None = None,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if max_features <= 0:
            raise ValueError("max_features must be positive.")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.token_type_router_layers = token_type_router_layers
        self.max_position_embeddings = max_position_embeddings
        self.max_features = max_features
        self.local_attention_window = max(1, int(local_attention_window))
        self.max_routing_clusters = max(1, int(max_routing_clusters))
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.id_to_token = list(id_to_token) if id_to_token is not None else None
        self.gradient_checkpointing = False

        self.token_embeddings = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.type_embeddings = nn.Embedding(2, d_model)
        self.role_embeddings = nn.Embedding(NUM_ROLES, d_model)
        self.slot_embeddings = nn.Embedding(max_features + 1, d_model)
        self.routing_centroids = nn.Parameter(torch.randn(self.max_routing_clusters, d_model) / math.sqrt(d_model))
        self.token_type_router = LearnedTokenTypeRouter(
            d_model,
            num_heads=num_heads,
            num_layers=token_type_router_layers,
            ff_multiplier=2,
            dropout=dropout,
        )

        self.string_input_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.numeric_encoder = NumericFeatureEncoder(d_model, dropout=dropout)
        self.numeric_input_gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                DualPathTransformerLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    ff_multiplier=ff_multiplier,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.attention_pooler = AttentionPooling(d_model, dropout=dropout)
        self.retrieval_projection = (
            nn.Linear(d_model, retrieval_projection_dim, bias=False)
            if retrieval_projection_dim is not None and retrieval_projection_dim != d_model
            else None
        )

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

    def _resolve_token_texts(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_texts: Sequence[Sequence[str]] | None,
    ) -> list[list[str]]:
        if token_texts is not None:
            resolved: list[list[str]] = []
            for batch_index in range(input_ids.size(0)):
                sequence_length = int(attention_mask[batch_index].sum().item())
                resolved.append(list(token_texts[batch_index][:sequence_length]))
            return resolved

        if self.id_to_token is None:
            raise ValueError("token_texts are required unless id_to_token lookup is attached to the product encoder.")

        resolved = []
        for batch_index in range(input_ids.size(0)):
            sequence_length = int(attention_mask[batch_index].sum().item())
            row_tokens = []
            for token_id in input_ids[batch_index, :sequence_length].detach().cpu().tolist():
                row_tokens.append(self.id_to_token[token_id] if token_id < len(self.id_to_token) else "")
            resolved.append(row_tokens)
        return resolved

    def _build_product_sparse_self_attn_mask(
        self,
        states: Tensor,
        attention_mask: Tensor,
        role_ids: Tensor,
    ) -> Tensor:
        batch_size, seq_len, _dim = states.shape
        device = states.device
        disallow = torch.ones((batch_size, seq_len, seq_len), dtype=torch.bool, device=device)
        if seq_len == 0:
            return disallow

        positions = torch.arange(seq_len, device=device)
        local_allowed = (positions[:, None] - positions[None, :]).abs() <= self.local_attention_window
        global_roles = (
            (role_ids == ROLE_CLS)
            | (role_ids == ROLE_TITLE)
            | (role_ids == ROLE_DESCRIPTION)
            | (role_ids == ROLE_FEATURE_NAME)
        )

        centroid_bank = F.normalize(self.routing_centroids, p=2, dim=-1)
        for batch_index in range(batch_size):
            active = attention_mask[batch_index].to(dtype=torch.bool)
            active_indices = active.nonzero(as_tuple=False).flatten()
            if active_indices.numel() == 0:
                disallow[batch_index].fill_(False)
                continue

            allowed = local_allowed & active[:, None] & active[None, :]
            global_mask = global_roles[batch_index] & active
            if global_mask.any():
                allowed[global_mask, :] = active.unsqueeze(0).expand(int(global_mask.sum().item()), -1)
                allowed[:, global_mask] = active.unsqueeze(-1).expand(-1, int(global_mask.sum().item()))

            active_count = int(active_indices.numel())
            cluster_count = max(1, min(self.max_routing_clusters, math.ceil(math.sqrt(active_count))))
            active_states = F.normalize(states[batch_index, active_indices].detach().float(), p=2, dim=-1)
            assignments = torch.argmax(active_states @ centroid_bank[:cluster_count].float().transpose(0, 1), dim=-1)
            same_cluster = assignments[:, None] == assignments[None, :]
            allowed[active_indices[:, None], active_indices[None, :]] = (
                allowed[active_indices[:, None], active_indices[None, :]] | same_cluster
            )

            # Padded queries still need finite attention rows; their outputs are masked later.
            allowed[~active, :] = active.unsqueeze(0).expand(int((~active).sum().item()), -1)
            disallow[batch_index] = ~allowed
        return disallow

    def prepare_inputs(
        self,
        cards: Sequence[ProductCard | Mapping[str, Any]],
        tokenizer: Any,
        *,
        numeric_token_id: int | None = None,
        max_length: int | None = None,
        device: torch.device | str | None = None,
    ) -> PreparedProductCardBatch:
        return prepare_product_card_batch_from_tokenizer(
            cards=cards,
            tokenizer=tokenizer,
            max_features=self.max_features,
            max_length=max_length or self.max_position_embeddings,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            numeric_token_id=numeric_token_id,
            device=device if device is not None else self.token_embeddings.weight.device,
        )

    def prepare_inputs_from_tokenizer_json(
        self,
        cards: Sequence[ProductCard | Mapping[str, Any]],
        tokenizer_path: str | Path,
        *,
        numeric_token_id: int | None = None,
        max_length: int | None = None,
        device: torch.device | str | None = None,
    ) -> PreparedProductCardBatch:
        tokenizer = load_tokenizer_from_json(tokenizer_path)
        return self.prepare_inputs(
            cards=cards,
            tokenizer=tokenizer,
            numeric_token_id=numeric_token_id,
            max_length=max_length,
            device=device,
        )

    def encode_cards(
        self,
        cards: Sequence[ProductCard | Mapping[str, Any]],
        tokenizer: Any,
        *,
        numeric_token_id: int | None = None,
        max_length: int | None = None,
        device: torch.device | str | None = None,
        compute_token_type_loss: bool = False,
    ) -> ProductCardEncoderOutput:
        prepared = self.prepare_inputs(
            cards=cards,
            tokenizer=tokenizer,
            numeric_token_id=numeric_token_id,
            max_length=max_length,
            device=device,
        )
        return self(
            input_ids=prepared.input_ids,
            attention_mask=prepared.attention_mask,
            token_texts=prepared.token_texts,
            numeric_token_id=numeric_token_id,
            role_ids=prepared.role_ids,
            slot_ids=prepared.slot_ids,
            compute_token_type_loss=compute_token_type_loss,
        )

    def encode_cards_from_tokenizer_json(
        self,
        cards: Sequence[ProductCard | Mapping[str, Any]],
        tokenizer_path: str | Path,
        *,
        numeric_token_id: int | None = None,
        max_length: int | None = None,
        device: torch.device | str | None = None,
        compute_token_type_loss: bool = False,
    ) -> ProductCardEncoderOutput:
        tokenizer = load_tokenizer_from_json(tokenizer_path)
        return self.encode_cards(
            cards=cards,
            tokenizer=tokenizer,
            numeric_token_id=numeric_token_id,
            max_length=max_length,
            device=device,
            compute_token_type_loss=compute_token_type_loss,
        )

    def forward(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor,
        token_texts: Sequence[Sequence[str]] | None = None,
        numeric_token_id: int | None = None,
        numeric_mask: Tensor | None = None,
        numeric_values: Tensor | None = None,
        role_ids: Tensor,
        slot_ids: Tensor,
        compute_token_type_loss: bool = False,
    ) -> ProductCardEncoderOutput:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape [batch, seq].")

        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_position_embeddings:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_position_embeddings={self.max_position_embeddings}."
            )
        if role_ids.shape != input_ids.shape or slot_ids.shape != input_ids.shape:
            raise ValueError("role_ids and slot_ids must match input_ids shape.")

        attention_mask = attention_mask.to(dtype=torch.bool, device=input_ids.device)
        resolved_token_texts = self._resolve_token_texts(input_ids, attention_mask, token_texts)

        raw_role_emb = self.role_embeddings(role_ids.to(device=input_ids.device))
        raw_slot_emb = self.slot_embeddings(slot_ids.to(device=input_ids.device).clamp(min=0, max=self.max_features))
        raw_token_emb = self.token_embeddings(input_ids)
        raw_base_states = self.dropout(raw_token_emb + raw_role_emb + raw_slot_emb)

        routing_output = self.token_type_router(
            raw_base_states,
            attention_mask,
            attn_mask=None,
            use_gradient_checkpointing=self.gradient_checkpointing,
        )
        token_type_logits = routing_output.logits
        token_type_probabilities = torch.softmax(token_type_logits, dim=-1)
        token_type_probabilities = token_type_probabilities * attention_mask.unsqueeze(-1).to(
            dtype=token_type_probabilities.dtype
        )
        soft_numeric = build_soft_numeric_span_batch(
            token_texts=resolved_token_texts,
            attention_mask=attention_mask,
            token_type_probabilities=token_type_probabilities,
            join_logits=routing_output.join_logits,
            numeric_encoder=self.numeric_encoder,
            role_ids=role_ids.to(device=input_ids.device),
            slot_ids=slot_ids.to(device=input_ids.device),
            separator_role_id=ROLE_SEPARATOR,
            posterior_temperature=NUMERIC_SPAN_POSTERIOR_TEMPERATURE,
            device=input_ids.device,
        )

        if numeric_mask is not None and numeric_values is not None:
            numeric_mask = numeric_mask.to(dtype=torch.bool, device=input_ids.device)[:, :seq_len] & attention_mask
            numeric_values = numeric_values.to(dtype=torch.float32, device=input_ids.device)[:, :seq_len]
            numeric_values = numeric_values * numeric_mask.to(dtype=torch.float32)
            numeric_embeddings = self.numeric_encoder(numeric_values, numeric_mask)
            numeric_mass = numeric_mask.to(dtype=token_type_probabilities.dtype)
            viterbi_token_texts = resolved_token_texts
        else:
            numeric_mask = soft_numeric.numeric_mask & attention_mask
            numeric_values = soft_numeric.numeric_values
            numeric_embeddings = soft_numeric.numeric_embeddings
            numeric_mass = soft_numeric.numeric_mass
            viterbi_token_texts = soft_numeric.viterbi_token_texts
        numeric_mass = numeric_mass * attention_mask.to(dtype=numeric_mass.dtype)
        string_mask = attention_mask
        numeric_attention_mask = numeric_mask & attention_mask

        string_type_emb = self.type_embeddings.weight[0].view(1, 1, -1)
        numeric_type_emb = self.type_embeddings.weight[1].view(1, 1, -1)
        role_emb = self.role_embeddings(role_ids.to(device=input_ids.device))
        slot_emb = self.slot_embeddings(slot_ids.to(device=input_ids.device).clamp(min=0, max=self.max_features))

        token_emb = self.token_embeddings(input_ids)
        string_base_states = self.dropout(token_emb + string_type_emb + role_emb + slot_emb)
        string_weight = token_type_probabilities[..., 0] * attention_mask.to(dtype=token_type_probabilities.dtype)
        string_route = routing_output.route_residual * string_weight.unsqueeze(-1)
        numeric_route = routing_output.route_residual * numeric_mass.unsqueeze(-1)

        string_states = self.string_input_mlp(string_base_states + string_route)
        string_states = string_states * string_weight.unsqueeze(-1).to(dtype=string_states.dtype)

        numeric_states = self.numeric_input_gate(numeric_embeddings + numeric_type_emb + role_emb + slot_emb + numeric_route)
        numeric_states = numeric_states * numeric_mass.unsqueeze(-1).to(dtype=numeric_states.dtype)

        for layer in self.layers:
            routing_states = string_states + numeric_states
            self_attn_mask = self._build_product_sparse_self_attn_mask(
                routing_states,
                attention_mask,
                role_ids.to(device=input_ids.device),
            )
            if self.gradient_checkpointing and self.training:
                string_states, numeric_states = checkpoint(
                    lambda s, n, m: layer(
                        s,
                        n,
                        string_mask=string_mask,
                        numeric_mask=numeric_attention_mask,
                        self_attn_mask=m,
                        cross_attn_mask=None,
                    ),
                    string_states,
                    numeric_states,
                    self_attn_mask,
                    use_reentrant=False,
                )
            else:
                string_states, numeric_states = layer(
                    string_states,
                    numeric_states,
                    string_mask=string_mask,
                    numeric_mask=numeric_attention_mask,
                    self_attn_mask=self_attn_mask,
                    cross_attn_mask=None,
                )

        mixed_states = string_states + numeric_states
        route_weight = (string_weight + numeric_mass).clamp(max=1.0)
        mixed_states = mixed_states + routing_output.route_residual * route_weight.unsqueeze(-1)
        soft_type_emb = string_type_emb * string_weight.unsqueeze(-1) + numeric_type_emb * numeric_mass.unsqueeze(-1)
        mixed_states = mixed_states + soft_type_emb
        mixed_states = self.final_norm(mixed_states)
        mixed_states = mixed_states * attention_mask.unsqueeze(-1).to(dtype=mixed_states.dtype)

        cls_state = mixed_states[:, 0, :]
        pooled_state = self.attention_pooler(mixed_states, attention_mask)
        retrieval_embedding = self.retrieval_projection(pooled_state) if self.retrieval_projection is not None else pooled_state
        normalized_embedding = F.normalize(retrieval_embedding, p=2, dim=-1)
        token_type_loss = None
        merge_boundary_loss = None
        if compute_token_type_loss:
            auxiliary_losses: list[Tensor] = []
            weak_targets, weak_target_mask = build_token_type_weak_targets(
                resolved_token_texts,
                attention_mask,
                device=input_ids.device,
            )
            if weak_target_mask.any():
                selected_logits = routing_output.logits[weak_target_mask]
                selected_targets = weak_targets[weak_target_mask]
                token_type_loss = F.cross_entropy(selected_logits, selected_targets)
                auxiliary_losses.append(token_type_loss)
            join_targets, join_target_mask = build_merge_join_weak_targets(
                resolved_token_texts,
                attention_mask,
                device=input_ids.device,
            )
            if join_target_mask.any():
                merge_losses = [
                    F.binary_cross_entropy_with_logits(
                        routing_output.join_logits[join_target_mask],
                        join_targets[join_target_mask].to(dtype=routing_output.join_logits.dtype),
                    )
                ]
                if soft_numeric.span_loss is not None:
                    merge_losses.append(soft_numeric.span_loss)
                if soft_numeric.bad_span_loss is not None:
                    merge_losses.append(soft_numeric.bad_span_loss * NUMERIC_BAD_SPAN_LOSS_WEIGHT)
                merge_boundary_loss = torch.stack(merge_losses).mean()
                auxiliary_losses.append(merge_boundary_loss)
            else:
                merge_losses = []
                if soft_numeric.span_loss is not None:
                    merge_losses.append(soft_numeric.span_loss)
                if soft_numeric.bad_span_loss is not None:
                    merge_losses.append(soft_numeric.bad_span_loss * NUMERIC_BAD_SPAN_LOSS_WEIGHT)
                if merge_losses:
                    merge_boundary_loss = torch.stack(merge_losses).mean()
                    auxiliary_losses.append(merge_boundary_loss)
            if auxiliary_losses:
                token_type_loss = torch.stack(auxiliary_losses).mean()

        return ProductCardEncoderOutput(
            last_hidden_state=mixed_states,
            cls_state=cls_state,
            retrieval_embedding=retrieval_embedding,
            normalized_embedding=normalized_embedding,
            numeric_mask=numeric_mask,
            numeric_values=numeric_values,
            role_ids=role_ids.to(device=input_ids.device),
            slot_ids=slot_ids.to(device=input_ids.device),
            token_type_loss=token_type_loss,
            token_type_logits=token_type_logits,
            token_type_probabilities=token_type_probabilities,
            merge_boundary_loss=merge_boundary_loss,
            merge_join_logits=routing_output.join_logits,
            merge_join_probabilities=routing_output.join_probabilities,
            merged_token_texts=viterbi_token_texts,
        )


def build_bge_m3_product_card_encoder_from_tokenizer_json(
    tokenizer_path: str | Path,
    *,
    d_model: int = 1024,
    num_heads: int = 16,
    num_layers: int = 24,
    token_type_router_layers: int = 4,
    ff_multiplier: int = 4,
    max_position_embeddings: int = 8194,
    max_features: int = 14,
    local_attention_window: int = 128,
    max_routing_clusters: int = 128,
    dropout: float = 0.1,
    pad_token_id: int | None = None,
    bos_token_id: int | None = None,
    eos_token_id: int | None = None,
    retrieval_projection_dim: int | None = None,
) -> BgeM3ProductCardEncoder:
    tokenizer = load_tokenizer_from_json(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    vocab = tokenizer.get_vocab()
    id_to_token = [""] * vocab_size
    for token, token_id in vocab.items():
        if 0 <= int(token_id) < vocab_size:
            id_to_token[int(token_id)] = token

    if pad_token_id is None and hasattr(tokenizer, "token_to_id"):
        pad_token_id = tokenizer.token_to_id("<pad>")
    if bos_token_id is None and hasattr(tokenizer, "token_to_id"):
        bos_token_id = tokenizer.token_to_id("<s>")
        if bos_token_id is None:
            bos_token_id = tokenizer.token_to_id("<bos>")
    if eos_token_id is None and hasattr(tokenizer, "token_to_id"):
        eos_token_id = tokenizer.token_to_id("</s>")
        if eos_token_id is None:
            eos_token_id = tokenizer.token_to_id("<eos>")

    if pad_token_id is None:
        raise ValueError("pad_token_id could not be inferred from tokenizer.")
    if bos_token_id is None:
        raise ValueError("bos_token_id could not be inferred from tokenizer.")
    if eos_token_id is None:
        raise ValueError("eos_token_id could not be inferred from tokenizer.")

    return BgeM3ProductCardEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        token_type_router_layers=token_type_router_layers,
        ff_multiplier=ff_multiplier,
        max_position_embeddings=max_position_embeddings,
        max_features=max_features,
        local_attention_window=local_attention_window,
        max_routing_clusters=max_routing_clusters,
        dropout=dropout,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        id_to_token=id_to_token,
        retrieval_projection_dim=retrieval_projection_dim,
    )


def describe_product_encoder(model: BgeM3ProductCardEncoder) -> dict[str, int]:
    return {
        "vocab_size": model.vocab_size,
        "d_model": model.d_model,
        "num_heads": model.num_heads,
        "num_layers": model.num_layers,
        "token_type_router_layers": model.token_type_router_layers,
        "max_features": model.max_features,
        "local_attention_window": model.local_attention_window,
        "max_routing_clusters": model.max_routing_clusters,
        "parameter_count": count_parameters(model),
    }
