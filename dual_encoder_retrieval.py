from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from hybrid_dual_path_model import (
    HybridDualPathTransformer,
    HybridTransformerOutput,
    HybridModelSizeSpec,
    build_model_from_tokenizer_json,
    build_target_size_retrieval_model_from_tokenizer_json,
    contrastive_retrieval_loss,
    cosine_similarity_matrix,
    count_parameters,
)
from product_card_encoder import (
    BgeM3ProductCardEncoder,
    ProductCard,
    ProductCardEncoderOutput,
    build_bge_m3_product_card_encoder_from_tokenizer_json,
)


@dataclass(slots=True)
class DualEncoderOutput:
    query_output: HybridTransformerOutput
    product_output: ProductCardEncoderOutput
    similarity: Tensor
    loss: Tensor | None = None


@dataclass(slots=True)
class DualEncoderBuildSpec:
    tokenizer_path: str
    embedding_dim: int
    query_model_size: HybridModelSizeSpec
    product_d_model: int
    product_num_heads: int
    product_num_layers: int
    product_token_type_router_layers: int
    product_max_position_embeddings: int
    product_max_features: int
    product_local_attention_window: int
    product_max_routing_clusters: int


class ProductQueryDualEncoder(nn.Module):
    def __init__(
        self,
        query_encoder: HybridDualPathTransformer,
        product_encoder: BgeM3ProductCardEncoder,
        *,
        temperature: float = 0.05,
    ) -> None:
        super().__init__()
        self.query_encoder = query_encoder
        self.product_encoder = product_encoder
        self.temperature = temperature

        query_dim = self._encoder_output_dim(self.query_encoder)
        product_dim = self._encoder_output_dim(self.product_encoder)
        if query_dim != product_dim:
            raise ValueError(
                f"Query and product encoders must emit the same embedding dimension, got {query_dim} vs {product_dim}."
            )
        self.embedding_dim = query_dim

    def gradient_checkpointing_enable(self) -> None:
        if hasattr(self.query_encoder, "gradient_checkpointing_enable"):
            self.query_encoder.gradient_checkpointing_enable()
        if hasattr(self.product_encoder, "gradient_checkpointing_enable"):
            self.product_encoder.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self) -> None:
        if hasattr(self.query_encoder, "gradient_checkpointing_disable"):
            self.query_encoder.gradient_checkpointing_disable()
        if hasattr(self.product_encoder, "gradient_checkpointing_disable"):
            self.product_encoder.gradient_checkpointing_disable()

    @staticmethod
    def _encoder_output_dim(encoder: nn.Module) -> int:
        projection = getattr(encoder, "retrieval_projection", None)
        if projection is not None:
            return int(projection.out_features)
        return int(getattr(encoder, "d_model"))

    def encode_queries(
        self,
        queries: Sequence[str],
        tokenizer_path: str | Path,
        *,
        max_length: int | None = None,
        numeric_token_id: int | None = None,
        device: torch.device | str | None = None,
        compute_token_type_loss: bool = False,
    ) -> HybridTransformerOutput:
        return self.query_encoder.encode_texts_from_tokenizer_json(
            texts=queries,
            tokenizer_path=tokenizer_path,
            numeric_token_id=numeric_token_id,
            max_length=max_length,
            device=device,
            compute_token_type_loss=compute_token_type_loss,
        )

    def encode_products(
        self,
        products: Sequence[ProductCard | Mapping[str, Any]],
        tokenizer_path: str | Path,
        *,
        max_length: int | None = None,
        numeric_token_id: int | None = None,
        device: torch.device | str | None = None,
        compute_token_type_loss: bool = False,
    ) -> ProductCardEncoderOutput:
        return self.product_encoder.encode_cards_from_tokenizer_json(
            cards=products,
            tokenizer_path=tokenizer_path,
            numeric_token_id=numeric_token_id,
            max_length=max_length,
            device=device,
            compute_token_type_loss=compute_token_type_loss,
        )

    def forward(
        self,
        *,
        queries: Sequence[str] | None = None,
        products: Sequence[ProductCard | Mapping[str, Any]] | None = None,
        tokenizer_path: str | Path | None = None,
        query_max_length: int | None = None,
        product_max_length: int | None = None,
        numeric_token_id: int | None = None,
        device: torch.device | str | None = None,
        query_output: HybridTransformerOutput | None = None,
        product_output: ProductCardEncoderOutput | None = None,
        compute_loss: bool = True,
        symmetric_loss: bool = True,
        compute_token_type_loss: bool = False,
    ) -> DualEncoderOutput:
        if query_output is None or product_output is None:
            if queries is None or products is None or tokenizer_path is None:
                raise ValueError(
                    "Either precomputed query_output/product_output or raw queries/products with tokenizer_path must be provided."
                )
            query_output = self.encode_queries(
                queries=queries,
                tokenizer_path=tokenizer_path,
                max_length=query_max_length,
                numeric_token_id=numeric_token_id,
                device=device,
                compute_token_type_loss=compute_token_type_loss,
            )
            product_output = self.encode_products(
                products=products,
                tokenizer_path=tokenizer_path,
                max_length=product_max_length,
                numeric_token_id=numeric_token_id,
                device=device,
                compute_token_type_loss=compute_token_type_loss,
            )
        similarity = cosine_similarity_matrix(
            query_output.normalized_embedding,
            product_output.normalized_embedding,
        )
        loss = None
        if compute_loss:
            loss = contrastive_retrieval_loss(
                query_output.normalized_embedding,
                product_output.normalized_embedding,
                temperature=self.temperature,
                symmetric=symmetric_loss,
            )
        return DualEncoderOutput(
            query_output=query_output,
            product_output=product_output,
            similarity=similarity,
            loss=loss,
        )

    def score_pairs(
        self,
        queries: Sequence[str],
        products: Sequence[ProductCard | Mapping[str, Any]],
        tokenizer_path: str | Path,
        *,
        query_max_length: int | None = None,
        product_max_length: int | None = None,
        numeric_token_id: int | None = None,
        device: torch.device | str | None = None,
        compute_loss: bool = True,
        symmetric_loss: bool = True,
        compute_token_type_loss: bool = False,
    ) -> DualEncoderOutput:
        return self(
            queries=queries,
            products=products,
            tokenizer_path=tokenizer_path,
            query_max_length=query_max_length,
            product_max_length=product_max_length,
            numeric_token_id=numeric_token_id,
            device=device,
            query_output=None,
            product_output=None,
            compute_loss=compute_loss,
            symmetric_loss=symmetric_loss,
            compute_token_type_loss=compute_token_type_loss,
        )

    def describe(self) -> dict[str, int | float]:
        return {
            "embedding_dim": self.embedding_dim,
            "query_parameters": count_parameters(self.query_encoder),
            "product_parameters": count_parameters(self.product_encoder),
            "total_parameters": count_parameters(self),
            "temperature": float(self.temperature),
        }


def build_product_query_dual_encoder_from_tokenizer_json(
    tokenizer_path: str | Path,
    *,
    embedding_dim: int = 1024,
    query_target_parameters: int = 3_000_000_000,
    query_num_layers: int = 30,
    query_token_type_router_layers: int = 4,
    query_num_heads: int = 16,
    query_ff_multiplier: int = 4,
    query_max_position_embeddings: int = 2048,
    query_dropout: float = 0.1,
    product_d_model: int = 1024,
    product_num_heads: int = 16,
    product_num_layers: int = 30,
    product_token_type_router_layers: int = 4,
    product_ff_multiplier: int = 4,
    product_max_position_embeddings: int = 8194,
    product_max_features: int = 14,
    product_local_attention_window: int = 128,
    product_max_routing_clusters: int = 128,
    product_dropout: float = 0.1,
    temperature: float = 0.05,
) -> tuple[ProductQueryDualEncoder, DualEncoderBuildSpec]:
    query_encoder, query_spec = build_target_size_retrieval_model_from_tokenizer_json(
        tokenizer_path,
        target_parameters=query_target_parameters,
        num_layers=query_num_layers,
        token_type_router_layers=query_token_type_router_layers,
        num_heads=query_num_heads,
        ff_multiplier=query_ff_multiplier,
        max_position_embeddings=query_max_position_embeddings,
        dropout=query_dropout,
        pooling="attention",
        retrieval_projection_dim=embedding_dim,
    )

    product_encoder = build_bge_m3_product_card_encoder_from_tokenizer_json(
        tokenizer_path,
        d_model=product_d_model,
        num_heads=product_num_heads,
        num_layers=product_num_layers,
        token_type_router_layers=product_token_type_router_layers,
        ff_multiplier=product_ff_multiplier,
        max_position_embeddings=product_max_position_embeddings,
        max_features=product_max_features,
        local_attention_window=product_local_attention_window,
        max_routing_clusters=product_max_routing_clusters,
        dropout=product_dropout,
        retrieval_projection_dim=embedding_dim,
    )

    model = ProductQueryDualEncoder(
        query_encoder=query_encoder,
        product_encoder=product_encoder,
        temperature=temperature,
    )
    build_spec = DualEncoderBuildSpec(
        tokenizer_path=str(tokenizer_path),
        embedding_dim=embedding_dim,
        query_model_size=query_spec,
        product_d_model=product_d_model,
        product_num_heads=product_num_heads,
        product_num_layers=product_num_layers,
        product_token_type_router_layers=product_token_type_router_layers,
        product_max_position_embeddings=product_max_position_embeddings,
        product_max_features=product_max_features,
        product_local_attention_window=product_local_attention_window,
        product_max_routing_clusters=product_max_routing_clusters,
    )
    return model, build_spec


def build_small_dual_encoder_from_tokenizer_json(
    tokenizer_path: str | Path,
    *,
    embedding_dim: int = 256,
    temperature: float = 0.05,
) -> ProductQueryDualEncoder:
    query_encoder = build_model_from_tokenizer_json(
        tokenizer_path,
        d_model=256,
        num_heads=8,
        num_layers=4,
        max_position_embeddings=1024,
        output_logits=False,
        pooling="attention",
        retrieval_projection_dim=embedding_dim,
    )
    product_encoder = build_bge_m3_product_card_encoder_from_tokenizer_json(
        tokenizer_path,
        d_model=256,
        num_heads=8,
        num_layers=4,
        max_position_embeddings=1024,
        max_features=14,
        retrieval_projection_dim=embedding_dim,
    )
    return ProductQueryDualEncoder(
        query_encoder=query_encoder,
        product_encoder=product_encoder,
        temperature=temperature,
    )
