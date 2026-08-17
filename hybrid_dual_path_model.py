from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


NUMERIC_TOKEN_RE = re.compile(
    r"""
    ^[+-]?
    (?:
        (?:\d+(?:[.,]\d+)?)
        |
        (?:\d{1,3}(?:[ _]\d{3})+(?:[.,]\d+)?)
    )
    %?$
    """,
    flags=re.VERBOSE,
)

TOKEN_PREFIXES = ("▁", "Ġ", "##")
NUMERIC_SPAN_RE = re.compile(r"\d+(?:[ _]\d{3})*(?:[.,]\d+)?%?")
NUMERIC_SPAN_POSTERIOR_TEMPERATURE = 0.05
NUMERIC_BAD_SPAN_LOSS_WEIGHT = 0.1
NUMERIC_SPAN_GRADIENT_MIX = 0.02
TOKEN_PREFIXES = ("\u2581", "Ġ", "##", *TOKEN_PREFIXES)


@dataclass(slots=True)
class HybridTransformerOutput:
    last_hidden_state: Tensor
    string_hidden_state: Tensor
    numeric_hidden_state: Tensor
    pooled_state: Tensor
    retrieval_embedding: Tensor
    normalized_embedding: Tensor
    numeric_mask: Tensor
    numeric_values: Tensor
    logits: Tensor | None = None
    loss: Tensor | None = None
    token_type_loss: Tensor | None = None
    token_type_logits: Tensor | None = None
    token_type_probabilities: Tensor | None = None
    merge_boundary_loss: Tensor | None = None
    merge_join_logits: Tensor | None = None
    merge_join_probabilities: Tensor | None = None
    merged_token_texts: list[list[str]] | None = None


@dataclass(slots=True)
class HybridModelSizeSpec:
    vocab_size: int
    d_model: int
    num_heads: int
    num_layers: int
    token_type_router_layers: int
    ff_multiplier: int
    max_position_embeddings: int
    pooling: str
    output_logits: bool
    retrieval_projection_dim: int | None
    estimated_parameters: int


@dataclass(slots=True)
class MergedTokenSpan:
    token_type: str
    text: str
    token_id: int
    source_ids: tuple[int, ...]
    source_tokens: tuple[str, ...]
    start: int
    end: int
    numeric_value: float | None = None
    source_positions: tuple[int, ...] = ()


@dataclass(slots=True)
class PreparedHybridBatch:
    input_ids: Tensor
    attention_mask: Tensor
    token_texts: list[list[str]]
    merged_tokens: list[list[MergedTokenSpan]]
    numeric_mask: Tensor | None = None
    numeric_values: Tensor | None = None


@dataclass(slots=True)
class TokenTypeRoutingOutput:
    hidden_states: Tensor
    logits: Tensor
    probabilities: Tensor
    route_residual: Tensor
    join_logits: Tensor
    join_probabilities: Tensor


@dataclass(slots=True)
class SoftNumericSpanBatch:
    numeric_embeddings: Tensor
    numeric_values: Tensor
    numeric_mass: Tensor
    numeric_mask: Tensor
    span_loss: Tensor | None
    bad_span_loss: Tensor | None
    viterbi_token_texts: list[list[str]]


@dataclass(slots=True)
class NumericSpanCandidate:
    start: int
    end: int
    text: str
    value: float
    score: Tensor


class TokenTypeMaskBuilder:
    def __init__(self, id_to_token: Sequence[str] | None = None) -> None:
        self.id_to_token = list(id_to_token) if id_to_token is not None else None

    @classmethod
    def from_tokenizer_json(cls, path: str | Path) -> "TokenTypeMaskBuilder":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        id_to_token: dict[int, str] = {}

        for item in payload.get("added_tokens", []):
            token_id = int(item["id"])
            id_to_token[token_id] = str(item["content"])

        for token, token_id in payload["model"]["vocab"].items():
            id_to_token[int(token_id)] = token

        ordered = [""] * (max(id_to_token) + 1)
        for token_id, token in id_to_token.items():
            ordered[token_id] = token
        return cls(ordered)

    @staticmethod
    def normalize_token_text(token: str) -> str:
        token = token.strip()
        for prefix in TOKEN_PREFIXES:
            if token.startswith(prefix):
                token = token[len(prefix) :]
        return token.strip()

    @classmethod
    def parse_numeric_token(cls, token: str) -> tuple[bool, float]:
        token = cls.normalize_token_text(token)
        if not token or token.startswith("<") and token.endswith(">"):
            return False, 0.0

        token = token.replace("−", "-").replace("\u2212", "-").replace("_", "")
        token = token.replace(" ", "")
        token = token.rstrip("%")
        if not NUMERIC_TOKEN_RE.fullmatch(token):
            return False, 0.0

        numeric_token = token.replace(",", ".")
        try:
            return True, float(numeric_token)
        except ValueError:
            return False, 0.0

    def build_from_token_texts(
        self,
        token_texts: Sequence[Sequence[str]],
        attention_mask: Tensor | None = None,
        *,
        device: torch.device | None = None,
    ) -> tuple[Tensor, Tensor]:
        if device is None and attention_mask is not None:
            device = attention_mask.device
        batch_size = len(token_texts)
        seq_len = max((len(seq) for seq in token_texts), default=0)
        numeric_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        numeric_values = torch.zeros(batch_size, seq_len, dtype=torch.float32, device=device)

        for batch_index, sequence in enumerate(token_texts):
            for token_index, token in enumerate(sequence):
                is_numeric, value = self.parse_numeric_token(token)
                if is_numeric:
                    numeric_mask[batch_index, token_index] = True
                    numeric_values[batch_index, token_index] = value

        if attention_mask is not None:
            mask = attention_mask.to(dtype=torch.bool, device=device)
            numeric_mask = numeric_mask & mask
            numeric_values = numeric_values * mask.to(dtype=torch.float32)
        return numeric_mask, numeric_values

    def build_from_input_ids(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if self.id_to_token is None:
            raise ValueError("id_to_token mapping is required to infer numeric_mask from input_ids.")

        device = input_ids.device
        token_texts: list[list[str]] = []
        for row in input_ids.detach().cpu().tolist():
            token_texts.append(
                [self.id_to_token[token_id] if token_id < len(self.id_to_token) else "" for token_id in row]
            )
        return self.build_from_token_texts(token_texts, attention_mask=attention_mask, device=device)


def normalize_token_fragment(token: str) -> str:
    token = TokenTypeMaskBuilder.normalize_token_text(token)
    return token.replace("в€’", "-").replace("\u2212", "-")


def is_special_token_fragment(token: str) -> bool:
    token = token.strip()
    return bool(token) and token.startswith("<") and token.endswith(">")


def is_numeric_fragment_candidate(token: str) -> bool:
    token = normalize_token_fragment(token)
    if not token or is_special_token_fragment(token):
        return False
    return all(character.isdigit() or character in ".,+-_% " for character in token)


def build_token_type_prior_logits(
    token_texts: Sequence[Sequence[str]],
    attention_mask: Tensor,
    *,
    device: torch.device | None = None,
) -> Tensor:
    if device is None:
        device = attention_mask.device

    batch_size, seq_len = attention_mask.shape
    prior_logits = torch.zeros((batch_size, seq_len, 2), dtype=torch.float32, device=device)
    for batch_index in range(batch_size):
        for token_index, token in enumerate(token_texts[batch_index][:seq_len]):
            if not attention_mask[batch_index, token_index]:
                continue
            normalized = normalize_token_fragment(token)
            if is_special_token_fragment(normalized):
                prior_score = -4.0
            elif any(character.isdigit() for character in normalized) and is_numeric_fragment_candidate(normalized):
                prior_score = 4.0
            elif normalized in {",", ".", "+", "-", "_", "%"}:
                prior_score = 2.0
            elif is_numeric_fragment_candidate(normalized):
                prior_score = 1.0
            else:
                prior_score = -4.0
            prior_logits[batch_index, token_index, 1] = prior_score
            prior_logits[batch_index, token_index, 0] = -prior_score
    return prior_logits


def build_token_type_weak_targets(
    token_texts: Sequence[Sequence[str]],
    attention_mask: Tensor,
    *,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor]:
    if device is None:
        device = attention_mask.device

    batch_size, seq_len = attention_mask.shape
    targets = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
    target_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
    for batch_index in range(batch_size):
        for token_index, token in enumerate(token_texts[batch_index][:seq_len]):
            if not attention_mask[batch_index, token_index]:
                continue
            normalized = normalize_token_fragment(token)
            if not normalized or is_special_token_fragment(normalized):
                continue
            if any(character.isdigit() for character in normalized) and is_numeric_fragment_candidate(normalized):
                targets[batch_index, token_index] = 1
            elif normalized in {",", ".", "+", "-", "_", "%"}:
                targets[batch_index, token_index] = 1
            else:
                targets[batch_index, token_index] = 0
            target_mask[batch_index, token_index] = True
    return targets, target_mask


def build_merge_join_weak_targets(
    token_texts: Sequence[Sequence[str]],
    attention_mask: Tensor,
    *,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor]:
    if device is None:
        device = attention_mask.device

    batch_size, seq_len = attention_mask.shape
    if seq_len <= 1:
        return (
            torch.zeros((batch_size, 0), dtype=torch.float32, device=device),
            torch.zeros((batch_size, 0), dtype=torch.bool, device=device),
        )

    targets = torch.zeros((batch_size, seq_len - 1), dtype=torch.float32, device=device)
    target_mask = torch.zeros((batch_size, seq_len - 1), dtype=torch.bool, device=device)
    for batch_index in range(batch_size):
        row_tokens = token_texts[batch_index][:seq_len]
        for token_index in range(len(row_tokens) - 1):
            if not (attention_mask[batch_index, token_index] and attention_mask[batch_index, token_index + 1]):
                continue
            left = normalize_token_fragment(row_tokens[token_index])
            right = normalize_token_fragment(row_tokens[token_index + 1])
            if (
                not is_numeric_fragment_candidate(left)
                or not is_numeric_fragment_candidate(right)
                or is_special_token_fragment(left)
                or is_special_token_fragment(right)
            ):
                continue
            joined = left + right
            target_mask[batch_index, token_index] = True
            is_decimal_tail = left in {",", "."} and any(character.isdigit() for character in right)
            is_sign_prefix = left in {"+", "-"} and any(character.isdigit() for character in right)
            is_percent_suffix = right == "%" and any(character.isdigit() for character in left)
            targets[batch_index, token_index] = (
                1.0 if is_valid_numeric_partial_text(joined) or is_decimal_tail or is_sign_prefix or is_percent_suffix else 0.0
            )
    return targets, target_mask


def is_valid_numeric_partial_text(text: str) -> bool:
    normalized = text.replace("в€’", "-").replace("\u2212", "-")
    if not normalized:
        return False
    if any(not (character.isdigit() or character in ".,+-_% ") for character in normalized):
        return False
    sign_count = normalized.count("+") + normalized.count("-")
    if sign_count > 1:
        return False
    for sign in ("+", "-"):
        if sign in normalized and not normalized.startswith(sign):
            return False

    if normalized.count("%") > 1:
        return False
    if "%" in normalized and not normalized.endswith("%"):
        return False

    decimal_count = normalized.count(",") + normalized.count(".")
    if decimal_count > 1:
        return False

    core = normalized
    if core[:1] in "+-":
        core = core[1:]
    if core.endswith("%"):
        core = core[:-1]
    if not core:
        return False
    if core[:1] in ",.":
        return False
    collapsed = core.replace(" ", "").replace("_", "").replace(",", "").replace(".", "")
    return bool(collapsed) and collapsed.isdigit()


def parse_numeric_span(text: str) -> float | None:
    normalized = text.replace(" ", "").replace("_", "").replace(",", ".").rstrip("%")
    if not NUMERIC_TOKEN_RE.fullmatch(normalized):
        return None
    try:
        return float(normalized)
    except ValueError:
        return None


def _is_numeric_run_token(token: str) -> bool:
    normalized = normalize_token_fragment(token)
    return bool(normalized) and is_numeric_fragment_candidate(normalized) and not is_special_token_fragment(normalized)


def _iter_numeric_runs(
    token_texts: Sequence[str],
    active_mask: Sequence[bool],
    *,
    role_ids: Sequence[int] | None = None,
    slot_ids: Sequence[int] | None = None,
    separator_role_id: int | None = None,
) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    position = 0
    token_count = min(len(token_texts), len(active_mask))
    while position < token_count:
        if not active_mask[position] or not _is_numeric_run_token(token_texts[position]):
            position += 1
            continue
        if separator_role_id is not None and role_ids is not None and role_ids[position] == separator_role_id:
            position += 1
            continue

        start = position
        base_role = role_ids[position] if role_ids is not None else None
        base_slot = slot_ids[position] if slot_ids is not None else None
        position += 1
        while position < token_count:
            if not active_mask[position] or not _is_numeric_run_token(token_texts[position]):
                break
            if separator_role_id is not None and role_ids is not None and role_ids[position] == separator_role_id:
                break
            if role_ids is not None and role_ids[position] != base_role:
                break
            if slot_ids is not None and slot_ids[position] != base_slot:
                break
            position += 1
        runs.append((start, position))
    return runs


def _enumerate_numeric_span_candidates(
    fragments: Sequence[str],
    join_logits: Tensor,
    run_start: int,
) -> list[NumericSpanCandidate]:
    candidates: list[NumericSpanCandidate] = []
    if not fragments:
        return candidates

    join_logprob = F.logsigmoid(join_logits)
    split_logprob = F.logsigmoid(-join_logits)
    run_length = len(fragments)
    for start in range(run_length):
        candidate_text = ""
        for end in range(start + 1, run_length + 1):
            candidate_text += fragments[end - 1]
            if not is_valid_numeric_partial_text(candidate_text):
                break
            numeric_value = parse_numeric_span(candidate_text)
            if numeric_value is None:
                continue

            score = join_logits.new_zeros(())
            if end - start > 1:
                score = score + join_logprob[run_start + start : run_start + end - 1].sum()
            if end < run_length:
                score = score + split_logprob[run_start + end - 1]
            candidates.append(
                NumericSpanCandidate(
                    start=start,
                    end=end,
                    text=candidate_text,
                    value=numeric_value,
                    score=score,
                )
            )
    return candidates


def _build_gold_numeric_segmentation(
    fragments: Sequence[str],
) -> list[tuple[int, int, str, float]] | None:
    position = 0
    spans: list[tuple[int, int, str, float]] = []
    while position < len(fragments):
        best: tuple[int, int, str, float] | None = None
        candidate_text = ""
        for end in range(position + 1, len(fragments) + 1):
            candidate_text += fragments[end - 1]
            if not is_valid_numeric_partial_text(candidate_text):
                break
            numeric_value = parse_numeric_span(candidate_text)
            if numeric_value is not None:
                best = (position, end, candidate_text, numeric_value)
        if best is None:
            return None
        spans.append(best)
        position = best[1]
    return spans


def _semimarkov_forward_backward(
    candidates: Sequence[NumericSpanCandidate],
    run_length: int,
) -> tuple[Tensor | None, list[Tensor], list[list[int]], list[list[int]]]:
    if run_length <= 0:
        return None, [], [], []

    by_end: list[list[int]] = [[] for _ in range(run_length + 1)]
    by_start: list[list[int]] = [[] for _ in range(run_length + 1)]
    for index, candidate in enumerate(candidates):
        by_end[candidate.end].append(index)
        by_start[candidate.start].append(index)

    if candidates:
        neg_inf = candidates[0].score.new_tensor(-1.0e30)
    else:
        return None, [], by_start, by_end

    alpha = [neg_inf for _ in range(run_length + 1)]
    alpha[0] = candidates[0].score.new_zeros(())
    for end in range(1, run_length + 1):
        terms = [alpha[candidates[index].start] + candidates[index].score for index in by_end[end]]
        if terms:
            alpha[end] = torch.logsumexp(torch.stack(terms), dim=0)

    log_z = alpha[run_length]
    if not torch.isfinite(log_z) or bool((log_z <= neg_inf / 2).detach().cpu().item()):
        return None, [], by_start, by_end

    beta = [neg_inf for _ in range(run_length + 1)]
    beta[run_length] = candidates[0].score.new_zeros(())
    for start in range(run_length - 1, -1, -1):
        terms = [candidates[index].score + beta[candidates[index].end] for index in by_start[start]]
        if terms:
            beta[start] = torch.logsumexp(torch.stack(terms), dim=0)

    log_posteriors = [
        alpha[candidate.start] + candidate.score + beta[candidate.end] - log_z
        for candidate in candidates
    ]
    return log_z, log_posteriors, by_start, by_end


def _viterbi_numeric_segmentation(
    candidates: Sequence[NumericSpanCandidate],
    run_length: int,
    by_end: Sequence[Sequence[int]],
) -> list[int]:
    if run_length <= 0 or not candidates:
        return []

    neg_inf = -1.0e30
    best_scores = [neg_inf for _ in range(run_length + 1)]
    backpointers: list[int | None] = [None for _ in range(run_length + 1)]
    best_scores[0] = 0.0
    for end in range(1, run_length + 1):
        best_index: int | None = None
        best_score = neg_inf
        for index in by_end[end]:
            candidate = candidates[index]
            previous_score = best_scores[candidate.start]
            if previous_score <= neg_inf / 2:
                continue
            score = previous_score + float(candidate.score.detach().cpu().item())
            if score > best_score:
                best_score = score
                best_index = index
        best_scores[end] = best_score
        backpointers[end] = best_index

    if backpointers[run_length] is None:
        return []

    path: list[int] = []
    cursor = run_length
    while cursor > 0:
        index = backpointers[cursor]
        if index is None:
            return []
        path.append(index)
        cursor = candidates[index].start
    path.reverse()
    return path


def build_soft_numeric_span_batch(
    token_texts: Sequence[Sequence[str]],
    attention_mask: Tensor,
    token_type_probabilities: Tensor,
    join_logits: Tensor,
    numeric_encoder: NumericFeatureEncoder,
    *,
    role_ids: Tensor | None = None,
    slot_ids: Tensor | None = None,
    separator_role_id: int | None = None,
    posterior_temperature: float = 0.05,
    device: torch.device | None = None,
) -> SoftNumericSpanBatch:
    if device is None:
        device = attention_mask.device

    batch_size, seq_len = attention_mask.shape
    d_model = numeric_encoder.output_proj.out_features
    numeric_embeddings = token_type_probabilities.new_zeros((batch_size, seq_len, d_model))
    numeric_values = token_type_probabilities.new_zeros((batch_size, seq_len), dtype=torch.float32)
    numeric_mass = token_type_probabilities.new_zeros((batch_size, seq_len))
    numeric_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
    span_losses: list[Tensor] = []
    bad_span_losses: list[Tensor] = []
    viterbi_token_texts: list[list[str]] = []
    temperature = max(float(posterior_temperature), 1e-4)

    for batch_index in range(batch_size):
        sequence_length = int(attention_mask[batch_index].sum().item())
        row_tokens = list(token_texts[batch_index][:sequence_length])
        row_active = [True] * sequence_length
        row_role_ids = (
            role_ids[batch_index, :sequence_length].detach().cpu().tolist()
            if role_ids is not None
            else None
        )
        row_slot_ids = (
            slot_ids[batch_index, :sequence_length].detach().cpu().tolist()
            if slot_ids is not None
            else None
        )
        viterbi_segments: dict[int, tuple[int, str]] = {}
        runs = _iter_numeric_runs(
            row_tokens,
            row_active,
            role_ids=row_role_ids,
            slot_ids=row_slot_ids,
            separator_role_id=separator_role_id,
        )

        for run_start, run_end in runs:
            fragments = [normalize_token_fragment(token) for token in row_tokens[run_start:run_end]]
            candidates = _enumerate_numeric_span_candidates(
                fragments,
                join_logits[batch_index],
                run_start,
            )
            if not candidates:
                continue

            run_length = run_end - run_start
            log_z, log_posteriors, by_start, by_end = _semimarkov_forward_backward(candidates, run_length)
            if log_z is None or not log_posteriors:
                continue

            gold_spans = _build_gold_numeric_segmentation(fragments)
            gold_keys: set[tuple[int, int]] = set()
            if gold_spans is not None:
                gold_score = join_logits[batch_index].new_zeros(())
                for gold_start, gold_end, _gold_text, _gold_value in gold_spans:
                    for candidate in candidates:
                        if candidate.start == gold_start and candidate.end == gold_end:
                            gold_score = gold_score + candidate.score
                            gold_keys.add((candidate.start, candidate.end))
                            break
                if len(gold_keys) == len(gold_spans):
                    span_losses.append(log_z - gold_score)

            bad_log_posteriors = [
                log_posterior
                for candidate, log_posterior in zip(candidates, log_posteriors, strict=False)
                if (candidate.start, candidate.end) not in gold_keys
            ]
            if bad_log_posteriors:
                bad_span_losses.append(torch.exp(torch.stack(bad_log_posteriors)).mean())

            candidate_values = torch.tensor(
                [candidate.value for candidate in candidates],
                dtype=torch.float32,
                device=device,
            ).view(1, -1)
            candidate_mask = torch.ones_like(candidate_values, dtype=torch.bool, device=device)
            candidate_embeddings = numeric_encoder(candidate_values, candidate_mask).squeeze(0)
            candidate_value_tensor = candidate_values.squeeze(0).to(dtype=numeric_values.dtype)

            for relative_position in range(run_length):
                covering_indices = [
                    index
                    for index, candidate in enumerate(candidates)
                    if candidate.start <= relative_position < candidate.end
                ]
                if not covering_indices:
                    continue

                cover_log_posteriors = torch.stack([log_posteriors[index] for index in covering_indices])
                sharpened_weights = torch.softmax(cover_log_posteriors / temperature, dim=0)
                gradient_weights = torch.softmax(cover_log_posteriors, dim=0)
                span_weights = (
                    (1.0 - NUMERIC_SPAN_GRADIENT_MIX) * sharpened_weights
                    + NUMERIC_SPAN_GRADIENT_MIX * gradient_weights
                ).to(dtype=candidate_embeddings.dtype)
                selected_embeddings = candidate_embeddings[covering_indices]
                selected_values = candidate_value_tensor[covering_indices]
                absolute_position = run_start + relative_position
                int_mass = token_type_probabilities[batch_index, absolute_position, 1]
                cover_mass = torch.exp(torch.logsumexp(cover_log_posteriors, dim=0)).clamp(max=1.0)
                token_numeric_mass = int_mass * cover_mass.to(dtype=int_mass.dtype)
                numeric_embeddings[batch_index, absolute_position] = (
                    selected_embeddings * span_weights.unsqueeze(-1)
                ).sum(dim=0)
                numeric_values[batch_index, absolute_position] = (
                    selected_values * span_weights.to(dtype=selected_values.dtype)
                ).sum()
                numeric_mass[batch_index, absolute_position] = token_numeric_mass
                numeric_mask[batch_index, absolute_position] = True

            viterbi_path = _viterbi_numeric_segmentation(candidates, run_length, by_end)
            if viterbi_path:
                for index in viterbi_path:
                    candidate = candidates[index]
                    viterbi_segments[run_start + candidate.start] = (
                        run_start + candidate.end,
                        candidate.text,
                    )

        row_viterbi: list[str] = []
        position = 0
        while position < sequence_length:
            segment = viterbi_segments.get(position)
            if segment is None:
                row_viterbi.append(row_tokens[position])
                position += 1
                continue
            segment_end, segment_text = segment
            row_viterbi.append(segment_text)
            position = max(segment_end, position + 1)
        viterbi_token_texts.append(row_viterbi)

    span_loss = torch.stack(span_losses).mean() if span_losses else None
    bad_span_loss = torch.stack(bad_span_losses).mean() if bad_span_losses else None
    numeric_embeddings = numeric_embeddings * numeric_mask.unsqueeze(-1).to(dtype=numeric_embeddings.dtype)
    numeric_values = numeric_values * numeric_mask.to(dtype=numeric_values.dtype)
    numeric_mass = numeric_mass * numeric_mask.to(dtype=numeric_mass.dtype)
    return SoftNumericSpanBatch(
        numeric_embeddings=numeric_embeddings,
        numeric_values=numeric_values,
        numeric_mass=numeric_mass,
        numeric_mask=numeric_mask,
        span_loss=span_loss,
        bad_span_loss=bad_span_loss,
        viterbi_token_texts=viterbi_token_texts,
    )


def find_numeric_spans(text: str) -> list[tuple[int, int, str, float]]:
    spans: list[tuple[int, int, str, float]] = []
    for match in NUMERIC_SPAN_RE.finditer(text):
        span_text = match.group(0)
        numeric_value = parse_numeric_span(span_text)
        if numeric_value is None:
            continue
        spans.append((match.start(), match.end(), span_text, numeric_value))
    return spans


def find_numeric_span_index(start: int, end: int, spans: Sequence[tuple[int, int, str, float]]) -> int | None:
    if end <= start:
        return None

    for index, (span_start, span_end, _span_text, _span_value) in enumerate(spans):
        if start >= span_start and end <= span_end:
            return index

    for index, (span_start, span_end, _span_text, _span_value) in enumerate(spans):
        if end <= span_start or start >= span_end:
            continue
        return index

    return None


def merge_numeric_tokens(
    text: str,
    token_ids: Sequence[int],
    tokens: Sequence[str],
    offsets: Sequence[tuple[int, int]],
    *,
    numeric_token_id: int | None = None,
) -> list[MergedTokenSpan]:
    spans = find_numeric_spans(text)
    merged_tokens: list[MergedTokenSpan] = []
    open_numeric_span: int | None = None

    for position, (token, token_id, (start, end)) in enumerate(zip(tokens, token_ids, offsets, strict=False)):
        span_index = find_numeric_span_index(start, end, spans)

        if span_index is not None:
            span_start, span_end, span_text, numeric_value = spans[span_index]
            merged_token_id = numeric_token_id if numeric_token_id is not None else token_id
            if open_numeric_span == span_index and merged_tokens:
                previous = merged_tokens[-1]
                merged_tokens[-1] = MergedTokenSpan(
                    token_type="INT",
                    text=span_text,
                    token_id=previous.token_id,
                    source_ids=previous.source_ids + (token_id,),
                    source_tokens=previous.source_tokens + (token,),
                    start=span_start,
                    end=span_end,
                    numeric_value=numeric_value,
                    source_positions=previous.source_positions + (position,),
                )
            else:
                merged_tokens.append(
                    MergedTokenSpan(
                        token_type="INT",
                        text=span_text,
                        token_id=merged_token_id,
                        source_ids=(token_id,),
                        source_tokens=(token,),
                        start=span_start,
                        end=span_end,
                        numeric_value=numeric_value,
                        source_positions=(position,),
                    )
                )
                open_numeric_span = span_index
            continue

        open_numeric_span = None
        merged_tokens.append(
            MergedTokenSpan(
                token_type="STR",
                text=token,
                token_id=token_id,
                source_ids=(token_id,),
                source_tokens=(token,),
                start=start,
                end=end,
                numeric_value=None,
                source_positions=(position,),
            )
        )

    return merged_tokens


def merge_predicted_numeric_tokens(
    token_ids: Sequence[int],
    tokens: Sequence[str],
    hard_numeric_flags: Sequence[bool],
    *,
    hard_join_flags: Sequence[bool] | None = None,
    numeric_token_id: int | None = None,
) -> list[MergedTokenSpan]:
    merged_tokens: list[MergedTokenSpan] = []
    token_count = min(len(token_ids), len(tokens), len(hard_numeric_flags))
    position = 0

    while position < token_count:
        token_text = tokens[position]
        normalized_text = normalize_token_fragment(token_text)
        if (
            not hard_numeric_flags[position]
            or not is_numeric_fragment_candidate(normalized_text)
            or is_special_token_fragment(normalized_text)
        ):
            merged_tokens.append(
                MergedTokenSpan(
                    token_type="STR",
                    text=token_text,
                    token_id=token_ids[position],
                    source_ids=(token_ids[position],),
                    source_tokens=(token_text,),
                    start=position,
                    end=position + 1,
                    numeric_value=None,
                    source_positions=(position,),
                )
            )
            position += 1
            continue

        best_end: int | None = None
        best_text = ""
        best_numeric_value: float | None = None
        candidate_text = ""
        cursor = position

        while cursor < token_count:
            cursor_token = tokens[cursor]
            cursor_text = normalize_token_fragment(cursor_token)
            if (
                cursor > position
                and hard_join_flags is not None
                and (cursor - 1) < len(hard_join_flags)
                and not hard_join_flags[cursor - 1]
            ):
                break
            if (
                not hard_numeric_flags[cursor]
                or not is_numeric_fragment_candidate(cursor_text)
                or is_special_token_fragment(cursor_text)
            ):
                break

            next_candidate = candidate_text + cursor_text
            if not is_valid_numeric_partial_text(next_candidate):
                break
            candidate_text = next_candidate

            numeric_value = parse_numeric_span(candidate_text)
            if numeric_value is not None:
                best_end = cursor + 1
                best_text = candidate_text
                best_numeric_value = numeric_value
            cursor += 1

        if best_end is None:
            merged_tokens.append(
                MergedTokenSpan(
                    token_type="STR",
                    text=token_text,
                    token_id=token_ids[position],
                    source_ids=(token_ids[position],),
                    source_tokens=(token_text,),
                    start=position,
                    end=position + 1,
                    numeric_value=None,
                    source_positions=(position,),
                )
            )
            position += 1
            continue

        group_token_id = numeric_token_id if numeric_token_id is not None else token_ids[position]
        merged_tokens.append(
            MergedTokenSpan(
                token_type="INT",
                text=best_text,
                token_id=group_token_id,
                source_ids=tuple(token_ids[position:best_end]),
                source_tokens=tuple(tokens[position:best_end]),
                start=position,
                end=best_end,
                numeric_value=best_numeric_value,
                source_positions=tuple(range(position, best_end)),
            )
        )
        position = best_end

    return merged_tokens


def prepare_hybrid_batch_from_encodings(
    texts: Sequence[str],
    encodings: Sequence[Any],
    *,
    pad_token_id: int = 0,
    numeric_token_id: int | None = None,
    max_length: int | None = None,
    device: torch.device | str | None = None,
) -> PreparedHybridBatch:
    raw_batch: list[list[MergedTokenSpan]] = []
    token_texts: list[list[str]] = []

    for _text, encoding in zip(texts, encodings, strict=False):
        raw_tokens = [
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
            for index, (token_id, token) in enumerate(zip(encoding.ids, encoding.tokens, strict=False))
        ]
        if max_length is not None:
            raw_tokens = raw_tokens[:max_length]
        raw_batch.append(raw_tokens)
        token_texts.append([token.text for token in raw_tokens])

    batch_size = len(raw_batch)
    seq_len = max((len(tokens) for tokens in raw_batch), default=0)
    input_ids = torch.full((batch_size, seq_len), fill_value=pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)

    for batch_index, raw_tokens in enumerate(raw_batch):
        for token_index, raw_token in enumerate(raw_tokens):
            input_ids[batch_index, token_index] = raw_token.token_id
            attention_mask[batch_index, token_index] = True

    return PreparedHybridBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_texts=token_texts,
        merged_tokens=raw_batch,
    )


def prepare_hybrid_batch_from_tokenizer(
    texts: Sequence[str],
    tokenizer: Any,
    *,
    pad_token_id: int = 0,
    numeric_token_id: int | None = None,
    max_length: int | None = None,
    device: torch.device | str | None = None,
) -> PreparedHybridBatch:
    encodings = tokenizer.encode_batch(list(texts))
    return prepare_hybrid_batch_from_encodings(
        texts=texts,
        encodings=encodings,
        pad_token_id=pad_token_id,
        numeric_token_id=numeric_token_id,
        max_length=max_length,
        device=device,
    )


@lru_cache(maxsize=8)
def _load_tokenizer_from_json_cached(tokenizer_path: str) -> Any:
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise ImportError("tokenizers is required to prepare raw text batches for the hybrid encoder.") from exc
    return Tokenizer.from_file(tokenizer_path)


def load_tokenizer_from_json(tokenizer_path: str | Path) -> Any:
    return _load_tokenizer_from_json_cached(str(Path(tokenizer_path).resolve()))


def prepare_hybrid_batch_from_tokenizer_json(
    texts: Sequence[str],
    tokenizer_path: str | Path,
    *,
    pad_token_id: int = 0,
    numeric_token_id: int | None = None,
    max_length: int | None = None,
    device: torch.device | str | None = None,
) -> PreparedHybridBatch:
    tokenizer = load_tokenizer_from_json(tokenizer_path)
    return prepare_hybrid_batch_from_tokenizer(
        texts=texts,
        tokenizer=tokenizer,
        pad_token_id=pad_token_id,
        numeric_token_id=numeric_token_id,
        max_length=max_length,
        device=device,
    )


class MaskedSequenceBatchNorm1d(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, states: Tensor, mask: Tensor | None = None) -> Tensor:
        if states.dim() != 3:
            raise ValueError("states must have shape [batch, seq, hidden].")

        weight = self.weight.to(dtype=states.dtype, device=states.device)
        bias = self.bias.to(dtype=states.dtype, device=states.device)

        if mask is None:
            flat_states = states.reshape(-1, states.size(-1))
            if flat_states.size(0) == 0:
                return states
            mean = flat_states.float().mean(dim=0, keepdim=True)
            var = flat_states.float().var(dim=0, unbiased=False, keepdim=True)
            normalized = (flat_states.float() - mean) / torch.sqrt(var + self.eps)
            normalized = normalized.to(dtype=states.dtype)
            normalized = normalized * weight + bias
            return normalized.view_as(states)

        flat_mask = mask.to(dtype=torch.bool, device=states.device).reshape(-1)
        if not flat_mask.any():
            return states

        flat_states = states.reshape(-1, states.size(-1))
        valid_states = flat_states[flat_mask]
        if valid_states.size(0) == 0:
            return states

        mean = valid_states.float().mean(dim=0, keepdim=True)
        var = valid_states.float().var(dim=0, unbiased=False, keepdim=True)
        normalized_valid = (valid_states.float() - mean) / torch.sqrt(var + self.eps)
        normalized_valid = normalized_valid.to(dtype=states.dtype)
        normalized_valid = normalized_valid * weight + bias
        output = flat_states.clone()
        output[flat_mask] = normalized_valid
        return output.view_as(states)


class NumericFeatureEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_frequencies: int = 16,
        num_rbf_bins: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_frequencies = num_frequencies
        self.register_buffer("frequencies", 2.0 ** torch.arange(num_frequencies, dtype=torch.float32), persistent=False)
        self.rbf_centers = nn.Parameter(torch.linspace(-6.0, 6.0, steps=num_rbf_bins))
        self.rbf_log_width = nn.Parameter(torch.full((num_rbf_bins,), math.log(0.75)))
        feature_dim = 4 + 2 * num_frequencies + num_rbf_bins
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.input_proj = nn.Linear(feature_dim, d_model * 2)
        self.hidden_batch_norm = MaskedSequenceBatchNorm1d(d_model * 2)
        self.hidden_activation = nn.GELU()
        self.hidden_dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(d_model * 2, d_model)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, numeric_values: Tensor, numeric_mask: Tensor) -> Tensor:
        values = numeric_values.to(dtype=torch.float32)
        signed_log = torch.sign(values) * torch.log1p(values.abs())
        magnitudes = torch.log1p(values.abs())
        signs = torch.sign(values)
        is_zero = values.eq(0).to(dtype=torch.float32)

        phases = signed_log.unsqueeze(-1) * self.frequencies.view(1, 1, -1)
        sinusoidal = torch.cat((torch.sin(phases), torch.cos(phases)), dim=-1)

        widths = self.rbf_log_width.exp().view(1, 1, -1).clamp_min(1e-4)
        centers = self.rbf_centers.view(1, 1, -1)
        rbf = torch.exp(-((signed_log.unsqueeze(-1) - centers) ** 2) / (2 * widths.pow(2)))

        scalar = torch.stack((signed_log, magnitudes, signs, is_zero), dim=-1)
        features = torch.cat((scalar, sinusoidal, rbf), dim=-1)
        features = self.feature_norm(features)
        hidden = self.input_proj(features)
        hidden = self.hidden_batch_norm(hidden, numeric_mask)
        hidden = self.hidden_activation(hidden)
        hidden = self.hidden_dropout(hidden)
        encoded = self.output_proj(hidden)
        encoded = self.output_norm(encoded)
        return encoded * numeric_mask.unsqueeze(-1).to(dtype=encoded.dtype)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, ff_multiplier: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        hidden_dim = d_model * ff_multiplier
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class AttentionPooling(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.score = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1, bias=False),
        )

    def forward(self, states: Tensor, mask: Tensor) -> Tensor:
        # Compute pooling scores in fp32 for numerical stability under bf16 autocast.
        scores = self.score(self.norm(states)).squeeze(-1).float()
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = weights * mask.to(dtype=weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        weights = weights.to(dtype=states.dtype)
        return (states * weights.unsqueeze(-1)).sum(dim=1)


def _rotate_half(x: Tensor) -> Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def apply_rope(x: Tensor, *, base: float = 10000.0) -> Tensor:
    # x: [batch, heads, seq, head_dim]
    head_dim = x.size(-1)
    rotary_dim = head_dim if head_dim % 2 == 0 else head_dim - 1
    if rotary_dim <= 0 or x.size(-2) <= 1:
        return x
    positions = torch.arange(x.size(-2), device=x.device, dtype=torch.float32)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, device=x.device, dtype=torch.float32) / rotary_dim)
    )
    angles = torch.einsum("s,d->sd", positions, inv_freq)
    angles = torch.repeat_interleave(angles, repeats=2, dim=-1)
    cos = angles.cos().to(dtype=x.dtype).view(1, 1, x.size(-2), rotary_dim)
    sin = angles.sin().to(dtype=x.dtype).view(1, 1, x.size(-2), rotary_dim)
    rotated = (x[..., :rotary_dim] * cos) + (_rotate_half(x[..., :rotary_dim]) * sin)
    if rotary_dim == head_dim:
        return rotated
    return torch.cat((rotated, x[..., rotary_dim:]), dim=-1)


class RotaryMultiheadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, use_rope: bool = True) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = float(dropout)
        self.use_rope = use_rope
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _split_heads(self, states: Tensor) -> Tensor:
        batch_size, seq_len, _dim = states.shape
        return states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, states: Tensor) -> Tensor:
        batch_size, _heads, seq_len, _head_dim = states.shape
        return states.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(
        self,
        query_states: Tensor,
        context_states: Tensor,
        *,
        context_mask: Tensor,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        batch_size, query_len, _ = query_states.shape
        context_len = context_states.size(1)
        if context_len == 0:
            return query_states.new_zeros(query_states.shape)

        empty_context = ~context_mask.any(dim=1)
        safe_context_mask = context_mask
        if empty_context.any():
            safe_context_mask = context_mask.clone()
            safe_context_mask[empty_context, 0] = True

        q = self._split_heads(self.q_proj(query_states))
        k = self._split_heads(self.k_proj(context_states))
        v = self._split_heads(self.v_proj(context_states))
        if self.use_rope:
            q = apply_rope(q)
            k = apply_rope(k)

        additive_mask = query_states.new_zeros((batch_size, 1, query_len, context_len))
        additive_mask = additive_mask.masked_fill(~safe_context_mask[:, None, None, :], torch.finfo(query_states.dtype).min)
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                if attn_mask.dim() == 2:
                    additive_mask = additive_mask.masked_fill(attn_mask[None, None, :, :], torch.finfo(query_states.dtype).min)
                elif attn_mask.dim() == 3:
                    additive_mask = additive_mask.masked_fill(attn_mask[:, None, :, :], torch.finfo(query_states.dtype).min)
                elif attn_mask.dim() == 4:
                    additive_mask = additive_mask.masked_fill(attn_mask, torch.finfo(query_states.dtype).min)
                else:
                    raise ValueError("Unsupported boolean attn_mask rank.")
            else:
                if attn_mask.dim() == 2:
                    additive_mask = additive_mask + attn_mask[None, None, :, :].to(dtype=additive_mask.dtype)
                elif attn_mask.dim() == 3:
                    additive_mask = additive_mask + attn_mask[:, None, :, :].to(dtype=additive_mask.dtype)
                elif attn_mask.dim() == 4:
                    additive_mask = additive_mask + attn_mask.to(dtype=additive_mask.dtype)
                else:
                    raise ValueError("Unsupported additive attn_mask rank.")

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=additive_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attn_output = self._merge_heads(attn_output)
        attn_output = self.out_proj(attn_output)
        if empty_context.any():
            attn_output = attn_output.masked_fill(empty_context[:, None, None], 0.0)
        return attn_output


class MaskedAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.context_norm = nn.LayerNorm(d_model)
        self.attn = RotaryMultiheadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout, use_rope=True)
        self.batch_norm = MaskedSequenceBatchNorm1d(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query_states: Tensor,
        query_mask: Tensor,
        *,
        context_states: Tensor | None = None,
        context_mask: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        if context_states is None:
            context_states = query_states
            context_mask = query_mask
        if context_mask is None:
            raise ValueError("context_mask must be provided for cross attention.")

        normalized_query = self.query_norm(query_states)
        normalized_context = self.context_norm(context_states)
        attn_output = self.attn(
            normalized_query,
            normalized_context,
            context_mask=context_mask,
            attn_mask=attn_mask,
        )
        attn_output = self.batch_norm(attn_output, query_mask)
        attn_output = attn_output * query_mask.unsqueeze(-1).to(dtype=attn_output.dtype)
        return self.dropout(attn_output) * query_mask.unsqueeze(-1).to(dtype=attn_output.dtype)


class MaskedFeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, ff_multiplier: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, ff_multiplier=ff_multiplier, dropout=dropout)
        self.batch_norm = MaskedSequenceBatchNorm1d(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, states: Tensor, mask: Tensor) -> Tensor:
        ff_output = self.ff(self.norm(states))
        ff_output = self.batch_norm(ff_output, mask)
        ff_output = ff_output * mask.unsqueeze(-1).to(dtype=ff_output.dtype)
        return self.dropout(ff_output) * mask.unsqueeze(-1).to(dtype=ff_output.dtype)


class TokenTypeTransformerLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_multiplier: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MaskedAttentionBlock(d_model, num_heads, dropout=dropout)
        self.ffn = MaskedFeedForwardBlock(d_model, ff_multiplier=ff_multiplier, dropout=dropout)

    def forward(self, states: Tensor, mask: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        attn_delta = self.self_attn(
            states,
            mask,
            context_mask=mask,
            attn_mask=attn_mask,
        )
        ffn_delta = self.ffn(states + attn_delta, mask)
        updated = states + attn_delta + ffn_delta
        return updated * mask.unsqueeze(-1).to(dtype=updated.dtype)


class LearnedTokenTypeRouter(nn.Module):
    def __init__(
        self,
        d_model: int,
        *,
        num_heads: int,
        num_layers: int = 2,
        ff_multiplier: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TokenTypeTransformerLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    ff_multiplier=ff_multiplier,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.type_head = nn.Linear(d_model, 2)
        self.join_head = nn.Sequential(
            nn.LayerNorm(d_model * 4),
            nn.Linear(d_model * 4, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        nn.init.constant_(self.join_head[-1].bias, 2.0)
        self.route_ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        states: Tensor,
        attention_mask: Tensor,
        *,
        attn_mask: Tensor | None = None,
        use_gradient_checkpointing: bool = False,
    ) -> TokenTypeRoutingOutput:
        hidden_states = states
        for layer in self.layers:
            if use_gradient_checkpointing and self.training:
                hidden_states = checkpoint(
                    lambda x: layer(x, attention_mask, attn_mask=attn_mask),
                    hidden_states,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(hidden_states, attention_mask, attn_mask=attn_mask)

        hidden_states = self.output_norm(hidden_states)
        logits = self.type_head(hidden_states)
        probabilities = torch.softmax(logits, dim=-1)
        probabilities = probabilities * attention_mask.unsqueeze(-1).to(dtype=probabilities.dtype)
        if hidden_states.size(1) > 1:
            left_states = hidden_states[:, :-1, :]
            right_states = hidden_states[:, 1:, :]
            join_features = torch.cat(
                (
                    left_states,
                    right_states,
                    left_states * right_states,
                    (left_states - right_states).abs(),
                ),
                dim=-1,
            )
            join_logits = self.join_head(join_features).squeeze(-1)
            pair_mask = attention_mask[:, :-1] & attention_mask[:, 1:]
            join_probabilities = torch.sigmoid(join_logits) * pair_mask.to(dtype=join_logits.dtype)
        else:
            join_logits = hidden_states.new_zeros((hidden_states.size(0), 0))
            join_probabilities = hidden_states.new_zeros((hidden_states.size(0), 0))
        route_residual = self.route_ffn(hidden_states) * attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
        return TokenTypeRoutingOutput(
            hidden_states=hidden_states,
            logits=logits,
            probabilities=probabilities,
            route_residual=route_residual,
            join_logits=join_logits,
            join_probabilities=join_probabilities,
        )


def build_merged_token_batch(
    input_ids: Tensor,
    attention_mask: Tensor,
    token_texts: Sequence[Sequence[str]],
    hard_numeric_flags: Tensor,
    *,
    hard_join_flags: Tensor | None = None,
    pad_token_id: int,
    numeric_token_id: int | None = None,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, list[list[MergedTokenSpan]]]:
    if device is None:
        device = input_ids.device

    batch_groups: list[list[MergedTokenSpan]] = []
    for batch_index in range(input_ids.size(0)):
        sequence_length = int(attention_mask[batch_index].sum().item())
        groups = merge_predicted_numeric_tokens(
            token_ids=input_ids[batch_index, :sequence_length].detach().cpu().tolist(),
            tokens=list(token_texts[batch_index][:sequence_length]),
            hard_numeric_flags=hard_numeric_flags[batch_index, :sequence_length].detach().cpu().tolist(),
            hard_join_flags=(
                hard_join_flags[batch_index, : max(sequence_length - 1, 0)].detach().cpu().tolist()
                if hard_join_flags is not None
                else None
            ),
            numeric_token_id=numeric_token_id,
        )
        batch_groups.append(groups)

    merged_seq_len = max((len(groups) for groups in batch_groups), default=0)
    batch_size = len(batch_groups)
    merged_input_ids = torch.full((batch_size, merged_seq_len), pad_token_id, dtype=torch.long, device=device)
    merged_attention_mask = torch.zeros((batch_size, merged_seq_len), dtype=torch.bool, device=device)
    merged_numeric_mask = torch.zeros((batch_size, merged_seq_len), dtype=torch.bool, device=device)
    merged_numeric_values = torch.zeros((batch_size, merged_seq_len), dtype=torch.float32, device=device)

    for batch_index, groups in enumerate(batch_groups):
        for group_index, group in enumerate(groups):
            merged_input_ids[batch_index, group_index] = group.token_id
            merged_attention_mask[batch_index, group_index] = True
            if group.token_type == "INT":
                merged_numeric_mask[batch_index, group_index] = True
                merged_numeric_values[batch_index, group_index] = float(group.numeric_value or 0.0)

    return merged_input_ids, merged_attention_mask, merged_numeric_mask, merged_numeric_values, batch_groups


def aggregate_token_type_groups(
    groups_batch: Sequence[Sequence[MergedTokenSpan]],
    route_residual: Tensor,
    type_probabilities: Tensor,
    *,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor]:
    if device is None:
        device = route_residual.device

    batch_size = len(groups_batch)
    merged_seq_len = max((len(groups) for groups in groups_batch), default=0)
    d_model = route_residual.size(-1)
    group_residual = torch.zeros((batch_size, merged_seq_len, d_model), dtype=route_residual.dtype, device=device)
    group_probabilities = torch.zeros((batch_size, merged_seq_len, 2), dtype=type_probabilities.dtype, device=device)

    for batch_index, groups in enumerate(groups_batch):
        for group_index, group in enumerate(groups):
            source_positions = list(group.source_positions)
            if not source_positions:
                continue
            source_index_tensor = torch.tensor(source_positions, dtype=torch.long, device=device)
            source_residual = route_residual[batch_index, source_index_tensor]
            source_probs = type_probabilities[batch_index, source_index_tensor]
            class_index = 1 if group.token_type == "INT" else 0
            weights = source_probs[:, class_index]
            weights = weights / weights.sum().clamp_min(1e-6)
            group_residual[batch_index, group_index] = (source_residual * weights.unsqueeze(-1)).sum(dim=0)
            group_probabilities[batch_index, group_index] = source_probs.mean(dim=0)

    return group_residual, group_probabilities


class DualPathTransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_multiplier: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.string_self_attn = MaskedAttentionBlock(d_model, num_heads, dropout=dropout)
        self.numeric_self_attn = MaskedAttentionBlock(d_model, num_heads, dropout=dropout)
        self.string_ffn = MaskedFeedForwardBlock(d_model, ff_multiplier=ff_multiplier, dropout=dropout)
        self.numeric_ffn = MaskedFeedForwardBlock(d_model, ff_multiplier=ff_multiplier, dropout=dropout)

        self.numeric_to_string_cross = MaskedAttentionBlock(d_model, num_heads, dropout=dropout)
        self.string_to_numeric_cross = MaskedAttentionBlock(d_model, num_heads, dropout=dropout)
        self.string_post_cross_ffn = MaskedFeedForwardBlock(d_model, ff_multiplier=ff_multiplier, dropout=dropout)
        self.numeric_post_cross_ffn = MaskedFeedForwardBlock(d_model, ff_multiplier=ff_multiplier, dropout=dropout)

    def forward(
        self,
        string_states: Tensor,
        numeric_states: Tensor,
        *,
        string_mask: Tensor,
        numeric_mask: Tensor,
        attn_mask: Tensor | None = None,
        self_attn_mask: Tensor | None = None,
        cross_attn_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        resolved_self_attn_mask = self_attn_mask if self_attn_mask is not None else attn_mask
        resolved_cross_attn_mask = cross_attn_mask if cross_attn_mask is not None else attn_mask

        string_attn_delta = self.string_self_attn(
            string_states,
            string_mask,
            context_mask=string_mask,
            attn_mask=resolved_self_attn_mask,
        )
        numeric_attn_delta = self.numeric_self_attn(
            numeric_states,
            numeric_mask,
            context_mask=numeric_mask,
            attn_mask=resolved_self_attn_mask,
        )

        string_ffn_delta = self.string_ffn(string_states + string_attn_delta, string_mask)
        numeric_ffn_delta = self.numeric_ffn(numeric_states + numeric_attn_delta, numeric_mask)

        string_states = string_states + string_attn_delta + string_ffn_delta
        numeric_states = numeric_states + numeric_attn_delta + numeric_ffn_delta
        string_states = string_states * string_mask.unsqueeze(-1).to(dtype=string_states.dtype)
        numeric_states = numeric_states * numeric_mask.unsqueeze(-1).to(dtype=numeric_states.dtype)

        numeric_cross_delta = self.string_to_numeric_cross(
            numeric_states,
            numeric_mask,
            context_states=string_states,
            context_mask=string_mask,
            attn_mask=resolved_cross_attn_mask,
        )
        string_cross_delta = self.numeric_to_string_cross(
            string_states,
            string_mask,
            context_states=numeric_states,
            context_mask=numeric_mask,
            attn_mask=resolved_cross_attn_mask,
        )

        numeric_post_delta = self.numeric_post_cross_ffn(numeric_states + numeric_cross_delta, numeric_mask)
        string_post_delta = self.string_post_cross_ffn(string_states + string_cross_delta, string_mask)
        numeric_states = numeric_states + numeric_cross_delta + numeric_post_delta
        string_states = string_states + string_cross_delta + string_post_delta
        numeric_states = numeric_states * numeric_mask.unsqueeze(-1).to(dtype=numeric_states.dtype)
        string_states = string_states * string_mask.unsqueeze(-1).to(dtype=string_states.dtype)
        return string_states, numeric_states


class HybridDualPathTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 10,
        token_type_router_layers: int = 4,
        ff_multiplier: int = 4,
        max_position_embeddings: int = 2048,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        causal: bool = False,
        output_logits: bool = False,
        pooling: str = "attention",
        retrieval_projection_dim: int | None = None,
        mask_builder: TokenTypeMaskBuilder | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.token_type_router_layers = token_type_router_layers
        self.max_position_embeddings = max_position_embeddings
        self.pad_token_id = pad_token_id
        self.causal = causal
        self.output_logits = output_logits
        self.pooling = pooling
        self.mask_builder = mask_builder
        self.id_to_token = list(mask_builder.id_to_token) if mask_builder is not None and mask_builder.id_to_token else None
        self.gradient_checkpointing = False
        if self.pooling not in {"mean", "cls", "attention"}:
            raise ValueError("pooling must be one of: 'mean', 'cls', 'attention'.")

        self.token_embeddings = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.type_embeddings = nn.Embedding(2, d_model)
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
        self.attention_pooler = AttentionPooling(d_model, dropout=dropout) if pooling == "attention" else None
        self.retrieval_projection_dim = retrieval_projection_dim
        self.retrieval_projection = (
            nn.Linear(d_model, retrieval_projection_dim, bias=False)
            if retrieval_projection_dim is not None and retrieval_projection_dim != d_model
            else None
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False) if output_logits else None

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

    def _resolve_attention_mask(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None,
    ) -> Tensor:
        if attention_mask is None:
            attention_mask = input_ids.ne(self.pad_token_id)
        attention_mask = attention_mask.to(dtype=torch.bool, device=input_ids.device)
        return attention_mask

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
            raise ValueError("token_texts are required unless id_to_token lookup is attached to the model.")

        resolved = []
        for batch_index in range(input_ids.size(0)):
            sequence_length = int(attention_mask[batch_index].sum().item())
            row_tokens = []
            for token_id in input_ids[batch_index, :sequence_length].detach().cpu().tolist():
                row_tokens.append(self.id_to_token[token_id] if token_id < len(self.id_to_token) else "")
            resolved.append(row_tokens)
        return resolved

    def _build_attn_mask(self, seq_len: int, device: torch.device) -> Tensor | None:
        if not self.causal:
            return None
        return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)

    def _pool_states(self, states: Tensor, attention_mask: Tensor) -> Tensor:
        if self.pooling == "cls":
            return states[:, 0, :]
        if self.pooling == "attention":
            if self.attention_pooler is None:
                raise ValueError("attention_pooler is not initialized.")
            return self.attention_pooler(states, attention_mask)

        weights = attention_mask.unsqueeze(-1).to(dtype=states.dtype)
        summed = (states * weights).sum(dim=1)
        counts = weights.sum(dim=1).clamp_min(1.0)
        return summed / counts

    def prepare_inputs_from_texts(
        self,
        texts: Sequence[str],
        tokenizer: Any,
        *,
        numeric_token_id: int | None = None,
        max_length: int | None = None,
        device: torch.device | str | None = None,
    ) -> PreparedHybridBatch:
        return prepare_hybrid_batch_from_tokenizer(
            texts=texts,
            tokenizer=tokenizer,
            pad_token_id=self.pad_token_id,
            numeric_token_id=numeric_token_id,
            max_length=max_length,
            device=device if device is not None else self.token_embeddings.weight.device,
        )

    def encode_texts(
        self,
        texts: Sequence[str],
        tokenizer: Any,
        *,
        numeric_token_id: int | None = None,
        max_length: int | None = None,
        device: torch.device | str | None = None,
        compute_token_type_loss: bool = False,
    ) -> HybridTransformerOutput:
        prepared = self.prepare_inputs_from_texts(
            texts=texts,
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
            compute_token_type_loss=compute_token_type_loss,
        )

    def encode_texts_from_tokenizer_json(
        self,
        texts: Sequence[str],
        tokenizer_path: str | Path,
        *,
        numeric_token_id: int | None = None,
        max_length: int | None = None,
        device: torch.device | str | None = None,
        compute_token_type_loss: bool = False,
    ) -> HybridTransformerOutput:
        tokenizer = load_tokenizer_from_json(tokenizer_path)
        return self.encode_texts(
            texts=texts,
            tokenizer=tokenizer,
            numeric_token_id=numeric_token_id,
            max_length=max_length,
            device=device,
            compute_token_type_loss=compute_token_type_loss,
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        token_texts: Sequence[Sequence[str]] | None = None,
        numeric_token_id: int | None = None,
        numeric_mask: Tensor | None = None,
        numeric_values: Tensor | None = None,
        labels: Tensor | None = None,
        compute_token_type_loss: bool = False,
    ) -> HybridTransformerOutput:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape [batch, seq].")

        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_position_embeddings:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_position_embeddings={self.max_position_embeddings}."
            )

        attention_mask = self._resolve_attention_mask(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        resolved_token_texts = self._resolve_token_texts(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_texts=token_texts,
        )

        raw_token_emb = self.token_embeddings(input_ids)
        raw_base_states = self.dropout(raw_token_emb)

        raw_attn_mask = self._build_attn_mask(seq_len, input_ids.device)
        routing_output = self.token_type_router(
            raw_base_states,
            attention_mask,
            attn_mask=raw_attn_mask,
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
            posterior_temperature=NUMERIC_SPAN_POSTERIOR_TEMPERATURE,
            device=input_ids.device,
        )

        if numeric_mask is not None and numeric_values is not None:
            numeric_mask = numeric_mask.to(dtype=torch.bool, device=input_ids.device) & attention_mask
            numeric_values = numeric_values.to(dtype=torch.float32, device=input_ids.device)
            numeric_values = numeric_values[:, :seq_len] * numeric_mask.to(dtype=torch.float32)
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
        token_emb = self.token_embeddings(input_ids)
        string_base_states = self.dropout(token_emb + string_type_emb)
        string_weight = token_type_probabilities[..., 0] * attention_mask.to(dtype=token_type_probabilities.dtype)
        string_route = routing_output.route_residual * string_weight.unsqueeze(-1)
        numeric_route = routing_output.route_residual * numeric_mass.unsqueeze(-1)

        string_states = self.string_input_mlp(string_base_states + string_route)
        string_states = string_states * string_weight.unsqueeze(-1).to(dtype=string_states.dtype)

        numeric_states = self.numeric_input_gate(numeric_embeddings + numeric_type_emb + numeric_route)
        numeric_states = numeric_states * numeric_mass.unsqueeze(-1).to(dtype=numeric_states.dtype)

        attn_mask = self._build_attn_mask(seq_len, input_ids.device)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                string_states, numeric_states = checkpoint(
                    lambda s, n: layer(
                        s,
                        n,
                        string_mask=string_mask,
                        numeric_mask=numeric_attention_mask,
                        attn_mask=attn_mask,
                    ),
                    string_states,
                    numeric_states,
                    use_reentrant=False,
                )
            else:
                string_states, numeric_states = layer(
                    string_states,
                    numeric_states,
                    string_mask=string_mask,
                    numeric_mask=numeric_attention_mask,
                    attn_mask=attn_mask,
                )

        mixed_states = string_states + numeric_states
        route_weight = (string_weight + numeric_mass).clamp(max=1.0)
        mixed_states = mixed_states + routing_output.route_residual * route_weight.unsqueeze(-1)
        soft_type_emb = string_type_emb * string_weight.unsqueeze(-1) + numeric_type_emb * numeric_mass.unsqueeze(-1)
        mixed_states = mixed_states + soft_type_emb
        mixed_states = self.final_norm(mixed_states)
        mixed_states = mixed_states * attention_mask.unsqueeze(-1).to(dtype=mixed_states.dtype)

        pooled_state = self._pool_states(mixed_states, attention_mask)
        retrieval_embedding = (
            self.retrieval_projection(pooled_state) if self.retrieval_projection is not None else pooled_state
        )
        normalized_embedding = F.normalize(retrieval_embedding, p=2, dim=-1)

        logits = self.lm_head(mixed_states) if self.lm_head is not None else None
        loss = None
        token_type_loss = None
        merge_boundary_loss = None
        if labels is not None and logits is not None:
            if self.causal:
                shifted_logits = logits[:, :-1, :].contiguous()
                shifted_labels = labels[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shifted_logits.view(-1, shifted_logits.size(-1)),
                    shifted_labels.view(-1),
                    ignore_index=-100,
                )
            else:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)

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

        return HybridTransformerOutput(
            last_hidden_state=mixed_states,
            string_hidden_state=string_states,
            numeric_hidden_state=numeric_states,
            pooled_state=pooled_state,
            retrieval_embedding=retrieval_embedding,
            normalized_embedding=normalized_embedding,
            numeric_mask=numeric_mask,
            numeric_values=numeric_values,
            logits=logits,
            loss=loss,
            token_type_loss=token_type_loss,
            token_type_logits=token_type_logits,
            token_type_probabilities=token_type_probabilities,
            merge_boundary_loss=merge_boundary_loss,
            merge_join_logits=routing_output.join_logits,
            merge_join_probabilities=routing_output.join_probabilities,
            merged_token_texts=viterbi_token_texts,
        )


def build_model_from_tokenizer_json(
    tokenizer_path: str | Path,
    *,
    d_model: int = 512,
    num_heads: int = 8,
    num_layers: int = 10,
    token_type_router_layers: int = 4,
    ff_multiplier: int = 4,
    max_position_embeddings: int = 2048,
    dropout: float = 0.1,
    pad_token_id: int = 0,
    causal: bool = False,
    output_logits: bool = False,
    pooling: str = "attention",
    retrieval_projection_dim: int | None = None,
) -> HybridDualPathTransformer:
    mask_builder = TokenTypeMaskBuilder.from_tokenizer_json(tokenizer_path)
    vocab_size = len(mask_builder.id_to_token or [])
    return HybridDualPathTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        token_type_router_layers=token_type_router_layers,
        ff_multiplier=ff_multiplier,
        max_position_embeddings=max_position_embeddings,
        dropout=dropout,
        pad_token_id=pad_token_id,
        causal=causal,
        output_logits=output_logits,
        pooling=pooling,
        retrieval_projection_dim=retrieval_projection_dim,
        mask_builder=mask_builder,
    )


def build_numeric_mask_from_token_texts(
    token_texts: Sequence[Sequence[str]],
    attention_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    return TokenTypeMaskBuilder().build_from_token_texts(token_texts, attention_mask=attention_mask)


def iter_numeric_tokens(tokens: Iterable[str]) -> list[tuple[str, float]]:
    numeric_tokens: list[tuple[str, float]] = []
    for token in tokens:
        is_numeric, value = TokenTypeMaskBuilder.parse_numeric_token(token)
        if is_numeric:
            numeric_tokens.append((token, value))
    return numeric_tokens


def count_parameters(module: nn.Module, *, trainable_only: bool = True) -> int:
    params = module.parameters()
    if trainable_only:
        params = (parameter for parameter in params if parameter.requires_grad)
    return sum(parameter.numel() for parameter in params)


def cosine_similarity_matrix(left: Tensor, right: Tensor) -> Tensor:
    left = F.normalize(left, p=2, dim=-1)
    right = F.normalize(right, p=2, dim=-1)
    return left @ right.transpose(-1, -2)


def contrastive_retrieval_loss(
    query_embeddings: Tensor,
    target_embeddings: Tensor,
    *,
    temperature: float = 0.05,
    symmetric: bool = True,
) -> Tensor:
    if query_embeddings.size(0) != target_embeddings.size(0):
        raise ValueError("query_embeddings and target_embeddings must have the same batch size.")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    similarity = cosine_similarity_matrix(query_embeddings, target_embeddings) / temperature
    labels = torch.arange(similarity.size(0), device=similarity.device)
    query_to_target = F.cross_entropy(similarity, labels)
    if not symmetric:
        return query_to_target

    target_to_query = F.cross_entropy(similarity.transpose(0, 1), labels)
    return 0.5 * (query_to_target + target_to_query)


def estimate_hybrid_parameter_count(
    *,
    vocab_size: int,
    d_model: int,
    num_layers: int = 10,
    token_type_router_layers: int = 4,
    ff_multiplier: int = 4,
    max_position_embeddings: int = 2048,
    pooling: str = "attention",
    output_logits: bool = False,
    retrieval_projection_dim: int | None = None,
    num_frequencies: int = 16,
    num_rbf_bins: int = 16,
) -> int:
    feature_dim = 4 + 2 * num_frequencies + num_rbf_bins

    token_embeddings = vocab_size * d_model
    _max_position_embeddings = max_position_embeddings  # RoPE has no learned absolute position table.
    type_embeddings = 2 * d_model

    string_input_mlp = 2 * d_model * d_model + 4 * d_model
    numeric_encoder = (
        2 * num_rbf_bins
        + feature_dim * (2 * d_model)
        + (2 * d_model)
        + (2 * d_model * d_model)
        + d_model
    )
    numeric_input_gate = 2 * d_model * d_model + 4 * d_model

    attention_block = 4 * d_model * d_model + 6 * d_model
    token_type_ff_block = ((2 * 2 + 2 * 2) * d_model * d_model) + ((2 * 2 + 3) * d_model)
    token_type_layer = attention_block + token_type_ff_block
    token_type_type_head = 4 * d_model + 2
    token_type_join_head = 4 * d_model * d_model + 10 * d_model + 1
    token_type_route_head = 2 * d_model * d_model + 4 * d_model
    token_type_router = (
        token_type_router_layers * token_type_layer
        + (2 * d_model)
        + token_type_type_head
        + token_type_join_head
        + token_type_route_head
    )
    ff_block = (
        (ff_multiplier * ff_multiplier + 2 * ff_multiplier) * d_model * d_model
        + (2 * ff_multiplier + 3) * d_model
    )
    layer_total = 4 * attention_block + 4 * ff_block

    final_norm = 2 * d_model
    attention_pooler = d_model * d_model + 4 * d_model if pooling == "attention" else 0
    retrieval_projection = 0
    if retrieval_projection_dim is not None and retrieval_projection_dim != d_model:
        retrieval_projection = d_model * retrieval_projection_dim
    lm_head = vocab_size * d_model if output_logits else 0

    return (
        token_embeddings
        + type_embeddings
        + token_type_router
        + string_input_mlp
        + numeric_encoder
        + numeric_input_gate
        + num_layers * layer_total
        + final_norm
        + attention_pooler
        + retrieval_projection
        + lm_head
    )


def suggest_hybrid_retrieval_config(
    *,
    vocab_size: int,
    target_parameters: int = 2_000_000_000,
    num_layers: int = 10,
    token_type_router_layers: int = 4,
    num_heads: int = 16,
    ff_multiplier: int = 4,
    max_position_embeddings: int = 2048,
    pooling: str = "attention",
    output_logits: bool = False,
    retrieval_projection_dim: int | None = None,
    d_model_multiple: int = 64,
    min_d_model: int = 512,
    max_d_model: int = 4096,
) -> HybridModelSizeSpec:
    best_spec: HybridModelSizeSpec | None = None
    step = d_model_multiple

    for d_model in range(min_d_model, max_d_model + 1, step):
        if d_model % num_heads != 0:
            continue

        estimated_parameters = estimate_hybrid_parameter_count(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            token_type_router_layers=token_type_router_layers,
            ff_multiplier=ff_multiplier,
            max_position_embeddings=max_position_embeddings,
            pooling=pooling,
            output_logits=output_logits,
            retrieval_projection_dim=retrieval_projection_dim,
        )
        spec = HybridModelSizeSpec(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            token_type_router_layers=token_type_router_layers,
            ff_multiplier=ff_multiplier,
            max_position_embeddings=max_position_embeddings,
            pooling=pooling,
            output_logits=output_logits,
            retrieval_projection_dim=retrieval_projection_dim,
            estimated_parameters=estimated_parameters,
        )

        if best_spec is None or abs(spec.estimated_parameters - target_parameters) < abs(
            best_spec.estimated_parameters - target_parameters
        ):
            best_spec = spec

    if best_spec is None:
        raise ValueError("Could not find a valid d_model for the requested constraints.")
    return best_spec


def suggest_hybrid_retrieval_config_from_tokenizer_json(
    tokenizer_path: str | Path,
    *,
    target_parameters: int = 2_000_000_000,
    num_layers: int = 10,
    token_type_router_layers: int = 4,
    num_heads: int = 16,
    ff_multiplier: int = 4,
    max_position_embeddings: int = 2048,
    pooling: str = "attention",
    output_logits: bool = False,
    retrieval_projection_dim: int | None = None,
) -> HybridModelSizeSpec:
    mask_builder = TokenTypeMaskBuilder.from_tokenizer_json(tokenizer_path)
    vocab_size = len(mask_builder.id_to_token or [])
    return suggest_hybrid_retrieval_config(
        vocab_size=vocab_size,
        target_parameters=target_parameters,
        num_layers=num_layers,
        token_type_router_layers=token_type_router_layers,
        num_heads=num_heads,
        ff_multiplier=ff_multiplier,
        max_position_embeddings=max_position_embeddings,
        pooling=pooling,
        output_logits=output_logits,
        retrieval_projection_dim=retrieval_projection_dim,
    )


def build_target_size_retrieval_model_from_tokenizer_json(
    tokenizer_path: str | Path,
    *,
    target_parameters: int = 2_000_000_000,
    num_layers: int = 10,
    token_type_router_layers: int = 4,
    num_heads: int = 16,
    ff_multiplier: int = 4,
    max_position_embeddings: int = 2048,
    dropout: float = 0.1,
    pad_token_id: int = 0,
    pooling: str = "attention",
    retrieval_projection_dim: int | None = None,
) -> tuple[HybridDualPathTransformer, HybridModelSizeSpec]:
    spec = suggest_hybrid_retrieval_config_from_tokenizer_json(
        tokenizer_path,
        target_parameters=target_parameters,
        num_layers=num_layers,
        token_type_router_layers=token_type_router_layers,
        num_heads=num_heads,
        ff_multiplier=ff_multiplier,
        max_position_embeddings=max_position_embeddings,
        pooling=pooling,
        output_logits=False,
        retrieval_projection_dim=retrieval_projection_dim,
    )
    model = build_model_from_tokenizer_json(
        tokenizer_path,
        d_model=spec.d_model,
        num_heads=spec.num_heads,
        num_layers=spec.num_layers,
        token_type_router_layers=spec.token_type_router_layers,
        ff_multiplier=spec.ff_multiplier,
        max_position_embeddings=spec.max_position_embeddings,
        dropout=dropout,
        pad_token_id=pad_token_id,
        causal=False,
        output_logits=False,
        pooling=pooling,
        retrieval_projection_dim=retrieval_projection_dim,
    )
    return model, spec
