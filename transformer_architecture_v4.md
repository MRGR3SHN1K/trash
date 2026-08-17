# Dual Encoder Transformer Architecture V4

This document describes the current V4 architecture after replacing hard numeric merge with differentiable numeric segmentation.

Primary files:

- `hybrid_dual_path_model.py`
- `product_card_encoder.py`
- `dual_encoder_retrieval.py`
- `train_dual_encoder_on_parquet.py`

## 1. Top Level

```text
query text      -> QueryEncoder   -> 1024-d L2-normalized query vector
product card    -> ProductEncoder -> 1024-d L2-normalized product vector

score(query, product) = cosine(query_vector, product_vector)
```

Training uses contrastive in-batch negatives: for one positive product, the other products in the batch are negatives.

## 2. Default Configuration

```text
embedding_dim = 1024
temperature = 0.05

QueryEncoder:
  class = HybridDualPathTransformer
  target_parameters = 3_000_000_000
  num_layers = 30
  token_type_router_layers = 4
  num_heads = 16
  ff_multiplier = 4
  max_position_embeddings = 2048
  pooling = attention
  attention = full bidirectional

ProductEncoder:
  class = BgeM3ProductCardEncoder
  d_model = 1024
  num_layers = 30
  token_type_router_layers = 4
  num_heads = 16
  ff_multiplier = 4
  max_position_embeddings = 8194
  max_features = 14
  pooling = attention
  attention = Longformer-style local/global + routing clusters
  local_attention_window = 128
  max_routing_clusters = 128
```

## 3. Main V4 Idea

The previous numeric merge physically changed sequence length:

```text
6 0 0 -> one merged token 600
```

That was hard and non-differentiable because sequence length was changed through Python/list logic.

In V4, sequence length does not change. The model keeps raw tokenizer positions, but the numeric branch receives the embedding of the full numeric value:

```text
raw tokens: 6 0 0
numeric branch: NumericEncoder(600)
```

So the numeric branch sees value `600`, not three independent values `6`, `0`, and `0`.

## 4. Differentiable Numeric Merge

For each contiguous numeric-compatible run, the model builds a semi-Markov span lattice.

Example:

```text
tokens: 6 0 0

candidate segmentations:
  [600]
  [60] [0]
  [6] [00]
  [6] [0] [0]
```

The model predicts:

```text
p_int[i]   = P(token i belongs to the numeric path)
p_join[i]  = P(token i joins token i+1)
```

Each span receives a score from join/split logits. Semi-Markov forward-backward then produces posterior span probabilities.

The numeric embedding is computed as:

```text
expected_span_embedding[i] =
  sum over spans covering i:
    posterior(span) * NumericEncoder(value(span))
```

To prevent irrelevant segmentations from corrupting `600`, V4 uses a sharpened posterior:

```text
span_weights =
  0.98 * softmax(log_posterior / tau)
  + 0.02 * softmax(log_posterior)

tau = 0.05
```

The `0.02` gradient-mix term keeps retrieval-loss gradients alive for `join_head`; the actual representation is still dominated by the top span.

## 5. Numeric Merge Loss

The auxiliary loss now includes:

```text
token_type CE
+ join BCE
+ semi-Markov CRF NLL for gold numeric segmentation
+ 0.1 * bad-span suppression
```

Gold segmentation is built heuristically as the longest valid numeric span inside each numeric run.

Examples:

```text
6 0 0     -> gold [600]
2 , 5     -> gold [2,5]
- 7       -> gold [-7]
7 %       -> gold [7%]
```

Bad-span suppression penalizes posterior mass on alternatives such as:

```text
[60] [0]
[6] [00]
[6] [0] [0]
```

## 6. Token Type Router

The router is fully trainable:

```text
Token embeddings
  -> 4 x TokenTypeTransformerLayer
  -> type_head: STR / INT logits
  -> join_head: adjacent-token join logits
  -> route_ffn: residual signal for the downstream encoder
```

Text-prior logits are no longer added in forward. Heuristics are used only as weak-supervision targets in auxiliary losses.

`join_head` has a positive bias initialization so that, at startup, valid numeric runs prefer full spans such as `600` instead of random splits such as `60|0`.

## 7. Query Encoder

The query encoder uses full bidirectional attention.

Flow:

```text
raw input_ids
  -> token embeddings
  -> learned token type router
  -> differentiable semi-Markov numeric span lattice
  -> STR stream + INT stream
  -> 30 x DualPathTransformerLayer
  -> soft STR/INT merge
  -> final LayerNorm
  -> attention pooling
  -> optional projection to 1024
  -> L2 normalize
```

## 8. Product Encoder

Product input:

```text
<cls>
title tokens
<sep>
description tokens
<sep>
feature_1 name tokens
<sep>
feature_1 value tokens
<sep>
...
feature_14 name tokens
<sep>
feature_14 value tokens
<sep>
```

Embeddings:

```text
token_embedding
+ type_embedding
+ role_embedding
+ slot_embedding
```

Roles:

```text
CLS
TITLE
DESCRIPTION
FEATURE_NAME
FEATURE_VALUE
SEPARATOR
```

If there are more than 14 features, the selected features are ranked by:

```text
log1p(global_feature_frequency)
+ numeric_value_bonus
+ feature_name_in_title_bonus
```

## 9. Product Sparse Attention

The product encoder uses Longformer-style sparse attention:

```text
allowed attention =
  local window
  OR query/key token is global
  OR same routing cluster
```

Global tokens:

```text
CLS
TITLE
DESCRIPTION
FEATURE_NAME
```

`SEPARATOR` is not global.

Routing clusters are recomputed in every layer:

```text
cluster_count = ceil(sqrt(active_tokens))
cluster_count <= max_routing_clusters
```

## 10. DualPathTransformerLayer

Each layer keeps separate STR/INT streams and cross-attention between them:

```text
A_str = SelfAttention(norm(H_str))
A_int = SelfAttention(norm(H_int))

B_str = FF(norm(H_str + A_str))
B_int = FF(norm(H_int + A_int))

S_str = H_str + A_str + B_str
S_int = H_int + A_int + B_int

C_str = CrossAttention(norm(S_str), context=norm(S_int))
C_int = CrossAttention(norm(S_int), context=norm(S_str))

P_str = FF(norm(S_str + C_str))
P_int = FF(norm(S_int + C_int))

H_str_next = S_str + C_str + P_str
H_int_next = S_int + C_int + P_int
```

STR <-> INT cross-attention is preserved.

FFN:

```text
Linear
GELU
Dropout
Linear
GELU
Dropout
Linear
```

## 11. Positional Encoding

Learned absolute position embeddings were removed.

Position information is handled with RoPE inside attention:

```text
RoPE(q), RoPE(k)
```

The product encoder also keeps learned role/slot embeddings.

## 12. Changes From The Previous Version

- Removed hard numeric merge from forward.
- Added differentiable semi-Markov numeric span lattice.
- Numeric branch receives `NumericEncoder(value(span))`, for example `NumericEncoder(600)`.
- Added CRF NLL for correct numeric segmentation.
- Added bad-span suppression against splits such as `60|0`, `6|00`, `6|0|0`.
- Added sharpened posterior with a small gradient-mix component.
- Token type router is learned-only in forward.
- Learned absolute positions were replaced with RoPE.
- Product pooling is now attention-based.
- Product encoder now uses Longformer-style local/global attention.
- Routing clusters were added to every product layer.
- Maximum product features increased from 10 to 14.

## 13. Important Limitation

In V4, a number does not physically become one token in the hidden sequence. Sequence length remains raw-token length. This is intentional: physical merge would break differentiability. The numeric branch still receives the full-number embedding, and `merged_token_texts` is used as diagnostic Viterbi output.
