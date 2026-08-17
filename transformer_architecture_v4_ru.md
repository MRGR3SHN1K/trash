# Архитектура Dual Encoder V4

Документ фиксирует текущую V4-архитектуру после перехода от hard numeric merge к дифференцируемой numeric-сегментации.

Основные файлы:

- `hybrid_dual_path_model.py`
- `product_card_encoder.py`
- `dual_encoder_retrieval.py`
- `train_dual_encoder_on_parquet.py`

## 1. Верхний уровень

```text
query text      -> QueryEncoder   -> 1024-d L2-normalized query vector
product card    -> ProductEncoder -> 1024-d L2-normalized product vector

score(query, product) = cosine(query_vector, product_vector)
```

Обучение идет contrastive/in-batch negatives: для одного positive товара остальные товары батча выступают negative.

## 2. Дефолтная конфигурация

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

## 3. Главная идея V4

Раньше numeric merge физически менял sequence:

```text
6 0 0 -> один merged token 600
```

Это было hard/non-differentiable, потому что длина sequence менялась через Python/list logic.

В V4 sequence length больше не меняется. Модель оставляет raw tokenizer positions, но numeric branch получает embedding полного числа:

```text
raw tokens: 6 0 0
numeric branch: NumericEncoder(600)
```

То есть в числовую ветку идет представление значения `600`, а не три независимых значения `6`, `0`, `0`.

## 4. Дифференцируемый numeric merge

Для каждого непрерывного numeric-compatible run строится semi-Markov span lattice.

Пример:

```text
tokens: 6 0 0

candidate segmentations:
  [600]
  [60] [0]
  [6] [00]
  [6] [0] [0]
```

Модель предсказывает:

```text
p_int[i]   = P(token i относится к numeric path)
p_join[i]  = P(token i склеивается с token i+1)
```

Для каждого span считается score через join/split logits. Затем semi-Markov forward-backward дает posterior span probabilities.

Numeric embedding считается как:

```text
expected_span_embedding[i] =
  sum over spans covering i:
    posterior(span) * NumericEncoder(value(span))
```

Чтобы нерелевантные разбиения не шумели `600`, используется sharpened posterior:

```text
span_weights =
  0.98 * softmax(log_posterior / tau)
  + 0.02 * softmax(log_posterior)

tau = 0.05
```

Эти `0.02` нужны, чтобы retrieval loss давал живой градиент в `join_head`, но основной embedding почти полностью идет от top-span.

## 5. Loss для numeric merge

Auxiliary loss теперь состоит из нескольких частей:

```text
token_type CE
+ join BCE
+ semi-Markov CRF NLL по gold numeric segmentation
+ 0.1 * bad-span suppression
```

Gold segmentation строится эвристически как самый длинный валидный numeric span внутри numeric-run.

Примеры:

```text
6 0 0     -> gold [600]
2 , 5     -> gold [2,5]
- 7       -> gold [-7]
7 %       -> gold [7%]
```

Bad-span suppression штрафует posterior у альтернатив вроде:

```text
[60] [0]
[6] [00]
[6] [0] [0]
```

## 6. Token Type Router

Router полностью обучаемый:

```text
Token embeddings
  -> 4 x TokenTypeTransformerLayer
  -> type_head: STR / INT logits
  -> join_head: adjacent-token join logits
  -> route_ffn: residual signal for downstream encoder
```

Text-prior logits больше не добавляются в forward. Эвристика используется только как weak supervision target в auxiliary loss.

`join_head` имеет положительную bias-инициализацию, чтобы на старте внутри валидного numeric-run модель предпочитала цельный span (`600`), а не случайное разбиение (`60|0`).

## 7. Query Encoder

Query encoder использует full bidirectional attention.

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

Если характеристик больше 14, выбираются самые важные по ranking score:

```text
log1p(global_feature_frequency)
+ numeric_value_bonus
+ feature_name_in_title_bonus
```

## 9. Product Sparse Attention

Product encoder использует Longformer-style sparse attention:

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

`SEPARATOR` не global.

Routing clusters считаются в каждом слое:

```text
cluster_count = ceil(sqrt(active_tokens))
cluster_count <= max_routing_clusters
```

## 10. DualPathTransformerLayer

Каждый слой сохраняет две ветки STR/INT и cross-attention между ними:

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

Cross-attention STR <-> INT сохранен.

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

## 11. Позиционное кодирование

Learned absolute position embeddings удалены.

Позиция задается через RoPE внутри attention:

```text
RoPE(q), RoPE(k)
```

В product encoder дополнительно остаются learned role/slot embeddings.

## 12. Что изменилось относительно предыдущей версии

- Убран hard numeric merge из forward.
- Добавлен differentiable semi-Markov numeric span lattice.
- Numeric branch получает `NumericEncoder(value(span))`, например `NumericEncoder(600)`.
- Добавлен CRF NLL для правильной numeric segmentation.
- Добавлен bad-span suppression против разбиений `60|0`, `6|00`, `6|0|0`.
- Добавлен sharpened posterior с небольшой gradient-mix компонентой.
- Token type router стал learned-only в forward.
- Learned absolute positions заменены на RoPE.
- Product pooling стал attention-based.
- Product encoder получил Longformer-style local/global attention.
- Routing clusters добавлены в каждый product layer.
- Максимум характеристик поднят с 10 до 14.

## 13. Важное ограничение

В V4 число не становится физически одним token в hidden sequence. Sequence остается raw-token length. Это сделано намеренно: physical merge ломает дифференцируемость. При этом numeric branch получает embedding полного числа, а `merged_token_texts` используется как диагностический Viterbi-вывод.
