# Dual Encoder для поиска товаров по закупочным запросам

Проект собирает retrieval-пайплайн для поиска подходящих карточек товаров по текстовому запросу закупки. Финальная цель: получить два энкодера, которые переводят запрос и карточку товара в нормализованные векторы одинаковой размерности, после чего товары ранжируются по cosine similarity.

Текущая основная версия архитектуры: `V4`.

## Что решает проект

Входные данные:

- текстовый запрос закупки;
- база карточек товаров из всех файлов вида `data<digits>.parquet`, например `data0001.parquet ... data0029.parquet ... data0030.parquet`;
- у товара есть `title`, опциональное описание и JSON-поле `features`;
- в `sku` лежит до 5 текстовых запросов, которые считаются позитивными формулировками для товара;
- строки с `sku_error != None` исключаются из обучения.

Выход:

- `query_vector`: 1024-d L2-normalized embedding запроса;
- `product_vector`: 1024-d L2-normalized embedding карточки товара;
- `score = cosine(query_vector, product_vector)`;
- top-k/top-p выдача товаров;
- LLM-оценка релевантности кандидатов;
- метрики retrieval-качества и match-файлы вида `запрос -> подходящий товар`.

## Основные файлы

| Файл | Назначение |
| --- | --- |
| `hybrid_dual_path_model.py` | Query encoder, общий dual-path INT/STR механизм, numeric encoder, token type router, cross-attention, transformer layers. |
| `product_card_encoder.py` | Product/card encoder: title, description, feature-name/value slots, Longformer-style attention, routing clusters. |
| `dual_encoder_retrieval.py` | Обертка dual encoder: query encoder + product encoder, cosine scores, contrastive loss. |
| `train_dual_encoder_on_parquet.py` | Подготовка parquet-корпуса, train/valid split, обучение, FSDP, dynamic batching, checkpoints. |
| `continue_training_from_checkpoint.py` | Продолжение обучения с checkpoint-а с восстановлением optimizer state. |
| `vectorize_dual_encoder_catalog.py` | Векторизация всей базы товаров, single/multi-GPU, torchrun, динамический batch, catalog cache. |
| `evaluate_dual_encoder_with_llm.py` | Eval: retrieval по базе, LLM judge через scheduler, furniture/all статистика, метрики, match-файлы. |
| `interactive_catalog_search.py` | Онлайн-поиск top-10/top-k товаров по текстовому запросу. |
| `test_scheduler_llm.py` | Smoke-test scheduler WebSocket и LLM-инстансов. |
| `data_cleaner.py` | Исторический код взаимодействия с LLM/scheduler для разметки данных. |
| `transformer_architecture_v4_ru.md` | Подробное описание V4-архитектуры на русском. |
| `transformer_architecture_v4.md` | Английская версия описания V4. |

## Текущая архитектура V4

Общая схема:

```text
text query   -> QueryEncoder   -> 1024-d normalized vector
product card -> ProductEncoder -> 1024-d normalized vector

score(query, product) = cosine(query_vector, product_vector)
```

Текущие дефолты:

```text
embedding_dim = 1024
temperature = 0.05

QueryEncoder:
  class = HybridDualPathTransformer
  target_parameters = 3B
  transformer_layers = 30
  token_type_router_layers = 4
  attention = full bidirectional
  heads = 16
  ff_multiplier = 4
  max_position_embeddings = 2048
  pooling = attention

ProductEncoder:
  class = BgeM3ProductCardEncoder
  d_model = 1024
  transformer_layers = 30
  token_type_router_layers = 4
  max_features = 14
  attention = Longformer-style local/global + routing clusters
  local_attention_window = 128
  max_routing_clusters = 128
  pooling = attention
```

При текущих больших дефолтах суммарный размер двух энкодеров получается около 6B параметров. На практике обучение такой модели предполагает multi-GPU/FSDP.

## INT/STR dual-path

Идея модели: токены делятся на два потока.

- STR path отвечает за обычный текст.
- INT path отвечает за числа и числовые значения.
- Между потоками есть cross-attention.
- В конце состояния снова собираются в общий embedding.

Важная деталь V4: token type router полностью обучаемый. Эвристика используется как weak supervision target для auxiliary loss, но не подмешивается как жесткий prior в forward.

```text
raw tokenizer tokens
  -> token embeddings
  -> 4 x TokenTypeTransformerLayer
  -> STR/INT probabilities
  -> join probabilities
  -> STR stream + INT stream
  -> transformer/cross-attention blocks
  -> soft merge
  -> attention pooling
  -> normalized retrieval vector
```

## Дифференцируемый numeric merge

Проблема старого пайплайна: tokenizer мог разбить `600` как `6 0 0`, и hard merge в Python ломал дифференцируемость.

В V4 raw sequence length не меняется. Модель строит numeric span lattice и дает numeric path embedding полного числа:

```text
raw tokens:      6 0 0
numeric branch: NumericEncoder(600)
```

Для run-а numeric-compatible токенов строятся кандидаты:

```text
6 0 0 -> [600], [60][0], [6][00], [6][0][0]
2 , 5 -> [2,5], [2][,][5], ...
```

Модель предсказывает:

- `p_int[i]`: вероятность, что токен идет в numeric path;
- `p_join[i]`: вероятность склейки токена `i` с `i+1`;
- posterior по numeric spans через semi-Markov forward-backward.

Numeric embedding считается как weighted sum по span-кандидатам. Чтобы нерелевантные разбиения не зашумляли число, используется sharpened posterior:

```text
span_weights =
  0.98 * softmax(log_posterior / 0.05)
  + 0.02 * softmax(log_posterior)
```

`0.02` оставляет живой gradient path от retrieval loss к `join_head`, но основной embedding почти полностью идет от top-span.

Auxiliary numeric loss:

```text
token_type CE
+ join BCE
+ semi-Markov CRF NLL по gold numeric segmentation
+ bad-span suppression
```

## Product encoder

Карточка товара кодируется как последовательность:

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

Используются дополнительные embeddings:

- token embedding;
- STR/INT type embedding;
- role embedding: CLS, title, description, feature_name, feature_value, separator;
- slot embedding: номер характеристики.

Global tokens в product encoder:

- CLS;
- title tokens;
- description tokens;
- feature-name tokens.

Separator tokens не global.

Если характеристик больше 14, выбираются самые важные по ranking-логике:

- частота характеристики в корпусе;
- бонус за numeric value;
- бонус за встречаемость имени характеристики в title.

## Attention и routing

Query encoder использует full bidirectional attention.

Product encoder использует:

- Longformer-style local attention;
- global attention для важных ролей;
- routing clusters в каждом слое;
- число кластеров выбирается как `ceil(sqrt(n))`, где `n` - число активных непаддинговых product tokens, с верхним cap `max_routing_clusters`.

Transformer block устроен как pre-norm residual:

```text
H_{n+1} = H_n + resblock(H_n)

resblock(H):
  a(H) = dropout(TransformerAttention(norm(H)))
  b(H) = dropout(FF(norm(H + a(H))))
  return a(H) + b(H)
```

Feed-forward:

```text
Linear -> GELU -> Dropout -> Linear -> GELU -> Dropout -> Linear
```

## Данные и подготовка корпуса

Скрипт обучения принимает папку с parquet-файлами:

```text
data0001.parquet
data0002.parquet
...
data0029.parquet
data0030.parquet
...
```

Из них читаются:

- `title`: текст карточки;
- `features`: JSON уровня вложенности 1;
- `sku`: список/поле с поисковыми запросами;
- `sku_error`: если не `None`, строка игнорируется.

Во время подготовки:

- считается frequency каждой характеристики;
- формируются `train.parquet` и `valid.parquet`;
- дефолтный split: `80/20`;
- split перемешивается по `seed`;
- train DataLoader тоже перемешивается;
- dynamic batching сортирует только внутри bucket-а по примерной стоимости, чтобы не ловить OOM.

## Обучение

Текущая схема обучения:

- batch содержит `N` товаров;
- у каждого товара семплируются `5` позитивных SKU-запросов;
- для товара его 5 запросов считаются positives;
- `5 * (N - 1)` запросов от других товаров являются in-batch negatives;
- symmetric loss по умолчанию включен;
- используется gradient clipping;
- есть dynamic batching;
- есть FSDP для multi-GPU;
- есть live loss CSV/PNG/TensorBoard.

Пример запуска на 8 GPU:

```bash
torchrun --nproc_per_node=8 train_dual_encoder_on_parquet.py \
  --data-dir /home/jovyan/pasha/data/data1 \
  --tokenizer-path /home/jovyan/pasha/models/gemma-3-27b-it/tokenizer.json \
  --output-dir /home/jovyan/pasha/dual_encoder_runs/run_v4_multi_positive \
  --epochs 1 \
  --batch-size 16 \
  --positive-queries-per-product 5 \
  --dynamic-batching \
  --fsdp \
  --gradient-checkpointing
```

Важно: `--batch-size` сейчас означает product batch size per GPU. Фактический query batch будет `batch_size * positive_queries_per_product`.

## Checkpoints

Скрипт сохраняет:

- metadata checkpoint: `last_checkpoint.pt`, `best_checkpoint.pt`;
- full model weights: `.safetensors` или `.model.pt`;
- отдельно query encoder weights;
- отдельно product encoder weights;
- optimizer state для точного продолжения обучения;
- metrics/logs.

В GitHub checkpoint-и не загружаются. Они слишком большие и должны храниться отдельно.

Игнорируются:

```text
*.pt
*.ckpt
*.pth
*.bin
*.safetensors
```

## Продолжение обучения

Для дополнительных эпох на тех же данных:

```bash
torchrun --nproc_per_node=8 continue_training_from_checkpoint.py \
  --checkpoint-path /home/jovyan/pasha/dual_encoder_runs/run_v4_multi_positive/last_checkpoint.pt \
  --data-dir /home/jovyan/pasha/data/data1 \
  --tokenizer-path /home/jovyan/pasha/models/gemma-3-27b-it/tokenizer.json \
  --output-dir /home/jovyan/pasha/dual_encoder_runs/run_v4_multi_positive_continue \
  --additional-epochs 1 \
  --positive-queries-per-product 5 \
  --dynamic-batching \
  --fsdp \
  --gradient-checkpointing \
  --resume-optimizer
```

`--resume-optimizer` нужен, если требуется продолжить не просто с весов, а с тем же optimizer state.

## Векторизация каталога

Перед eval/search нужно векторизовать всю базу товаров.

Режим через `torchrun` на 8 GPU:

```bash
torchrun --nproc_per_node=8 vectorize_dual_encoder_catalog.py \
  --data-dir /home/jovyan/pasha/data/data1 \
  --checkpoint-path /home/jovyan/pasha/dual_encoder_runs/run_v4_multi_positive/last_checkpoint.pt \
  --tokenizer-path /home/jovyan/pasha/models/gemma-3-27b-it/tokenizer.json \
  --device cuda
```

Альтернативный режим без `torchrun`:

```bash
python vectorize_dual_encoder_catalog.py \
  --data-dir /home/jovyan/pasha/data/data1 \
  --checkpoint-path /home/jovyan/pasha/dual_encoder_runs/run_v4_multi_positive/last_checkpoint.pt \
  --tokenizer-path /home/jovyan/pasha/models/gemma-3-27b-it/tokenizer.json \
  --device cuda \
  --num-gpus 8
```

Векторизатор:

- грузит только product encoder;
- распределяет каталог по GPU;
- использует dynamic catalog batching;
- при OOM рекурсивно дробит batch;
- пишет shards и сливает их в `catalog_embeddings.float16.npy`;
- сохраняет manifest, чтобы eval/search проверяли совместимость.

## Eval с LLM judge

Eval делает:

1. Загружает query encoder из checkpoint-а.
2. Загружает precomputed product embeddings.
3. Кодирует позиции из Excel.
4. Считает cosine similarity query-vector к полному каталогу.
5. Берет retrieval pool.
6. Отправляет кандидаты в LLM judge через scheduler.
7. LLM возвращает binary relevance `1/0`.
8. Считаются метрики по top-k/top-p/score/probability thresholds.
9. Отдельно ведется статистика:
   - по всем объектам;
   - только по объектам, которые LLM классифицировала как мебель.

Пример:

```bash
python evaluate_dual_encoder_with_llm.py \
  --data-dir /home/jovyan/pasha/data/data1 \
  --checkpoint-path /home/jovyan/pasha/dual_encoder_runs/run_v4_multi_positive/last_checkpoint.pt \
  --tokenizer-path /home/jovyan/pasha/models/gemma-3-27b-it/tokenizer.json \
  --positions-xlsx /home/jovyan/pasha/big_model/dual_encoder_training/bentch.xlsx \
  --output-dir /home/jovyan/pasha/eval_runs/run_v4_multi_positive \
  --device cuda:0 \
  --retrieval-pool-size 100 \
  --top-k 1,3,5,10,20,50,100 \
  --top-p 0.5,0.8,0.9,0.95 \
  --score-thresholds 0.2,0.3,0.4 \
  --probability-thresholds 0.01,0.02 \
  --llm-max-in-flight 160 \
  --furniture-llm-max-in-flight 160 \
  --llm-max-attempts 5 \
  --llm-request-timeout-seconds 300 \
  --save-llm-raw-responses \
  --save-llm-prompt-bodies
```

Scheduler defaults:

```text
WS_URL = ws://10.233.3.148:9000/ws
MODEL_NAME = Llama-3_3-Nemotron-Super-49B-v1_5
GPU_COUNT = 16
REQUESTS_PER_GPU = 10
WINDOW_N = 160
TEMPERATURE = 0.7
MAX_TOKENS = 3000
```

Eval outputs:

- `evaluation_overview.json`;
- per-model metrics;
- `retrieval_rankings.parquet`;
- `query_to_matching_files.csv/.parquet`;
- `query_to_best_matching_file.csv/.parquet`;
- `query_to_best_matching_file.txt`;
- `llm_judgement_raw.jsonl`;
- `procurement_expert_llm_answers.jsonl`;
- `procurement_expert_llm_answers.log`;
- prompt artifacts, если включено сохранение prompt bodies.

## Онлайн-поиск

После векторизации каталога можно искать товары интерактивно:

```bash
python interactive_catalog_search.py \
  --data-dir /home/jovyan/pasha/data/data1 \
  --checkpoint-path /home/jovyan/pasha/dual_encoder_runs/run_v4_multi_positive/last_checkpoint.pt \
  --tokenizer-path /home/jovyan/pasha/models/gemma-3-27b-it/tokenizer.json \
  --catalog-vector-root /home/jovyan/pasha/dual_encoder_runs/run_v4_multi_positive/catalog_vectors \
  --device cuda:0 \
  --top-k 10 \
  --show-features
```

Одноразовый запрос:

```bash
python interactive_catalog_search.py \
  --data-dir /home/jovyan/pasha/data/data1 \
  --checkpoint-path /home/jovyan/pasha/dual_encoder_runs/run_v4_multi_positive/last_checkpoint.pt \
  --tokenizer-path /home/jovyan/pasha/models/gemma-3-27b-it/tokenizer.json \
  --catalog-vector-root /home/jovyan/pasha/dual_encoder_runs/run_v4_multi_positive/catalog_vectors \
  --device cuda:0 \
  --top-k 10 \
  --once "стол письменный 1200 мм"
```

## Как развивался пайплайн

Ключевые этапы:

1. Базовый query encoder с INT/STR mask на уровне tokenizer-а.
2. Разделение STR и INT потоков.
3. Numeric encoder через log-transform и multi-frequency encoding.
4. Double cross-attention между INT и STR.
5. Transformer stack + feed-forward blocks.
6. Переход к dual encoder: query encoder + product/card encoder.
7. Добавление title, description и features в product encoder.
8. Подсчет частот характеристик и выбор top features.
9. Dynamic batching для переменной длины карточек.
10. Multi-GPU/FSDP обучение.
11. Checkpoint saving с optimizer state и отдельными encoder weights.
12. LLM-based eval через scheduler.
13. Разделение eval на vectorization stage и LLM metrics stage.
14. Interactive search по precomputed catalog vectors.
15. V4: learned token type router, differentiable numeric merge, Longformer-style product attention, routing clusters, 14 features.
16. Multi-positive training: 5 positives per product + in-batch negatives.

## Что уже улучшено относительно ранней версии

- `600` больше не обязан жить как три независимых числа `6`, `0`, `0` в numeric path.
- Numeric merge стал дифференцируемым.
- Token type router стал обучаемым.
- Product encoder учитывает title, description и до 14 характеристик.
- Product pooling стал attention-based.
- В product encoder есть local/global sparse attention.
- Добавлен routing-transformer-like механизм кластеров.
- Dynamic batching снижает OOM на длинных карточках.
- Векторизация каталога распределяется по нескольким GPU.
- Eval теперь работает по precomputed product vectors.
- LLM judge имеет retries/timeouts и raw logs.
- Есть match-файлы для анализа `запрос -> подходящий товар`.
- Обучение перешло от "только позитивов" к multi-positive contrastive loss.

## Текущие ограничения

Самый важный риск сейчас не в архитектуре, а в обучающем сигнале.

In-batch negatives в основном случайные. Они полезны, но часто слишком легкие. Для закупочного matching-а нужны сложные негативы:

- похожий товар, но другой размер;
- та же категория, но другой материал;
- похожий title, но неподходящая комплектация;
- частично релевантный товар, который нельзя считать жестким `0`;
- товары, которые LLM считает подходящими, но которые попали в batch как negative.

Если этого не исправить, большая модель может хорошо отделять очевидный мусор, но плохо ранжировать похожие товары.

## Рекомендуемый следующий цикл улучшения качества

Наиболее полезный следующий шаг:

```text
train
  -> vectorize full catalog
  -> retrieve top-50/top-200 per query
  -> LLM judge candidates
  -> mine hard negatives and extra positives
  -> train with hard negatives + soft labels
  -> evaluate on shared judged candidate pool
```

Что стоит добавить дальше:

- hard negative mining из top выдачи текущей модели;
- soft labels вместо жесткого `0/1` для спорных товаров;
- teacher/reranker distillation от LLM или cross-encoder;
- category-aware batches;
- auxiliary loss на matching характеристик;
- отдельный penalty за numeric mismatch;
- BM25/lexical candidates в eval pool, чтобы LLM оценивала общий pool для всех моделей;
- calibration по score/probability thresholds.

## GitHub policy для этого репозитория

В репозиторий загружается код, документация, небольшие тестовые данные и конфиги.

Не загружаются:

- model checkpoints;
- optimizer checkpoints;
- `.pt`;
- `.safetensors`;
- большие training artifacts;
- catalog vector caches.

Для больших весов нужно использовать отдельное хранилище: S3/MinIO/HF Hub/internal artifact storage или локальную папку на training machine.
