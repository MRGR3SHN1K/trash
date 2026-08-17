# Product Card Encoder Diagram

Текущая реализация находится в `product_card_encoder.py`.

Идея:
- backbone взят в стиле `BGE-M3`: encoder-only, bidirectional attention, `CLS`-style dense embedding
- структура карточки товара добавлена сверху:
  - `title`
  - `description`
  - до `10` характеристик
  - у каждой характеристики есть `name` и `value`
- numeric spans внутри текста и значений склеиваются:
  - `6 | 0 | 0 -> 600`
  - `2 | . | 5 -> 2.5`
- numeric tokens идут в `INT` branch, остальные в `STR` branch

## 1. Card Layout

```text
ProductCard
  title: str
  description: str
  features[0..9]:
    - name: str
    - value: str | int | float
```

## 2. Sequence Construction

```mermaid
flowchart TD
    A[Product Card] --> B[Tokenizer per segment]
    B --> C[Numeric Span Merge]

    C --> D[<cls>]
    C --> E[Title Tokens]
    C --> F[Description Tokens]
    C --> G[Feature 1 Name]
    C --> H[Feature 1 Value]
    C --> I[...]
    C --> J[Feature 10 Name]
    C --> K[Feature 10 Value]

    D --> L[Flattened Product Sequence]
    E --> L
    F --> L
    G --> L
    H --> L
    I --> L
    J --> L
    K --> L
```

## 3. Encoder Architecture

```mermaid
flowchart TD
    A[Flattened Product Sequence] --> B[input_ids]
    A --> C[numeric_mask]
    A --> D[numeric_values]
    A --> E[role_ids<br/>CLS / title / description / feature_name / feature_value / sep]
    A --> F[slot_ids<br/>0..10]

    B --> G[Token Embedding]
    C --> H[Type Embedding<br/>STR=0 / INT=1]
    E --> I[Role Embedding]
    F --> J[Slot Embedding]
    K[Position Embedding] --> L[Base States]
    G --> L
    H --> L
    I --> L
    J --> L

    L --> M[String Input MLP]
    D --> N[Numeric Feature Encoder<br/>log1p + multi-frequency + RBF + linear]
    N --> O[Numeric Input Gate]
    H --> O
    I --> O
    J --> O
    K --> O

    M --> P
    O --> P

    subgraph P[24 x DualPathTransformerLayer by default]
        P1[String Self-Attention]
        P2[String FFN]
        P3[Numeric Self-Attention]
        P4[Numeric FFN]
        P5[Cross Attention NUM <- STR]
        P6[Cross Attention STR <- NUM]
        P7[Numeric Post-Cross FFN]
        P8[String Post-Cross FFN]
    end

    P --> Q[Masked Merge]
    Q --> R[Final LayerNorm]
    R --> S[CLS State]
    S --> T[Dense Projection Optional]
    T --> U[L2 Normalize]
    U --> V[Product Embedding]
```

## 4. Output

```text
Product Card
  -> normalized_embedding
  -> cosine similarity with query embedding
```

## 5. Code Map

- `ProductCard`, `ProductFeature`: структура входной карточки
- `prepare_product_card_batch_from_tokenizer*`: готовит batch из карточек
- `BgeM3ProductCardEncoder`: сам encoder
- `encode_cards*`: путь от карточек до embedding-векторов
