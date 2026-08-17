from pathlib import Path
import asyncio
import ast
import contextlib
import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional

import pandas as pd
import websockets

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

try:
    from IPython import get_ipython
    from IPython.display import HTML, display
except ImportError:
    get_ipython = None
    HTML = None
    display = None

# Укажи один или несколько входных файлов и папку, куда сохранить parquet-чанки с колонкой sku.
INPUT_DATASET_PATHS = [
    Path(r'D:\\path\\to\\input_dataset_1.csv'),
    Path(r'D:\\path\\to\\input_dataset_2.csv'),
]
OUTPUT_DATASET_DIR = Path(r'D:\\path\\to\\output_dataset')

TITLE_COLUMN = 'title'
SKU_COLUMN = 'sku'
FEATURES_COLUMN_CANDIDATES = ('features', 'characteristics', 'attrs', 'attributes')
BRAND_COLUMN_CANDIDATES = (
    'brand',
    'brand_name',
    'vendor',
    'vendor_name',
    'manufacturer',
    'producer',
    'trademark',
)

WS_URL = 'ws://10.233.3.148:9000/ws'
MODEL_NAME = 'Llama-3_3-Nemotron-Super-49B-v1_5'
GPU_COUNT = 8
REQUESTS_PER_GPU = 10
WINDOW_N = GPU_COUNT * REQUESTS_PER_GPU
TEMPERATURE = 0.7
MAX_TOKENS = 3000
ROW_LIMIT = None
DEBUG = False
DEBUG_SAMPLE_ROWS = 3
TARGET_QUERY_COUNT = 5
MIN_FEATURES_PER_QUERY = 6
MAX_FEATURES_PER_QUERY = 7
FEATURE_POOL_LIMIT = 12
COMMON_FEATURES_COUNT = 2
PROGRESS_LOG_EVERY = 10_000
OUTPUT_CHUNK_SIZE = 100_000

PROMPT = '''
Ты генерируешь поисковые запросы для одного товара по title и features.

Верни только JSON следующего формата:
{
  "queries": [
    {"query": "..."},
    {"query": "..."},
    {"query": "..."},
    {"query": "..."},
    {"query": "..."}
  ]
}

Правила:
- Верни ровно 5 запросов.
- Все 5 запросов должны быть заметно разными по набору характеристик и по акцентам. Нельзя возвращать дубликаты, почти одинаковые перефразирования и запросы с одним и тем же набором характеристик.
- Каждый запрос должен относиться к одному и тому же товару.
- Каждый запрос должен начинаться с наименования товара, извлеченного из поля title.
- После наименования обязательно добавь бренд, если он известен.
- После бренда добавь 6 или 7 характеристик товара, если они доступны. Если характеристик меньше, используй максимально доступное число.
- Характеристики можно объединять в компактные фрагменты. Хорошо: "размер 500x600x200", "материал корпуса металл", "цвет белый". Плохо: бессмысленно дробить один и тот же признак на много отдельных коротких кусочков.
- Разные запросы должны использовать разные комбинации характеристик из feature_usage_plan. Меняй порядок и акцент: в одном запросе важнее размеры и материал, в другом цвет и тип, в третьем комплектация и назначение, и так далее.
- Не копируй title целиком, а выдели из него именно наименование товара.
- Не выдумывай бренд и характеристики.
- Не используй цену, скидки, доставку, артикулы, ID, SKU, рейтинг, продавца и служебные поля.
- Используй характеристики из feature_usage_plan.
- Более частотные характеристики могут встречаться чаще, но итоговые 5 запросов всё равно должны оставаться различными.
- Формулируй запросы естественно и кратко, в стиле реального поиска.
- Все слова пиши раздельно через обычный пробел. Нельзя склеивать слова без пробелов.
'''.strip()

BRAND_FEATURE_NAMES = {
    'бренд',
    'марка',
    'торговая марка',
    'производитель',
    'manufacturer',
    'brand',
    'vendor',
    'vendor name',
    'brand name',
    'maker',
}

IGNORED_FEATURE_NAMES = {
    'id',
    'sku',
    'gtin',
    'ean',
    'upc',
    'barcode',
    'bar code',
    'артикул',
    'код товара',
    'внутренний код',
    'ссылка',
    'url',
    'ссылка на товар',
    'продавец',
    'seller',
    'shop',
    'магазин',
    'цена',
    'price',
    'скидка',
    'discount',
    'доставка',
    'delivery',
    'наличие',
    'availability',
    'остаток',
    'rating',
    'рейтинг',
    'количество отзывов',
    'число отзывов',
    'review count',
}

IGNORED_VALUE_PATTERNS = (
    re.compile(r'https?://', flags=re.IGNORECASE),
    re.compile(r'^\\d{7,}$'),
    re.compile(r'^[A-Z0-9_-]{8,}$'),
)

JSON_FENCE_RE = re.compile(r'```(?:json)?\\s*(\\{.*?\\}|\\[.*?\\])\\s*```', flags=re.IGNORECASE | re.DOTALL)
INVALID_CATEGORY_PATTERN = re.compile(r'^[A-Za-zА-Яа-яЁё0-9\\s.,!?()/-]*$')


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == '.csv':
        return pd.read_csv(path)
    if suffix == '.parquet':
        return pd.read_parquet(path)
    if suffix in {'.xlsx', '.xls'}:
        return pd.read_excel(path)
    raise ValueError(f'Unsupported input format: {path.suffix}')


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == '.csv':
        df.to_csv(path, index=False)
        return
    if suffix == '.parquet':
        df.to_parquet(path, index=False)
        return
    if suffix in {'.xlsx', '.xls'}:
        df.to_excel(path, index=False)
        return
    raise ValueError(f'Unsupported output format: {path.suffix}')


def get_input_dataset_paths() -> list[Path]:
    paths = [Path(path) for path in INPUT_DATASET_PATHS]
    if not paths:
        raise ValueError('Добавь хотя бы один путь в INPUT_DATASET_PATHS.')
    if any(str(path).startswith('D:\\path\\to\\') for path in paths):
        raise ValueError('Заполни INPUT_DATASET_PATHS и OUTPUT_DATASET_DIR в начале ячейки.')
    return paths


def get_output_dataset_dir() -> Path:
    output_dir = Path(OUTPUT_DATASET_DIR)
    if str(output_dir).startswith('D:\\path\\to\\'):
        raise ValueError('Заполни INPUT_DATASET_PATHS и OUTPUT_DATASET_DIR в начале ячейки.')
    return output_dir


def load_input_tables(paths: list[Path]) -> pd.DataFrame:
    frames = [read_table(path) for path in paths]
    if len(frames) == 1:
        return frames[0].copy()
    return pd.concat(frames, ignore_index=True, sort=False)


class CheckpointParquetWriter:
    def __init__(
        self,
        base_df: pd.DataFrame,
        output_dir: Path,
        chunk_size: int = OUTPUT_CHUNK_SIZE,
    ):
        self.base_df = base_df
        self.output_dir = output_dir
        self.chunk_size = max(int(chunk_size), 1)
        self.next_row_to_save = 0
        self.next_chunk_idx = 1
        self.chunk_paths: list[Path] = []

        self.output_dir.mkdir(parents=True, exist_ok=True)
        for existing_chunk in self.output_dir.glob('data*.parquet'):
            existing_chunk.unlink()

    def build_chunk_path(self) -> Path:
        return self.output_dir / f'data{self.next_chunk_idx:04d}.parquet'

    def flush_ready_chunks(
        self,
        completed_rows: list[bool],
        sku_values: list[Optional[str]],
        sku_errors: list[Optional[str]],
        force: bool = False,
    ) -> list[Path]:
        total_rows = len(self.base_df)
        contiguous_end = self.next_row_to_save
        while contiguous_end < total_rows and completed_rows[contiguous_end]:
            contiguous_end += 1

        if force:
            target_end = contiguous_end
        else:
            ready_rows = contiguous_end - self.next_row_to_save
            target_end = self.next_row_to_save + (ready_rows // self.chunk_size) * self.chunk_size

        new_chunk_paths: list[Path] = []
        while self.next_row_to_save < target_end:
            chunk_end = min(self.next_row_to_save + self.chunk_size, target_end)
            chunk_df = self.base_df.iloc[self.next_row_to_save:chunk_end].copy()
            chunk_df[SKU_COLUMN] = sku_values[self.next_row_to_save:chunk_end]
            chunk_df['sku_error'] = sku_errors[self.next_row_to_save:chunk_end]

            chunk_path = self.build_chunk_path()
            chunk_df.to_parquet(chunk_path, index=False)
            self.chunk_paths.append(chunk_path)
            new_chunk_paths.append(chunk_path)

            self.next_row_to_save = chunk_end
            self.next_chunk_idx += 1

        if force and total_rows == 0 and not self.chunk_paths:
            empty_path = self.build_chunk_path()
            empty_df = self.base_df.copy()
            empty_df[SKU_COLUMN] = []
            empty_df['sku_error'] = []
            empty_df.to_parquet(empty_path, index=False)
            self.chunk_paths.append(empty_path)
            new_chunk_paths.append(empty_path)
            self.next_chunk_idx += 1

        return new_chunk_paths


def write_parquet_chunks(
    df: pd.DataFrame,
    output_dir: Path,
    chunk_size: int = OUTPUT_CHUNK_SIZE,
) -> list[Path]:
    writer = CheckpointParquetWriter(df, output_dir=output_dir, chunk_size=chunk_size)
    completed_rows = [True] * len(df)
    sku_values = (
        df[SKU_COLUMN].tolist()
        if SKU_COLUMN in df.columns
        else [None] * len(df)
    )
    sku_errors = (
        df['sku_error'].tolist()
        if 'sku_error' in df.columns
        else [None] * len(df)
    )
    return writer.flush_ready_chunks(
        completed_rows=completed_rows,
        sku_values=sku_values,
        sku_errors=sku_errors,
        force=True,
    )


def detect_features_column(columns: list[str]) -> str:
    for column in FEATURES_COLUMN_CANDIDATES:
        if column in columns:
            return column
    raise KeyError(f'Features column not found. Available columns: {columns}')


def should_debug_row(row_idx: int) -> bool:
    return DEBUG and row_idx < DEBUG_SAMPLE_ROWS


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return 'unknown'
    seconds = max(int(seconds), 0)
    return str(timedelta(seconds=seconds))


def get_progress_snapshot(current: int, total: int, started_at: float) -> dict[str, Any]:
    elapsed = max(time.perf_counter() - started_at, 1e-9)
    rate = current / elapsed if current > 0 else 0.0
    remaining = max(total - current, 0)
    eta_seconds = (remaining / rate) if rate > 0 else None
    percent = int((current / max(total, 1)) * 100) if total > 0 else 100
    return {
        'current': current,
        'total': total,
        'elapsed_seconds': elapsed,
        'rate': rate,
        'eta_seconds': eta_seconds,
        'percent': percent,
    }


def get_progress_log_path() -> Path:
    return get_output_dataset_dir() / 'llm_markup.log'


class ProgressLogger:
    def __init__(self, path: Path, desc: str, total: int, log_every: int):
        self.path = path
        self.desc = desc
        self.total = total
        self.log_every = max(int(log_every), 1)
        self.next_log_at = self.log_every
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            f'[{datetime.now().isoformat(timespec="seconds")}] '
            f'{self.desc} started: total={self.total}\n',
            encoding='utf-8',
        )

    def maybe_log(self, snapshot: dict[str, Any], force: bool = False) -> None:
        current = int(snapshot['current'])
        if not force and current < self.next_log_at:
            return
        while self.next_log_at <= current:
            self.next_log_at += self.log_every
        line = (
            f'[{datetime.now().isoformat(timespec="seconds")}] '
            f'{self.desc}: {current}/{self.total} ({snapshot["percent"]}%), '
            f'rate={snapshot["rate"]:.2f} rec/s, '
            f'eta={format_duration(snapshot["eta_seconds"])}, '
            f'elapsed={format_duration(snapshot["elapsed_seconds"])}\n'
        )
        with self.path.open('a', encoding='utf-8') as log_file:
            log_file.write(line)


def normalize_spaces(value: str) -> str:
    return re.sub(r'\\s+', ' ', value).strip()


def normalize_feature_name(value: str) -> str:
    value = value.lower().replace('_', ' ').replace('-', ' ')
    value = re.sub(r'\\s+', ' ', value)
    return value.strip(' :;,.')


def clean_text_value(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, bool):
        return 'да' if value else 'нет'
    if isinstance(value, float) and pd.isna(value):
        return ''
    if isinstance(value, (list, tuple, set)):
        parts = [clean_text_value(item) for item in value]
        parts = [part for part in parts if part]
        return ', '.join(dict.fromkeys(parts))
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            item_text = clean_text_value(item)
            if item_text:
                parts.append(f'{key}: {item_text}')
        return '; '.join(parts)
    return normalize_spaces(str(value))


def contains_invalid_chars(text: Any) -> bool:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return True
    return not bool(INVALID_CATEGORY_PATTERN.fullmatch(str(text)))


def clean_df(df: pd.DataFrame, features_column: str) -> pd.DataFrame:
    mask = df[TITLE_COLUMN].isna() | df[TITLE_COLUMN].astype(str).str.strip().eq('')
    if features_column in df.columns:
        mask |= df[features_column].isna() | df[features_column].astype(str).str.strip().eq('')
    if 'category_name' in df.columns:
        mask |= df['category_name'].apply(contains_invalid_chars)
    return df.loc[~mask].reset_index(drop=True).copy()


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


@dataclass(frozen=True)
class FeatureItem:
    name: str
    value: str
    norm_name: str


class SimpleProgressBar:
    def __init__(
        self,
        total: int,
        desc: str,
        log_path: Optional[Path] = None,
        log_every: int = PROGRESS_LOG_EVERY,
    ):
        self.total = max(int(total), 0)
        self.desc = desc
        self.current = 0
        self.last_percent = -1
        self.started_at = time.perf_counter()
        self.logger = (
            ProgressLogger(log_path, desc, self.total, log_every)
            if log_path is not None
            else None
        )
        self._render(force=True)

    def update(self, step: int = 1) -> None:
        self.current += int(step)
        self.current = min(self.current, self.total)
        snapshot = get_progress_snapshot(self.current, self.total, self.started_at)
        if self.logger is not None:
            self.logger.maybe_log(snapshot)
        self._render()

    def close(self) -> None:
        if self.logger is not None:
            snapshot = get_progress_snapshot(self.current, self.total, self.started_at)
            self.logger.maybe_log(snapshot, force=True)
        self._render(force=True)

    def _render(self, force: bool = False) -> None:
        if self.total <= 0:
            if force:
                print(f'{self.desc}: 0/0 (100%)')
            return
        snapshot = get_progress_snapshot(self.current, self.total, self.started_at)
        percent = snapshot['percent']
        if not force and percent < self.last_percent + 5 and self.current < self.total:
            return
        self.last_percent = percent
        print(
            f'{self.desc}: {self.current}/{self.total} ({percent}%), '
            f'{snapshot["rate"]:.2f} rec/s, '
            f'ETA {format_duration(snapshot["eta_seconds"])}'
        )


class NotebookHtmlProgressBar:
    def __init__(
        self,
        total: int,
        desc: str,
        log_path: Optional[Path] = None,
        log_every: int = PROGRESS_LOG_EVERY,
    ):
        self.total = max(int(total), 0)
        self.desc = desc
        self.current = 0
        self.started_at = time.perf_counter()
        self.logger = (
            ProgressLogger(log_path, desc, self.total, log_every)
            if log_path is not None
            else None
        )
        self.handle = display(HTML(self._render_html()), display_id=True)

    def update(self, step: int = 1) -> None:
        self.current += int(step)
        self.current = min(self.current, self.total)
        snapshot = get_progress_snapshot(self.current, self.total, self.started_at)
        if self.logger is not None:
            self.logger.maybe_log(snapshot)
        self.handle.update(HTML(self._render_html()))

    def close(self) -> None:
        if self.logger is not None:
            snapshot = get_progress_snapshot(self.current, self.total, self.started_at)
            self.logger.maybe_log(snapshot, force=True)
        self.handle.update(HTML(self._render_html(done=True)))

    def _render_html(self, done: bool = False) -> str:
        snapshot = get_progress_snapshot(self.current, self.total, self.started_at)
        percent = snapshot['percent']
        bar_color = '#2f7d32' if done or self.current >= self.total else '#1f6feb'
        safe_desc = self.desc.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        return f"""
        <div style="font-family: sans-serif; margin: 8px 0; max-width: 720px;">
          <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
            <strong>{safe_desc}</strong>
            <span>{self.current}/{self.total} ({percent}%)</span>
          </div>
          <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 12px; color: #4b5563;">
            <span>{snapshot["rate"]:.2f} rec/s</span>
            <span>ETA: {format_duration(snapshot["eta_seconds"])}</span>
            <span>Elapsed: {format_duration(snapshot["elapsed_seconds"])}</span>
          </div>
          <div style="width: 100%; height: 14px; background: #e5e7eb; border-radius: 999px; overflow: hidden;">
            <div style="width: {percent}%; height: 100%; background: {bar_color}; transition: width 0.2s ease;"></div>
          </div>
        </div>
        """


class TqdmProgressBar:
    def __init__(
        self,
        total: int,
        desc: str,
        log_path: Optional[Path] = None,
        log_every: int = PROGRESS_LOG_EVERY,
    ):
        self.total = max(int(total), 0)
        self.desc = desc
        self.current = 0
        self.started_at = time.perf_counter()
        self.logger = (
            ProgressLogger(log_path, desc, self.total, log_every)
            if log_path is not None
            else None
        )
        self.bar = tqdm(total=self.total, desc=desc, leave=True)
        self._refresh(force=True)

    def update(self, step: int = 1) -> None:
        step = int(step)
        self.current = min(self.current + step, self.total)
        self.bar.update(step)
        snapshot = get_progress_snapshot(self.current, self.total, self.started_at)
        if self.logger is not None:
            self.logger.maybe_log(snapshot)
        self._refresh()

    def close(self) -> None:
        if self.logger is not None:
            snapshot = get_progress_snapshot(self.current, self.total, self.started_at)
            self.logger.maybe_log(snapshot, force=True)
        self._refresh(force=True)
        self.bar.close()

    def _refresh(self, force: bool = False) -> None:
        snapshot = get_progress_snapshot(self.current, self.total, self.started_at)
        self.bar.set_postfix_str(
            f'{snapshot["rate"]:.2f} rec/s | ETA {format_duration(snapshot["eta_seconds"])}',
            refresh=force,
        )


def is_notebook_environment() -> bool:
    if get_ipython is None:
        return False
    try:
        shell = get_ipython()
    except Exception:
        return False
    if shell is None:
        return False
    return shell.__class__.__name__ == 'ZMQInteractiveShell'


def create_progress_bar(
    total: int,
    desc: str,
    log_path: Optional[Path] = None,
    log_every: int = PROGRESS_LOG_EVERY,
):
    if is_notebook_environment() and HTML is not None and display is not None:
        return NotebookHtmlProgressBar(
            total=total,
            desc=desc,
            log_path=log_path,
            log_every=log_every,
        )
    if tqdm is not None:
        return TqdmProgressBar(
            total=total,
            desc=desc,
            log_path=log_path,
            log_every=log_every,
        )
    return SimpleProgressBar(
        total=total,
        desc=desc,
        log_path=log_path,
        log_every=log_every,
    )


def looks_like_feature_record(value: dict[str, Any]) -> bool:
    normalized_keys = {normalize_feature_name(str(key)) for key in value.keys()}
    name_keys = {'name', 'feature', 'feature name', 'attribute', 'characteristic', 'title', 'key'}
    value_keys = {'value', 'values', 'text', 'description', 'value name', 'value_name', 'content'}
    return bool(normalized_keys & name_keys) and bool(normalized_keys & value_keys)


def extract_feature_name_from_record(value: dict[str, Any]) -> str:
    for key in ('name', 'feature', 'attribute', 'characteristic', 'title', 'key'):
        if key in value:
            return clean_text_value(value[key])
    return ''


def extract_feature_value_from_record(value: dict[str, Any]) -> str:
    for key in ('value', 'values', 'text', 'description', 'value_name', 'content'):
        if key in value:
            return clean_text_value(value[key])
    return ''


def flatten_features(raw_features: Any) -> list[FeatureItem]:
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

    def walk(value: Any, parent_name: Optional[str] = None) -> None:
        if value is None:
            return
        if isinstance(value, dict):
            if looks_like_feature_record(value):
                add_item(extract_feature_name_from_record(value), extract_feature_value_from_record(value))
                return
            for key, item in value.items():
                if isinstance(item, dict):
                    walk(item, str(key))
                elif isinstance(item, list):
                    walk(item, str(key))
                else:
                    add_item(key, item)
            return
        if isinstance(value, list):
            for item in value:
                if isinstance(item, (dict, list)):
                    walk(item, parent_name)
                elif parent_name:
                    add_item(parent_name, item)

    walk(parsed)

    deduped: list[FeatureItem] = []
    seen: set[tuple[str, str]] = set()
    for item in collected:
        dedupe_key = (item.norm_name, normalize_spaces(item.value.lower()))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        deduped.append(item)
    return deduped


def is_ignored_feature_name(norm_name: str) -> bool:
    if norm_name in BRAND_FEATURE_NAMES or norm_name in IGNORED_FEATURE_NAMES:
        return True
    noise_tokens = ('sku', 'id', 'артикул', 'код', 'price', 'цена', 'seller', 'продавец')
    return any(token in norm_name for token in noise_tokens)


def is_ignored_feature_value(value: str) -> bool:
    if not value:
        return True
    lowered = value.lower()
    if lowered in {'nan', 'none', 'null', '-', '--'}:
        return True
    if len(value) > 120:
        return True
    return any(pattern.search(value) for pattern in IGNORED_VALUE_PATTERNS)


def extract_brand(row: dict[str, Any], features: list[FeatureItem]) -> str:
    for column in BRAND_COLUMN_CANDIDATES:
        if column not in row:
            continue
        value = clean_text_value(row.get(column))
        if value and not is_ignored_feature_value(value):
            return value

    for feature in features:
        if feature.norm_name in BRAND_FEATURE_NAMES and not is_ignored_feature_value(feature.value):
            return feature.value

    return ''


def build_feature_frequency(records: list[dict[str, Any]], features_column: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in records:
        feature_names = {
            feature.norm_name
            for feature in flatten_features(row.get(features_column))
            if not is_ignored_feature_name(feature.norm_name) and not is_ignored_feature_value(feature.value)
        }
        counter.update(feature_names)
    return counter


def rank_row_features(row: dict[str, Any], features_column: str, frequency: Counter[str]) -> tuple[str, list[dict[str, Any]]]:
    features = flatten_features(row.get(features_column))
    brand = extract_brand(row, features)

    ranked: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for feature in features:
        if feature.norm_name in BRAND_FEATURE_NAMES:
            continue
        if is_ignored_feature_name(feature.norm_name) or is_ignored_feature_value(feature.value):
            continue
        if brand and normalize_spaces(feature.value.lower()) == normalize_spaces(brand.lower()):
            continue
        if feature.norm_name in seen_names:
            continue
        seen_names.add(feature.norm_name)
        ranked.append({
            'name': feature.name,
            'value': feature.value,
            'norm_name': feature.norm_name,
            'global_frequency': int(frequency.get(feature.norm_name, 0)),
        })

    ranked.sort(key=lambda item: (-item['global_frequency'], item['norm_name'], item['value'].lower()))
    return brand, ranked


def rotate_features(features: list[dict[str, Any]], shift: int) -> list[dict[str, Any]]:
    if not features:
        return []
    shift %= len(features)
    return features[shift:] + features[:shift]


def target_feature_count(feature_count: int, query_idx: int) -> int:
    if feature_count <= 0:
        return 0
    if feature_count <= MIN_FEATURES_PER_QUERY:
        return feature_count
    if query_idx % 2 == 0 and feature_count >= MAX_FEATURES_PER_QUERY:
        return MAX_FEATURES_PER_QUERY
    return min(feature_count, MIN_FEATURES_PER_QUERY)


def build_feature_usage_plan(ranked_features: list[dict[str, Any]], query_count: int = 5) -> list[list[dict[str, Any]]]:
    if not ranked_features:
        return [[] for _ in range(query_count)]

    selected = ranked_features[:FEATURE_POOL_LIMIT]
    if len(selected) <= MAX_FEATURES_PER_QUERY:
        return [
            rotate_features(selected, query_idx)[:len(selected)]
            for query_idx in range(query_count)
        ]

    common_count = min(COMMON_FEATURES_COUNT, max(1, len(selected) - MIN_FEATURES_PER_QUERY))
    common_features = selected[:common_count]
    variable_features = selected[common_count:]
    rotation_step = max(1, len(variable_features) // max(query_count, 1))
    plans: list[list[dict[str, Any]]] = []

    for query_idx in range(query_count):
        target_count = target_feature_count(len(selected), query_idx)
        rotated = rotate_features(variable_features, query_idx * rotation_step)
        priority_features = rotated[:min(3, len(rotated))]

        plan: list[dict[str, Any]] = []
        for feature in priority_features + common_features + rotated:
            if feature in plan:
                continue
            plan.append(feature)
            if len(plan) >= target_count:
                break

        if len(plan) < target_count:
            for feature in selected:
                if feature in plan:
                    continue
                plan.append(feature)
                if len(plan) >= target_count:
                    break

        plans.append(plan)

    return plans


def build_query_goal(plan: list[dict[str, Any]]) -> str:
    if not plan:
        return 'Сделай запрос отличным от остальных за счёт другого порядка слов и доступных признаков товара.'
    focus_names = [clean_text_value(feature.get('name')) for feature in plan[:3]]
    focus_names = [name for name in focus_names if name]
    if not focus_names:
        return 'Сделай запрос отличным от остальных и меняй акцент между разными признаками товара.'
    return (
        'Сделай этот запрос заметно непохожим на остальные. '
        f'Главный акцент: {", ".join(focus_names)}.'
    )


def build_model_input(row: dict[str, Any], features_column: str, brand: str, ranked_features: list[dict[str, Any]], feature_usage_plan: list[list[dict[str, Any]]]) -> str:
    top_ranked_features = ranked_features[:FEATURE_POOL_LIMIT]
    payload = {
        'title': clean_text_value(row.get(TITLE_COLUMN)),
        'brand': brand,
        'features': try_parse_json_like(row.get(features_column)),
        'ranked_features': [
            {
                'name': feature['name'],
                'value': feature['value'],
                'global_frequency': feature['global_frequency'],
            }
            for feature in top_ranked_features
        ],
        'feature_usage_plan': [
            {
                'query_index': index + 1,
                'query_goal': build_query_goal(plan),
                'target_feature_count': f'{MIN_FEATURES_PER_QUERY}-{MAX_FEATURES_PER_QUERY}',
                'priority_features': [
                    {
                        'name': feature['name'],
                        'value': feature['value'],
                        'global_frequency': feature['global_frequency'],
                    }
                    for feature in plan[:3]
                ],
                'supporting_features': [
                    {
                        'name': feature['name'],
                        'value': feature['value'],
                        'global_frequency': feature['global_frequency'],
                    }
                    for feature in plan[3:]
                ],
            }
            for index, plan in enumerate(feature_usage_plan)
        ],
        'notes': {
            'must_start_with_product_name_from_title': True,
            'must_include_brand_if_known': bool(brand),
            'target_query_count': TARGET_QUERY_COUNT,
            'target_features_per_query': f'{MIN_FEATURES_PER_QUERY}-{MAX_FEATURES_PER_QUERY}',
            'queries_must_be_distinct': True,
            'use_different_feature_combinations_between_queries': True,
            'combine_related_features_into_compact_phrases': True,
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def feature_to_query_part(feature: dict[str, Any]) -> str:
    name = clean_text_value(feature.get('name'))
    value = clean_text_value(feature.get('value'))
    if not name:
        return value
    if not value:
        return name
    if normalize_feature_name(name) in value.lower():
        return value
    return f'{name} {value}'


def detect_dimension_slot(norm_name: str) -> Optional[str]:
    if any(token in norm_name for token in ('длина', 'length')):
        return 'length'
    if any(token in norm_name for token in ('ширина', 'width')):
        return 'width'
    if any(token in norm_name for token in ('высота', 'height')):
        return 'height'
    if any(token in norm_name for token in ('глубина', 'depth')):
        return 'depth'
    return None


def build_feature_fragments(features: list[dict[str, Any]]) -> list[str]:
    dimensions: dict[str, str] = {}
    fragments: list[str] = []

    for feature in features:
        norm_name = clean_text_value(feature.get('norm_name'))
        value = clean_text_value(feature.get('value'))
        slot = detect_dimension_slot(norm_name)
        if slot and value:
            dimensions[slot] = value
            continue
        fragments.append(feature_to_query_part(feature))

    ordered_dimension_values = [
        dimensions[slot]
        for slot in ('length', 'width', 'height', 'depth')
        if dimensions.get(slot)
    ]
    if len(ordered_dimension_values) >= 2:
        fragments.insert(0, f"размер {'x'.join(ordered_dimension_values)}")
    elif len(ordered_dimension_values) == 1:
        fragments.insert(0, f"размер {ordered_dimension_values[0]}")

    return dedupe_keep_order(fragments)


def dedupe_keep_order(parts: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for part in parts:
        normalized = normalize_spaces(part)
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


def normalize_query_text(query: str) -> str:
    query = sanitize_text(clean_text_value(query))
    query = re.sub(r'(?<=[a-zа-яё])(?=[A-ZА-ЯЁ])', ' ', query)
    query = re.sub(r'(?<=[A-Za-zА-Яа-яЁё])(?=\d)', ' ', query)
    query = re.sub(r'(?<=\d)(?=[A-Za-zА-Яа-яЁё])', ' ', query)
    query = normalize_spaces(query)
    return query


def serialize_sku_queries(queries: list[str]) -> str:
    normalized_queries = [normalize_query_text(query) for query in queries if query]
    return json.dumps(normalized_queries, ensure_ascii=False)


def looks_like_missing_spaces(query: str) -> bool:
    query = normalize_query_text(query)
    if query.count(' ') < 3:
        return True
    if re.search(r'[A-Za-zА-Яа-яЁё]{35,}', query):
        return True
    return False


def build_fallback_queries(row: dict[str, Any], features_column: str, frequency: Counter[str]) -> list[str]:
    brand, ranked_features = rank_row_features(row, features_column, frequency)
    feature_usage_plan = build_feature_usage_plan(ranked_features)
    title = clean_text_value(row.get(TITLE_COLUMN))
    fallback_queries: list[str] = []

    for plan in feature_usage_plan:
        parts = [title]
        if brand:
            parts.append(brand)
        parts.extend(build_feature_fragments(plan))

        target_feature_total = min(target_feature_count(len(ranked_features), len(fallback_queries)), len(ranked_features))
        if len(parts) - (2 if brand else 1) < target_feature_total:
            for feature in ranked_features:
                extra_part = feature_to_query_part(feature)
                parts.append(extra_part)
                parts = dedupe_keep_order(parts)
                if len(parts) - (2 if brand else 1) >= target_feature_total:
                    break

        parts = dedupe_keep_order(parts)
        fallback_queries.append(normalize_query_text(' '.join(parts)))

    while len(fallback_queries) < TARGET_QUERY_COUNT:
        fallback_queries.append(fallback_queries[-1] if fallback_queries else title)

    return fallback_queries[:TARGET_QUERY_COUNT]


def extract_final_text(res: dict[str, Any]) -> str:
    text = res.get('choices', [{}])[0].get('text', '')
    if not isinstance(text, str):
        return ''
    if '</think>' in text:
        text = text.split('</think>', 1)[1]
    text = text.strip()
    text = re.sub(r'^\\s*```(?:\\w+)?\\s*', '', text)
    text = re.sub(r'\\s*```\\s*$', '', text)
    return text.strip()


def sanitize_text(value: str) -> str:
    value = value.replace('\ufeff', '')
    value = value.replace('\u200b', '')
    value = value.replace('\u00a0', ' ')
    return value.strip()


def extract_json_only(text: str) -> tuple[str, Any]:
    if not isinstance(text, str):
        raise ValueError('text is not str')

    text = sanitize_text(text)
    fenced_match = JSON_FENCE_RE.search(text)
    if fenced_match:
        candidate = sanitize_text(fenced_match.group(1))
        obj = json.loads(candidate)
        return json.dumps(obj, ensure_ascii=False), obj

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in '{[':
            continue
        try:
            obj, _ = decoder.raw_decode(text[index:])
            return json.dumps(obj, ensure_ascii=False), obj
        except json.JSONDecodeError:
            continue

    raise ValueError('No valid JSON found in model output')


def normalize_queries_output(obj: Any) -> list[str]:
    if isinstance(obj, dict):
        if 'queries' in obj:
            raw_queries = obj['queries']
        else:
            raw_queries = list(obj.values())
    elif isinstance(obj, list):
        raw_queries = obj
    else:
        raise ValueError('Model output must be a JSON object or JSON array')

    queries: list[str] = []
    for item in raw_queries:
        if isinstance(item, str):
            query = normalize_query_text(item)
        elif isinstance(item, dict):
            query = normalize_query_text(item.get('query') or item.get('text') or item.get('value'))
        else:
            query = normalize_query_text(item)
        if query:
            queries.append(query)

    if len(queries) != TARGET_QUERY_COUNT:
        raise ValueError(f'Expected {TARGET_QUERY_COUNT} queries, got {len(queries)}')

    return queries


def parse_queries_from_response(raw_res: dict[str, Any]) -> tuple[Optional[list[str]], Optional[str]]:
    try:
        _, obj = extract_json_only(extract_final_text(raw_res))
        queries = normalize_queries_output(obj)
        return queries, None
    except Exception as exc:
        return None, str(exc)


@dataclass
class TaskHandle:
    session_id: str
    done_future: Any

    async def result(self) -> dict[str, Any]:
        return await self.done_future


class SchedulerClient:
    def __init__(self, ws_url: str, model: str, path: str = '/v1/completions', method: str = 'POST'):
        self.ws_url = ws_url
        self.model = model
        self.path = path
        self.method = method
        self.ws = None
        self.recv_task = None
        self.submit_lock = None
        self.send_lock = None
        self.awaiting_ack = None
        self.started_events: dict[str, Any] = {}
        self.done_futures: dict[str, Any] = {}
        self.early_msgs: dict[str, list[dict[str, Any]]] = {}

    async def connect(self) -> None:
        self.submit_lock = asyncio.Lock()
        self.send_lock = asyncio.Lock()
        self.ws = await websockets.connect(self.ws_url, ping_interval=20, ping_timeout=20)
        self.recv_task = asyncio.create_task(self.receiver_loop())

    async def close(self) -> None:
        if self.recv_task:
            self.recv_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.recv_task
        if self.ws:
            await self.ws.close()

    async def ws_send(self, msg: dict[str, Any]) -> None:
        if not self.ws:
            raise RuntimeError('WebSocket is not connected')
        async with self.send_lock:
            await self.ws.send(json.dumps(msg, ensure_ascii=False))

    def handle_session_msg(self, session_id: str, msg: dict[str, Any]) -> None:
        msg_type = msg.get('type')

        if msg_type == 'accepted':
            self.started_events.setdefault(session_id, asyncio.Event()).set()
            return

        if msg_type == 'rerouted':
            return

        if msg_type == 'done':
            future = self.done_futures.get(session_id)
            if future and not future.done():
                future.set_result(msg.get('result', {}))
            return

        if msg_type == 'error':
            future = self.done_futures.get(session_id)
            error_text = msg.get('error') or 'job_failed'
            if future and not future.done():
                future.set_exception(RuntimeError(str(error_text)))
            return

        if msg_type == 'busy':
            return

    async def receiver_loop(self) -> None:
        if not self.ws:
            raise RuntimeError('WebSocket is not connected')

        async for raw_msg in self.ws:
            msg = json.loads(raw_msg)
            msg_type = msg.get('type')

            if msg_type in {'received', 'busy'}:
                if self.awaiting_ack and not self.awaiting_ack.done():
                    self.awaiting_ack.set_result(msg)
                continue

            session_id = msg.get('session_id')
            if not session_id:
                continue

            if msg_type in {'accepted', 'rerouted', 'done', 'error'}:
                self.handle_session_msg(session_id, msg)
            else:
                self.early_msgs.setdefault(session_id, []).append(msg)

    async def submit(self, text: str, prompt: str, temperature: float, max_tokens: int) -> TaskHandle:
        if not self.ws:
            raise RuntimeError('WebSocket is not connected')

        async with self.submit_lock:
            payload = {
                'model': self.model,
                'prompt': f'{prompt}\\n\\n{text}\\n',
                'temperature': float(temperature),
                'max_tokens': int(max_tokens),
                'stream': False,
            }
            request = {
                'type': 'request',
                'path': self.path,
                'method': self.method,
                'payload': payload,
            }

            self.awaiting_ack = asyncio.get_running_loop().create_future()
            await self.ws_send(request)
            ack = await self.awaiting_ack
            session_id = ack['session_id']

            self.started_events.setdefault(session_id, asyncio.Event())
            done_future = self.done_futures.setdefault(session_id, asyncio.get_running_loop().create_future())

            for early_msg in self.early_msgs.pop(session_id, []):
                self.handle_session_msg(session_id, early_msg)

            return TaskHandle(session_id=session_id, done_future=done_future)


async def markup_dataset(
    df: pd.DataFrame,
    features_column: str,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    records = df.to_dict('records')
    frequency = build_feature_frequency(records, features_column)

    feature_stats_df = pd.DataFrame(
        sorted(
            ({'feature_name': name, 'frequency': count} for name, count in frequency.items()),
            key=lambda item: (-item['frequency'], item['feature_name']),
        )
    )

    tasks = []
    for row_idx, row in enumerate(records):
        brand, ranked_features = rank_row_features(row, features_column, frequency)
        feature_usage_plan = build_feature_usage_plan(
            ranked_features,
            query_count=TARGET_QUERY_COUNT,
        )
        model_input = build_model_input(
            row,
            features_column,
            brand,
            ranked_features,
            feature_usage_plan,
        )

        tasks.append({
            'row_idx': row_idx,
            'text': model_input,
            'prompt': PROMPT,
            'temperature': TEMPERATURE,
            'max_tokens': MAX_TOKENS,
        })

    sku_values: list[Optional[str]] = [None] * len(tasks)
    sku_errors: list[Optional[str]] = [None] * len(tasks)
    completed_rows: list[bool] = [False] * len(tasks)
    checkpoint_writer = CheckpointParquetWriter(
        base_df=df,
        output_dir=output_dir,
        chunk_size=OUTPUT_CHUNK_SIZE,
    )

    client = SchedulerClient(WS_URL, model=MODEL_NAME)
    await client.connect()
    in_flight: set[Any] = set()
    progress_log_path = get_progress_log_path()
    llm_bar = create_progress_bar(
        total=len(tasks),
        desc='LLM markup',
        log_path=progress_log_path,
        log_every=PROGRESS_LOG_EVERY,
    )

    async def wrap_result(
        handle: TaskHandle,
        row_idx: int,
    ) -> tuple[int, Optional[dict[str, Any]], Optional[str]]:
        try:
            return row_idx, await handle.result(), None
        except Exception as exc:
            return row_idx, None, str(exc)

    async def handle_done_task(done_task: Any) -> None:
        row_idx, raw_res, request_error = await done_task
        try:
            if request_error is not None or raw_res is None:
                sku_errors[row_idx] = request_error or 'job_failed'
            else:
                raw_text = extract_final_text(raw_res)
                queries, error = parse_queries_from_response(raw_res)

                if queries is not None:
                    if any(looks_like_missing_spaces(query) for query in queries):
                        queries = build_fallback_queries(records[row_idx], features_column, frequency)
                    sku_values[row_idx] = serialize_sku_queries(queries)
                else:
                    sku_errors[row_idx] = error or raw_text
        except Exception as exc:
            sku_errors[row_idx] = str(exc)
        finally:
            completed_rows[row_idx] = True
            checkpoint_writer.flush_ready_chunks(
                completed_rows=completed_rows,
                sku_values=sku_values,
                sku_errors=sku_errors,
                force=False,
            )
            llm_bar.update(1)

    try:
        for task in tasks:
            while len(in_flight) >= WINDOW_N:
                done, pending = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
                in_flight = pending
                for completed in done:
                    await handle_done_task(completed)

            handle = await client.submit(
                text=task['text'],
                prompt=task['prompt'],
                temperature=task['temperature'],
                max_tokens=task['max_tokens'],
            )
            in_flight.add(asyncio.create_task(wrap_result(handle, task['row_idx'])))

        while in_flight:
            done, pending = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
            in_flight = pending
            for completed in done:
                await handle_done_task(completed)
    finally:
        checkpoint_writer.flush_ready_chunks(
            completed_rows=completed_rows,
            sku_values=sku_values,
            sku_errors=sku_errors,
            force=True,
        )
        llm_bar.close()
        await client.close()

    result_df = df.copy()
    result_df[SKU_COLUMN] = sku_values
    result_df['sku_error'] = sku_errors
    return result_df, feature_stats_df, checkpoint_writer.chunk_paths


async def main() -> tuple[pd.DataFrame, pd.DataFrame]:
    input_paths = get_input_dataset_paths()
    output_dir = get_output_dataset_dir()
    df = load_input_tables(input_paths)
    features_column = detect_features_column(df.columns.tolist())
    df = clean_df(df, features_column)

    if ROW_LIMIT is not None:
        df = df.head(ROW_LIMIT).copy()

    result_df, feature_stats_df, chunk_paths = await markup_dataset(
        df,
        features_column,
        output_dir=output_dir,
    )

    ready_count = int(result_df[SKU_COLUMN].notna().sum())
    error_count = int(result_df['sku_error'].notna().sum())
    print(f'Input files combined: {len(input_paths)}')
    print(f'Saved parquet chunks: {len(chunk_paths)} -> {output_dir}')
    print(f'Progress log: {get_progress_log_path()}')
    print(f'Rows with ready sku JSON: {ready_count} / {len(result_df)}')
    print(f'Rows with errors: {error_count}')
    return result_df, feature_stats_df


if __name__ == '__main__':
    asyncio.run(main())
