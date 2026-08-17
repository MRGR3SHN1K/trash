from __future__ import annotations

import argparse
import importlib.util
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


GROUP_NAMES = ("kantselyariya", "mebel", "odezhda")
REQUIRED_COLUMNS = ("category_name", "is_actual", "price", "feedbacks_count")


@dataclass
class RunStats:
    files_processed: int = 0
    total_rows: int = 0
    kept_rows: int = 0
    skipped_rows: int = 0
    rows_per_group: dict[str, int] = field(
        default_factory=lambda: {group_name: 0 for group_name in GROUP_NAMES}
    )


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Фильтрует parquet-файлы data* из downloads и раскладывает строки по "
            "downloads/<group>/v1 по спискам категорий."
        )
    )
    parser.add_argument(
        "--downloads-dir",
        type=Path,
        default=base_dir / "downloads",
        help="Папка с исходными parquet-файлами data*.",
    )
    parser.add_argument(
        "--categories-module",
        type=Path,
        default=base_dir / "categories_lists.py",
        help="Путь до Python-файла со списками kantselyariya, mebel, odezhda.",
    )
    parser.add_argument(
        "--version",
        default="v1",
        help="Имя подпапки версии внутри каждого выходного датасета.",
    )
    parser.add_argument(
        "--reset-output",
        action="store_true",
        help="Удалить ранее сгенерированные parquet-файлы в выходных папках перед запуском.",
    )
    return parser.parse_args()


def load_category_groups(module_path: Path) -> dict[str, set[str]]:
    if not module_path.exists():
        raise FileNotFoundError(f"Не найден файл со списками категорий: {module_path}")

    spec = importlib.util.spec_from_file_location("categories_lists", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Не удалось загрузить модуль категорий: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    groups: dict[str, set[str]] = {}
    for group_name in GROUP_NAMES:
        raw_categories = getattr(module, group_name, None)
        if raw_categories is None:
            raise AttributeError(
                f"В файле {module_path} не найден список '{group_name}'."
            )

        categories = {
            str(category).strip()
            for category in raw_categories
            if str(category).strip()
        }
        groups[group_name] = categories

    return groups


def build_category_lookup(groups: dict[str, set[str]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    duplicates: dict[str, set[str]] = {}

    for group_name, categories in groups.items():
        for category in categories:
            existing_group = lookup.get(category)
            if existing_group is None:
                lookup[category] = group_name
                continue

            if existing_group != group_name:
                duplicates.setdefault(category, {existing_group}).add(group_name)

    if duplicates:
        preview = "; ".join(
            f"{category} -> {', '.join(sorted(group_names))}"
            for category, group_names in sorted(duplicates.items())
        )
        raise ValueError(
            "Найдены категории, которые одновременно лежат в нескольких списках: "
            f"{preview}"
        )

    return lookup


def find_input_files(downloads_dir: Path) -> list[Path]:
    if not downloads_dir.exists():
        raise FileNotFoundError(f"Не найдена папка downloads: {downloads_dir}")
    if not downloads_dir.is_dir():
        raise NotADirectoryError(f"Путь downloads должен быть папкой: {downloads_dir}")

    files = [
        path
        for path in sorted(downloads_dir.iterdir())
        if path.is_file()
        and path.name.lower().startswith("data")
        and (path.suffix.lower() == ".parquet" or path.suffix == "")
    ]

    if not files:
        raise FileNotFoundError(
            f"В {downloads_dir} не найдено входных parquet-файлов вида data*."
        )

    return files


def reset_output_dirs(downloads_dir: Path, version: str) -> None:
    for group_name in GROUP_NAMES:
        target_dir = downloads_dir / group_name / version
        if not target_dir.exists():
            continue
        for parquet_file in target_dir.glob("*.parquet"):
            parquet_file.unlink()


def ensure_required_columns(df: pd.DataFrame, source_path: Path) -> None:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"В файле {source_path} не хватает колонок: {', '.join(missing_columns)}"
        )


def build_output_path(
    downloads_dir: Path,
    group_name: str,
    version: str,
    source_path: Path,
) -> Path:
    file_name = (
        source_path.name
        if source_path.suffix.lower() == ".parquet"
        else f"{source_path.name}.parquet"
    )
    return downloads_dir / group_name / version / file_name


def process_file(
    source_path: Path,
    downloads_dir: Path,
    version: str,
    category_lookup: dict[str, str],
    stats: RunStats,
) -> None:
    df = pd.read_parquet(source_path)
    ensure_required_columns(df, source_path)

    stats.files_processed += 1
    stats.total_rows += len(df)

    category_values = df["category_name"].astype("string").str.strip()
    bucket = category_values.map(category_lookup)
    is_actual_mask = pd.to_numeric(df["is_actual"], errors="coerce").eq(1)
    price_mask = pd.to_numeric(df["price"], errors="coerce").gt(0)
    feedbacks_mask = pd.to_numeric(df["feedbacks_count"], errors="coerce").gt(0)

    valid_mask = bucket.notna() & is_actual_mask & price_mask & feedbacks_mask
    filtered_df = df.loc[valid_mask].copy()
    filtered_df["category_name"] = category_values.loc[valid_mask].values
    filtered_df["__bucket"] = bucket.loc[valid_mask].values

    stats.kept_rows += len(filtered_df)
    stats.skipped_rows += len(df) - len(filtered_df)

    for group_name in GROUP_NAMES:
        output_path = build_output_path(downloads_dir, group_name, version, source_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        group_df = filtered_df.loc[filtered_df["__bucket"] == group_name].drop(
            columns="__bucket"
        )

        if group_df.empty:
            if output_path.exists():
                output_path.unlink()
            continue

        group_df.to_parquet(output_path, index=False)
        stats.rows_per_group[group_name] += len(group_df)


def print_summary(downloads_dir: Path, version: str, stats: RunStats) -> None:
    print(f"Обработано файлов: {stats.files_processed}")
    print(f"Прочитано строк: {stats.total_rows}")
    print(f"Оставлено строк: {stats.kept_rows}")
    print(f"Пропущено строк: {stats.skipped_rows}")
    for group_name in GROUP_NAMES:
        target_dir = downloads_dir / group_name / version
        print(f"{group_name}: {stats.rows_per_group[group_name]} строк -> {target_dir}")


def main() -> None:
    args = parse_args()
    downloads_dir = args.downloads_dir.resolve()
    categories_module = args.categories_module.resolve()

    groups = load_category_groups(categories_module)
    category_lookup = build_category_lookup(groups)
    input_files = find_input_files(downloads_dir)

    if args.reset_output:
        reset_output_dirs(downloads_dir, args.version)

    stats = RunStats()
    for source_path in input_files:
        process_file(source_path, downloads_dir, args.version, category_lookup, stats)

    print_summary(downloads_dir, args.version, stats)


if __name__ == "__main__":
    main()
