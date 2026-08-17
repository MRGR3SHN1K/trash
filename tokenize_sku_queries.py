from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from hybrid_dual_path_model import merge_numeric_tokens


def load_queries(input_path: Path) -> list[str]:
    text = input_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None

    if payload is not None:
        return extract_queries(payload)

    return [line.strip() for line in text.splitlines() if line.strip()]


def extract_queries(payload: Any) -> list[str]:
    if isinstance(payload, str):
        return [payload.strip()] if payload.strip() else []

    if isinstance(payload, list):
        queries: list[str] = []
        for item in payload:
            queries.extend(extract_queries(item))
        return queries

    if isinstance(payload, dict):
        if "queries" in payload:
            return extract_queries(payload["queries"])

        if "query" in payload and isinstance(payload["query"], str):
            query = payload["query"].strip()
            return [query] if query else []

        queries: list[str] = []
        for value in payload.values():
            queries.extend(extract_queries(value))
        return queries

    return []


def render_query(index: int, query: str, tokenizer: Tokenizer) -> str:
    encoding = tokenizer.encode(query)
    merged_tokens = merge_numeric_tokens(
        query,
        token_ids=encoding.ids,
        tokens=encoding.tokens,
        offsets=encoding.offsets,
    )
    token_rows: list[str] = []
    mask_bits: list[str] = []
    compact_tokens: list[str] = []

    for merged in merged_tokens:
        is_numeric = merged.token_type == "INT"
        value_suffix = f" value={merged.numeric_value}" if is_numeric else ""
        ids_repr = ",".join(str(token_id) for token_id in merged.source_ids)
        parts_suffix = ""
        if len(merged.source_tokens) > 1 or merged.text != merged.source_tokens[0]:
            parts_suffix = f" source_tokens={[token for token in merged.source_tokens]!r}"
        mask_bits.append("1" if is_numeric else "0")
        compact_tokens.append(f"{merged.text!r}:{merged.token_type}")
        token_rows.append(
            f"  [{merged.token_type}] ids=[{ids_repr}] token={merged.text!r}{value_suffix}{parts_suffix}"
        )

    lines = [
        f"Query {index}: {query}",
        f"Type mask (1=INT, 0=STR): {' '.join(mask_bits)}",
        f"Compact view: {' | '.join(compact_tokens)}",
        "Tokens:",
        *token_rows,
    ]
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize SKU/search queries and mark INT/STR tokens.")
    parser.add_argument(
        "--input",
        default="sku.txt",
        help="Input file with queries. Supports JSON arrays/objects or plain one-query-per-line txt.",
    )
    parser.add_argument(
        "--tokenizer",
        default="tokenizer (1).json",
        help="Path to tokenizer json file.",
    )
    parser.add_argument(
        "--output",
        default="sku_tokenized.txt",
        help="Output file with token breakdown.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    tokenizer_path = Path(args.tokenizer)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

    queries = load_queries(input_path)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    if not queries:
        output = (
            f"No queries found in {input_path.name}.\n"
            "Put one query per line or a JSON array/object with queries, then rerun the script.\n"
        )
    else:
        rendered_queries = [render_query(index + 1, query, tokenizer) for index, query in enumerate(queries)]
        output = "\n".join(rendered_queries)

    output_path.write_text(output, encoding="utf-8")
    print(f"Processed {len(queries)} queries -> {output_path}")


if __name__ == "__main__":
    main()
