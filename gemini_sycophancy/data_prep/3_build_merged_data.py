"""Combine the existing 300-entry partitioned/tree files with the newly-built
200-entry sycophancy files into 500-entry merged input files.

Run this after 1_partition_sycophancy.py and 2_tree_sycophancy.py have produced
their outputs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DATA_DIR = HERE.parent / "data"


def _read(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"ERROR: missing {path}")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"  wrote {len(rows)} entries -> {path}")


def _expect(label: str, rows: list[dict], expected: int) -> None:
    if len(rows) != expected:
        sys.exit(
            f"ERROR: {label} has {len(rows)} entries (expected {expected}). "
            f"Re-run the relevant data_prep script to completion before merging."
        )


def main() -> None:
    print("Merging partitioned data (existing 300 + new sycophancy 200 -> 500)")
    existing_part = REPO_ROOT / "benchmark_samples/persistbench/partitioned/gemini3_pro/full_benchmark.jsonl"
    new_part = DATA_DIR / "partitioned_sycophancy_only/google_gemini-3.1-pro-preview/full_benchmark.jsonl"
    merged_part = DATA_DIR / "partitioned_full_500/full_benchmark.jsonl"
    e_rows = _read(existing_part); n_rows = _read(new_part)
    _expect("existing partitioned (gemini3_pro 300)", e_rows, 300)
    _expect("new partitioned sycophancy (200)", n_rows, 200)
    _write(merged_part, e_rows + n_rows)

    print("\nMerging tree data (existing llama3p3 300 + new gemini sycophancy 200 -> 500)")
    existing_tree = REPO_ROOT / "benchmark_samples/persistbench/tree/llama3p3_70b/cross_domain_and_beneficial_tree_llama3p3.jsonl"
    new_tree = DATA_DIR / "tree_sycophancy_only/google_gemini-3.1-pro-preview/full_benchmark.jsonl"
    merged_tree = DATA_DIR / "tree_full_500/full_benchmark_tree.jsonl"
    et_rows = _read(existing_tree); nt_rows = _read(new_tree)
    _expect("existing tree (llama3p3 300)", et_rows, 300)
    _expect("new tree sycophancy (200)", nt_rows, 200)
    _write(merged_tree, et_rows + nt_rows)

    print("\nDone. Use these merged files as 'input' in the configs.")


if __name__ == "__main__":
    main()
