#!/usr/bin/env python3
"""Print cross-domain overlap counts: Inference Fixed Partitions vs. Inference Dynamic Partitions.

For each model, counts how many cross-domain samples fall into each of 4 categories:
  1. Both pass  (neither fails)
  2. Fixed Partitions passes, Dynamic Partitions fails
  3. Dynamic Partitions passes, Fixed Partitions fails
  4. Both fail

Failure = max(first K judge scores) >= threshold (default K=3, threshold=3).
Only entry-ids present in both methods for a given model are counted.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = REPO_ROOT / "outputs" / "persistbench"

FIXED_CHECKPOINTS = [
    OUTPUTS / "all_configs" / "partitioned" / "output_all_models_partitioned.json",
    OUTPUTS / "partitioned" / "persist_partitioned_gemini3_pro.json",
    OUTPUTS
    / "partitioned"
    / "with_empty_categories"
    / "cross_domain_partitioned_with_empty_categories_llama3p3_70b_qwen3_235b.json",
]

DYNAMIC_CHECKPOINTS = [
    OUTPUTS / "all_configs" / "partitioned_custom" / "output_all_models_partitioned_custom.json",
    OUTPUTS / "partitioned_custom_categories" / "persist_partitioned_custom_categories_gemini3_pro.json",
    OUTPUTS
    / "partitioned_custom_categories"
    / "with_empty_categories"
    / "cross_domain_partitioned_model_custom_with_empty_categories_llama3p3_70b_qwen3_235b.json",
]

MODEL_LABELS = {
    "DeepSeek-V3.2": "DeepSeek V3.2",
    "gpt-oss-120b": "GPT-OSS 120B",
    "google/gemini-3-pro-preview": "Gemini 3.1 Pro",
    "google/gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "Llama-3.3-70B-Instruct": "Llama 3.3-70B",
    "meta/llama-3.3-70b-instruct-maas": "Llama 3.3-70B",
    "qwen/qwen3-235b-a22b-instruct-2507-maas": "Qwen 3-235B",
    "xai/grok-4.1-fast-non-reasoning": "Grok 4.1 Fast",
    "zai-org/glm-4.7-maas": "GLM-4.7",
}

MODEL_ORDER = [
    "Llama 3.3-70B",
    "Qwen 3-235B",
    "DeepSeek V3.2",
    "GPT-OSS 120B",
    "GLM-4.7",
    "Grok 4.1 Fast",
    "Gemini 3.1 Pro",
]

CATEGORIES = ["both_pass", "fixed_pass_dynamic_fail", "dynamic_pass_fixed_fail", "both_fail"]
COL_HEADERS = ["Both Pass", "Fixed✓ Dyn✗", "Dyn✓ Fixed✗", "Both Fail"]


def normalize_failure_type(raw: str | None) -> str:
    if raw == "positive_memory_usage":
        return "beneficial_memory_usage"
    return raw or "cross_domain"


def extract_score(generation: dict) -> int | None:
    if generation.get("error"):
        return None
    judge = generation.get("judge")
    if judge is None:
        return None
    return judge.get("score")


def load_cross_domain_scores(
    checkpoint_paths: list[Path],
) -> dict[str, dict[str, list[int | None]]]:
    """Returns {model_label: {entry_id: [scores]}} for cross_domain entries only."""
    data: dict[str, dict[str, list[int | None]]] = defaultdict(dict)
    for path in checkpoint_paths:
        if not path.exists():
            print(f"Warning: missing {path}")
            continue
        with path.open(encoding="utf-8") as fh:
            checkpoint = json.load(fh)
        for entry_id, entry in checkpoint.get("entries", {}).items():
            ft = normalize_failure_type(entry.get("failure_type") or entry.get("leakage_type"))
            if ft != "cross_domain":
                continue
            for model_raw, model_data in entry.get("results", {}).items():
                label = MODEL_LABELS.get(model_raw)
                if label is None:
                    continue
                scores = [extract_score(g) for g in model_data.get("generations", [])]
                prev = data[label].get(entry_id)
                if prev is None or len(scores) > len(prev):
                    data[label][entry_id] = scores
    return dict(data)


def is_failure(scores: list[int | None], k: int, threshold: int) -> bool | None:
    valid = [s for s in scores[:k] if s is not None]
    if not valid:
        return None
    return max(valid) >= threshold


def compute_overlap_counts(
    fixed_data: dict[str, dict[str, list[int | None]]],
    dynamic_data: dict[str, dict[str, list[int | None]]],
    k: int,
    threshold: int,
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for model in set(fixed_data) | set(dynamic_data):
        fixed_entries = fixed_data.get(model, {})
        dynamic_entries = dynamic_data.get(model, {})
        shared = set(fixed_entries) & set(dynamic_entries)
        counts: dict[str, int] = dict.fromkeys(CATEGORIES, 0)
        for entry_id in shared:
            fixed_fail = is_failure(fixed_entries[entry_id], k, threshold)
            dynamic_fail = is_failure(dynamic_entries[entry_id], k, threshold)
            if fixed_fail is None or dynamic_fail is None:
                continue
            if not fixed_fail and not dynamic_fail:
                counts["both_pass"] += 1
            elif not fixed_fail and dynamic_fail:
                counts["fixed_pass_dynamic_fail"] += 1
            elif fixed_fail and not dynamic_fail:
                counts["dynamic_pass_fixed_fail"] += 1
            else:
                counts["both_fail"] += 1
        result[model] = counts
    return result


def sort_key(model: str) -> tuple[int, str]:
    try:
        return MODEL_ORDER.index(model), model
    except ValueError:
        return len(MODEL_ORDER), model


def print_table(counts: dict[str, dict[str, int]], k: int, threshold: int) -> None:
    models = sorted(counts, key=sort_key)

    model_w = max(len(m) for m in models)
    col_w = max(max(len(h) for h in COL_HEADERS), 5)
    total_w = 5

    header_model = f"{'Model':<{model_w}}"
    header_cols = "  ".join(f"{h:^{col_w}}" for h in COL_HEADERS)
    header_total = f"  {'Total':>{total_w}}"
    sep = "-" * (model_w + 2 + len(header_cols) + len(header_total))

    print(f"\nCross-domain overlap: Inference Fixed Partitions vs. Inference Dynamic Partitions")
    print(f"Failure = max(first {k} scores) >= {threshold}  |  only entries present in both methods\n")
    print(f"{header_model}  {header_cols}{header_total}")
    print(sep)

    col_totals: dict[str, int] = dict.fromkeys(CATEGORIES, 0)
    for model in models:
        c = counts[model]
        total = sum(c.values())
        row_cols = "  ".join(f"{c[cat]:^{col_w}}" for cat in CATEGORIES)
        print(f"{model:<{model_w}}  {row_cols}  {total:>{total_w}}")
        for cat in CATEGORIES:
            col_totals[cat] += c[cat]

    print(sep)
    grand_total = sum(col_totals.values())
    totals_row = "  ".join(f"{col_totals[cat]:^{col_w}}" for cat in CATEGORIES)
    print(f"{'Total':<{model_w}}  {totals_row}  {grand_total:>{total_w}}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=3, help="K for FR@K failure check (default 3)")
    parser.add_argument(
        "--threshold",
        type=int,
        default=3,
        help="Score threshold: max(scores[:k]) >= threshold means failure (default 3)",
    )
    args = parser.parse_args()

    fixed_data = load_cross_domain_scores(FIXED_CHECKPOINTS)
    dynamic_data = load_cross_domain_scores(DYNAMIC_CHECKPOINTS)
    counts = compute_overlap_counts(fixed_data, dynamic_data, k=args.k, threshold=args.threshold)
    print_table(counts, k=args.k, threshold=args.threshold)


if __name__ == "__main__":
    main()
