#!/usr/bin/env python3
"""Pairwise cross-domain failure comparison: Inference Fixed Partitions vs. Inference Dynamic Partitions.

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
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.persistbench.output_manifest import (
    MODEL_LABELS,
    MODEL_ORDER,
    PARTITIONED_CHECKPOINTS,
    PARTITIONED_CUSTOM_CHECKPOINTS,
)

FIXED_CHECKPOINTS = list(PARTITIONED_CHECKPOINTS)
DYNAMIC_CHECKPOINTS = list(PARTITIONED_CUSTOM_CHECKPOINTS)

OUTDIR = Path(__file__).resolve().parent / "fixed_vs_dynamic_partitions_cd_overlap"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


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
        counts: dict[str, int] = {
            "both_pass": 0,
            "fixed_pass_dynamic_fail": 0,
            "dynamic_pass_fixed_fail": 0,
            "both_fail": 0,
        }
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


def draw_figure(counts: dict[str, dict[str, int]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    models_sorted = sorted(counts, key=sort_key)
    models_display = list(reversed(models_sorted))

    categories = ["both_pass", "fixed_pass_dynamic_fail", "dynamic_pass_fixed_fail", "both_fail"]
    labels = [
        "Both pass",
        "Fixed pass,\nDynamic fail",
        "Dynamic pass,\nFixed fail",
        "Both fail",
    ]
    colors = ["#54A24B", "#4C78A8", "#F58518", "#E45756"]

    data_matrix = np.array(
        [[counts[m][cat] for cat in categories] for m in models_display]
    )

    fig_height = max(4.0, 0.55 * len(models_display) + 2.2)
    fig, ax = plt.subplots(figsize=(11.0, fig_height))

    lefts = np.zeros(len(models_display))
    for cat, label, color, col_vals in zip(categories, labels, colors, data_matrix.T):
        ax.barh(
            range(len(models_display)),
            col_vals,
            left=lefts,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            label=label,
            height=0.62,
        )
        for j, (v, left) in enumerate(zip(col_vals, lefts)):
            if v > 0:
                ax.text(
                    left + v / 2,
                    j,
                    str(int(v)),
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    fontweight="semibold",
                    color="white" if v >= 8 else "#222222",
                )
        lefts = lefts + col_vals

    ax.set_yticks(range(len(models_display)))
    ax.set_yticklabels(models_display, fontsize=10)
    ax.set_xlabel("Number of cross-domain samples", fontweight="semibold", labelpad=8)
    ax.set_title(
        "Cross-Domain Sample Outcomes:\nInference Fixed Partitions vs. Inference Dynamic Partitions",
        fontsize=12,
        fontweight="semibold",
        pad=10,
        color="#222222",
    )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        fontsize=8.5,
        framealpha=0.9,
        edgecolor="#aaaaaa",
        ncol=4,
    )

    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(0.8)
        ax.spines[side].set_color("#555555")

    ax.tick_params(axis="both", which="major", length=3, width=0.8)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(output_dir / "fixed_vs_dynamic_partitions_cd_overlap.png", bbox_inches="tight")
    fig.savefig(output_dir / "fixed_vs_dynamic_partitions_cd_overlap.pdf", bbox_inches="tight")
    plt.close(fig)


CATEGORIES = ["both_pass", "fixed_pass_dynamic_fail", "dynamic_pass_fixed_fail", "both_fail"]
COL_HEADERS = ["Both Pass", "Fixed✓ Dyn✗", "Dyn✓ Fixed✗", "Both Fail"]


def print_table(counts: dict[str, dict[str, int]], k: int, threshold: int) -> None:
    models = sorted(counts, key=sort_key)
    if not models:
        print("No models found in either method.")
        return

    model_w = max(len(m) for m in models)
    col_w = max(max(len(h) for h in COL_HEADERS), 5)
    total_w = 5

    header_model = f"{'Model':<{model_w}}"
    header_cols = "  ".join(f"{h:^{col_w}}" for h in COL_HEADERS)
    header_total = f"  {'Total':>{total_w}}"
    sep = "-" * (model_w + 2 + len(header_cols) + len(header_total))

    print(
        "\nCross-domain overlap: Inference Fixed Partitions vs. Inference Dynamic Partitions"
    )
    print(
        f"Failure = max(first {k} scores) >= {threshold}  |  "
        "only entries present in both methods\n"
    )
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
    parser.add_argument("--output-dir", type=Path, default=OUTDIR)
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip writing PNG/PDF figures (still prints the console table).",
    )
    parser.add_argument(
        "--no-print",
        action="store_true",
        help="Skip the console table (still writes figures).",
    )
    args = parser.parse_args()

    fixed_data = load_cross_domain_scores(FIXED_CHECKPOINTS)
    dynamic_data = load_cross_domain_scores(DYNAMIC_CHECKPOINTS)

    counts = compute_overlap_counts(
        fixed_data, dynamic_data, k=args.k, threshold=args.threshold
    )

    if not args.no_print:
        print_table(counts, k=args.k, threshold=args.threshold)

    if not args.no_figures:
        draw_figure(counts, args.output_dir)
        print(f"Figure saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
