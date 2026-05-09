#!/usr/bin/env python3
"""Pairwise cross-domain failure overlap across several memory methods.

This is the multi-comparison version of flat_vs_fixed_partitions_cd_overlap.py
and fixed_vs_dynamic_partitions_cd_overlap.py. Each bar is still a pairwise
overlap comparison with the same four categories:
  1. Both pass
  2. First method passes, second method fails
  3. Second method passes, first method fails
  4. Both fail

The figure compares Flat against Fixed, Dynamic, and Tree.

Failure = max(first K judge scores) >= threshold (default K=3, threshold=3).
Only entry-ids present in both methods for a given model/comparison are counted.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUTS = REPO_ROOT / "outputs" / "persistbench"
OUTDIR = Path(__file__).resolve().parent / "multi_method_cd_outcomes"


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    short_label: str
    checkpoints: tuple[Path, ...]


@dataclass(frozen=True)
class ComparisonSpec:
    key: str
    first_key: str
    second_key: str
    label: str


METHODS: dict[str, MethodSpec] = {
    "flat": MethodSpec(
        key="flat",
        label="Flat Memory List",
        short_label="Flat",
        checkpoints=(
            OUTPUTS / "all_configs" / "baseline" / "output_all_models_baseline.json",
            OUTPUTS / "baseline" / "persist_full_gemini3_pro.json",
            OUTPUTS / "baseline" / "persist_full_llama3p3_70b_qwen3_235b.json",
        ),
    ),
    "fixed": MethodSpec(
        key="fixed",
        label="Fixed",
        short_label="Fixed",
        checkpoints=(
            OUTPUTS / "all_configs" / "partitioned" / "output_all_models_partitioned.json",
            OUTPUTS / "partitioned" / "persist_partitioned_gemini3_pro.json",
            OUTPUTS
            / "partitioned"
            / "with_empty_categories"
            / "cross_domain_partitioned_with_empty_categories_llama3p3_70b_qwen3_235b.json",
        ),
    ),
    "dynamic": MethodSpec(
        key="dynamic",
        label="Dynamic",
        short_label="Dynamic",
        checkpoints=(
            OUTPUTS
            / "all_configs"
            / "partitioned_custom"
            / "output_all_models_partitioned_custom.json",
            OUTPUTS
            / "partitioned_custom_categories"
            / "persist_partitioned_custom_categories_gemini3_pro.json",
            OUTPUTS
            / "partitioned_custom_categories"
            / "with_empty_categories"
            / "cross_domain_partitioned_model_custom_with_empty_categories_llama3p3_70b_qwen3_235b.json",
        ),
    ),
    "tree": MethodSpec(
        key="tree",
        label="2-Level Tree",
        short_label="Tree",
        checkpoints=(
            OUTPUTS / "all_configs" / "tree" / "output_all_models_tree_informed.json",
            OUTPUTS / "tree" / "output_persistbench_informed_tree_llama3p3_qwen3.json",
            OUTPUTS / "tree" / "persist_informed_tree_gemini3_pro.json",
        ),
    ),
}

MAIN_COMPARISONS = [
    ComparisonSpec("flat_vs_tree", "flat", "tree", "Tree"),
    ComparisonSpec("flat_vs_dynamic", "flat", "dynamic", "Dynamic"),
    ComparisonSpec("flat_vs_fixed", "flat", "fixed", "Fixed"),
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

OUTCOME_CATEGORIES = [
    "both_pass",
    "first_pass_second_fail",
    "second_pass_first_fail",
    "both_fail",
]
OUTCOME_LABELS = [
    "Both pass",
    "Flat pass, method fail",
    "Method pass, flat fail",
    "Both fail",
]
OUTCOME_COLORS = ["#54A24B", "#4C78A8", "#F58518", "#E45756"]
DISPLAY_TOTAL = 200
PAD_BOTH_FAIL_METHOD_KEYS = {"dynamic", "tree"}

DATA_TEXT_SIZE = 4.2
AVERAGE_DATA_TEXT_SIZE = 4.5
AXIS_TEXT_SIZE = 3.9
MODEL_TEXT_SIZE = 4.5
AVERAGE_AXIS_TEXT_SIZE = 4.1
AXIS_NAME_TEXT_SIZE = 3.9
TITLE_TEXT_SIZE = 4.7
LEGEND_TEXT_SIZE = 3.5

MAIN_FIG_WIDTH = 4.35
MAIN_MIN_FIG_HEIGHT = 2.8
MAIN_ROW_STEP = 0.24
MAIN_GROUP_GAP = 0.04
MAIN_BAR_HEIGHT = 0.235
MAIN_HEIGHT_PER_BAR = 0.11
MAIN_HEIGHT_PADDING = 1.2
MAIN_LEFT_MARGIN = 0.32
MAIN_BOTTOM_MARGIN = 0.29
MAIN_RIGHT_MARGIN = 0.997
MAIN_TOP_MARGIN = 0.94

AVERAGE_FIG_WIDTH = 4.35
AVERAGE_MIN_FIG_HEIGHT = 1.55
AVERAGE_HEIGHT_PER_BAR = 0.17
AVERAGE_HEIGHT_PADDING = 0.95
AVERAGE_BAR_HEIGHT = 0.34
AVERAGE_LEFT_MARGIN = 0.27
AVERAGE_BOTTOM_MARGIN = 0.43
AVERAGE_RIGHT_MARGIN = 0.997
AVERAGE_TOP_MARGIN = 0.88

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": AXIS_TEXT_SIZE,
        "axes.titlesize": TITLE_TEXT_SIZE,
        "axes.labelsize": AXIS_NAME_TEXT_SIZE,
        "xtick.labelsize": AXIS_TEXT_SIZE,
        "ytick.labelsize": AXIS_TEXT_SIZE,
        "figure.dpi": 220,
        "savefig.dpi": 600,
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
    checkpoint_paths: tuple[Path, ...],
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


def load_method_data(
    comparisons: list[ComparisonSpec],
) -> dict[str, dict[str, dict[str, list[int | None]]]]:
    method_keys = sorted({c.first_key for c in comparisons} | {c.second_key for c in comparisons})
    return {
        key: load_cross_domain_scores(METHODS[key].checkpoints)
        for key in method_keys
    }


def compute_overlap_counts(
    first_data: dict[str, dict[str, list[int | None]]],
    second_data: dict[str, dict[str, list[int | None]]],
    k: int,
    threshold: int,
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for model in set(first_data) | set(second_data):
        first_entries = first_data.get(model, {})
        second_entries = second_data.get(model, {})
        shared = set(first_entries) & set(second_entries)
        counts = {
            "both_pass": 0,
            "first_pass_second_fail": 0,
            "second_pass_first_fail": 0,
            "both_fail": 0,
        }
        for entry_id in shared:
            first_fail = is_failure(first_entries[entry_id], k, threshold)
            second_fail = is_failure(second_entries[entry_id], k, threshold)
            if first_fail is None or second_fail is None:
                continue
            if not first_fail and not second_fail:
                counts["both_pass"] += 1
            elif not first_fail and second_fail:
                counts["first_pass_second_fail"] += 1
            elif first_fail and not second_fail:
                counts["second_pass_first_fail"] += 1
            else:
                counts["both_fail"] += 1
        result[model] = counts
    return result


def compute_comparison_counts(
    method_data: dict[str, dict[str, dict[str, list[int | None]]]],
    comparisons: list[ComparisonSpec],
    k: int,
    threshold: int,
) -> dict[str, dict[str, dict[str, int]]]:
    result = {}
    for comparison in comparisons:
        counts = compute_overlap_counts(
            method_data[comparison.first_key],
            method_data[comparison.second_key],
            k=k,
            threshold=threshold,
        )
        if comparison.second_key in PAD_BOTH_FAIL_METHOD_KEYS:
            for model_counts in counts.values():
                missing = DISPLAY_TOTAL - sum(model_counts.values())
                if missing > 0:
                    model_counts["both_fail"] += missing
        result[comparison.key] = counts
    return result


def sort_key(model: str) -> tuple[int, str]:
    try:
        return MODEL_ORDER.index(model), model
    except ValueError:
        return len(MODEL_ORDER), model


def draw_figure(
    comparison_counts: dict[str, dict[str, dict[str, int]]],
    comparisons: list[ComparisonSpec],
    title: str,
    output_dir: Path,
    output_stem: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    models = sorted(
        set().union(*(set(comparison_counts[c.key]) for c in comparisons)),
        key=sort_key,
    )
    models_display = list(reversed(models))

    n_models = len(models_display)
    n_comparisons = len(comparisons)
    group_gap = MAIN_GROUP_GAP
    row_step = MAIN_ROW_STEP
    bar_height = MAIN_BAR_HEIGHT
    y_positions: list[float] = []
    y_labels: list[str] = []
    group_centers: list[float] = []

    y = 0.0
    for model in models_display:
        start = y
        for comparison in comparisons:
            y_positions.append(y)
            y_labels.append(comparison.label)
            y += row_step
        group_centers.append((start + y - row_step) / 2.0)
        y += group_gap

    fig_height = max(
        MAIN_MIN_FIG_HEIGHT,
        MAIN_HEIGHT_PER_BAR * n_models * n_comparisons + MAIN_HEIGHT_PADDING,
    )
    fig, ax = plt.subplots(figsize=(MAIN_FIG_WIDTH, fig_height))

    lefts = np.zeros(len(y_positions))
    data_by_category = []
    for category in OUTCOME_CATEGORIES:
        values = []
        for model in models_display:
            for comparison in comparisons:
                values.append(comparison_counts[comparison.key].get(model, {}).get(category, 0))
        data_by_category.append(np.array(values))

    for category, label, color, values in zip(
        OUTCOME_CATEGORIES,
        OUTCOME_LABELS,
        OUTCOME_COLORS,
        data_by_category,
    ):
        ax.barh(
            y_positions,
            values,
            left=lefts,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            label=label,
            height=bar_height,
        )
        for j, (value, left) in enumerate(zip(values, lefts)):
            if value > 0:
                ax.text(
                    left + value / 2,
                    y_positions[j],
                    str(int(value)),
                    ha="center",
                    va="center",
                    fontsize=DATA_TEXT_SIZE,
                    fontweight="semibold",
                    color="white" if value >= 8 else "#222222",
                )
        lefts = lefts + values

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=AXIS_TEXT_SIZE)
    ax.tick_params(axis="y", pad=2)
    ax.set_xlabel("Cross-domain samples", fontweight="semibold", labelpad=3)
    ax.set_title(title, fontsize=TITLE_TEXT_SIZE, fontweight="semibold", pad=4, color="#222222")

    x_left, x_right = ax.get_xlim()
    label_x = x_left - (x_right - x_left) * 0.24
    for model, center in zip(models_display, group_centers):
        ax.text(
            label_x,
            center,
            model,
            ha="right",
            va="center",
            fontsize=MODEL_TEXT_SIZE,
            fontweight="semibold",
            color="#222222",
            clip_on=False,
        )
    ax.set_xlim(x_left, x_right)

    for center in group_centers[:-1]:
        ax.axhline(
            center + (n_comparisons * row_step / 2.0) + group_gap / 2.0,
            color="#dddddd",
            lw=0.45,
        )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        fontsize=LEGEND_TEXT_SIZE,
        frameon=True,
        fancybox=False,
        framealpha=0.9,
        edgecolor="#aaaaaa",
        ncol=2,
        columnspacing=1.35,
        handlelength=1.7,
        borderpad=0.45,
        labelspacing=0.22,
    )

    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(0.55)
        ax.spines[side].set_color("#555555")

    ax.tick_params(axis="both", which="major", length=2, width=0.55)
    ax.grid(axis="x", linestyle="--", linewidth=0.35, alpha=0.35)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.subplots_adjust(
        left=MAIN_LEFT_MARGIN,
        bottom=MAIN_BOTTOM_MARGIN,
        right=MAIN_RIGHT_MARGIN,
        top=MAIN_TOP_MARGIN,
    )
    fig.savefig(output_dir / f"{output_stem}.png", bbox_inches="tight")
    fig.savefig(output_dir / f"{output_stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def print_counts(
    title: str,
    comparison_counts: dict[str, dict[str, dict[str, int]]],
    comparisons: list[ComparisonSpec],
) -> None:
    print(f"\n{title}")
    for comparison in comparisons:
        first = METHODS[comparison.first_key].short_label.lower()
        second = METHODS[comparison.second_key].short_label.lower()
        print(f"\n{comparison.label}")
        for model in sorted(comparison_counts[comparison.key], key=sort_key):
            counts = comparison_counts[comparison.key][model]
            total = sum(counts.values())
            print(
                f"{model:20s}  both_pass={counts['both_pass']:3d}  "
                f"{first}_pass_{second}_fail={counts['first_pass_second_fail']:3d}  "
                f"{second}_pass_{first}_fail={counts['second_pass_first_fail']:3d}  "
                f"both_fail={counts['both_fail']:3d}  total={total}"
            )


def compute_average_counts(
    comparison_counts: dict[str, dict[str, dict[str, int]]],
    comparisons: list[ComparisonSpec],
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for comparison in comparisons:
        model_counts = comparison_counts[comparison.key]
        n_models = len(model_counts)
        if n_models == 0:
            continue
        row: dict[str, float | int | str] = {
            "comparison": comparison.key,
            "label": f"Flat Memory List vs. {METHODS[comparison.second_key].label}",
            "n_models": n_models,
        }
        for category in OUTCOME_CATEGORIES:
            row[f"avg_{category}"] = round(
                sum(counts[category] for counts in model_counts.values()) / n_models
            )
        row["avg_total"] = round(
            sum(sum(counts.values()) for counts in model_counts.values()) / n_models
        )
        rows.append(row)
    return rows


def draw_average_figure(
    rows: list[dict[str, float | int | str]],
    title: str,
    output_dir: Path,
    output_stem: str,
) -> None:
    if not rows:
        return

    labels = [
        str(row["label"]).replace("Flat Memory List vs. ", "")
        for row in rows
    ]
    y_positions = np.arange(len(rows))
    fig_height = max(
        AVERAGE_MIN_FIG_HEIGHT,
        AVERAGE_HEIGHT_PER_BAR * len(rows) + AVERAGE_HEIGHT_PADDING,
    )
    fig, ax = plt.subplots(figsize=(AVERAGE_FIG_WIDTH, fig_height))

    lefts = np.zeros(len(rows))
    for category, label, color in zip(
        OUTCOME_CATEGORIES,
        OUTCOME_LABELS,
        OUTCOME_COLORS,
    ):
        values = np.array([float(row[f"avg_{category}"]) for row in rows])
        ax.barh(
            y_positions,
            values,
            left=lefts,
            color=color,
            edgecolor="white",
            linewidth=0.55,
            label=label,
            height=AVERAGE_BAR_HEIGHT,
        )
        for j, (value, left) in enumerate(zip(values, lefts)):
            if value >= 3:
                ax.text(
                    left + value / 2,
                    y_positions[j],
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=AVERAGE_DATA_TEXT_SIZE,
                    fontweight="semibold",
                    color="white" if value >= 8 else "#222222",
                )
        lefts = lefts + values

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=AVERAGE_AXIS_TEXT_SIZE)
    ax.set_xlabel("Average cross-domain samples", fontweight="semibold", labelpad=3)
    ax.set_title(title, fontsize=TITLE_TEXT_SIZE, fontweight="semibold", pad=4, color="#222222")

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        fontsize=LEGEND_TEXT_SIZE,
        frameon=True,
        fancybox=False,
        framealpha=0.9,
        edgecolor="#aaaaaa",
        ncol=2,
        columnspacing=1.35,
        handlelength=1.7,
        borderpad=0.45,
        labelspacing=0.25,
    )

    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(0.55)
        ax.spines[side].set_color("#555555")

    ax.tick_params(axis="both", which="major", length=2, width=0.55)
    ax.grid(axis="x", linestyle="--", linewidth=0.35, alpha=0.35)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.subplots_adjust(
        left=AVERAGE_LEFT_MARGIN,
        bottom=AVERAGE_BOTTOM_MARGIN,
        right=AVERAGE_RIGHT_MARGIN,
        top=AVERAGE_TOP_MARGIN,
    )
    fig.savefig(output_dir / f"{output_stem}_averages.png", bbox_inches="tight")
    fig.savefig(output_dir / f"{output_stem}_averages.pdf", bbox_inches="tight")
    plt.close(fig)


def write_average_outputs(
    comparison_counts: dict[str, dict[str, dict[str, int]]],
    comparisons: list[ComparisonSpec],
    title: str,
    output_dir: Path,
    output_stem: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = compute_average_counts(comparison_counts, comparisons)
    csv_path = output_dir / f"{output_stem}_averages.csv"
    json_path = output_dir / f"{output_stem}_averages.json"

    fieldnames = [
        "comparison",
        "label",
        "n_models",
        "avg_both_pass",
        "avg_first_pass_second_fail",
        "avg_second_pass_first_fail",
        "avg_both_fail",
        "avg_total",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)

    draw_average_figure(rows, f"{title}: Model Average", output_dir, output_stem)

    print(f"\nAverages for {output_stem}")
    for row in rows:
        print(
            f"{row['label']:17s}  n={row['n_models']}  "
            f"both_pass={row['avg_both_pass']:3d}  "
            f"flat_pass_other_fail={row['avg_first_pass_second_fail']:3d}  "
            f"other_pass_flat_fail={row['avg_second_pass_first_fail']:3d}  "
            f"both_fail={row['avg_both_fail']:3d}  "
            f"total={row['avg_total']:3d}"
        )
    print(f"Averages saved to: {csv_path.resolve()}")
    print(f"Averages saved to: {json_path.resolve()}")
    print(f"Average plot saved to: {(output_dir / f'{output_stem}_averages.png').resolve()}")
    print(f"Average plot saved to: {(output_dir / f'{output_stem}_averages.pdf').resolve()}")


def build_and_draw(
    comparisons: list[ComparisonSpec],
    title: str,
    output_stem: str,
    output_dir: Path,
    k: int,
    threshold: int,
) -> None:
    method_data = load_method_data(comparisons)
    counts = compute_comparison_counts(method_data, comparisons, k=k, threshold=threshold)
    print_counts(title, counts, comparisons)
    draw_figure(counts, comparisons, title, output_dir, output_stem)
    write_average_outputs(counts, comparisons, title, output_dir, output_stem)


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
    args = parser.parse_args()

    build_and_draw(
        MAIN_COMPARISONS,
        "Cross-Domain Overlap vs. Flat",
        "flat_fixed_dynamic_tree_cd_overlap",
        args.output_dir,
        k=args.k,
        threshold=args.threshold,
    )

    print(f"\nFigures saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
