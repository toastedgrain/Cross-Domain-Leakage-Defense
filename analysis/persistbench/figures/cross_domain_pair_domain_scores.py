#!/usr/bin/env python3
"""Pair-level PersistBench cross-domain scores by memory/query domain."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.persistbench.figures.cross_domain_query_domain_scores import (
    DEFAULT_BENCHMARK,
    DOMAIN_LABELS,
    MEMORY_STRUCTURES,
    MODEL_LABELS,
    MODEL_ORDER,
)

OUTDIR = Path(__file__).resolve().parent / "cross_domain_pair_domain_scores"

METHODS = [
    "flat_memory_list",
    "partitions",
    "dynamic_partitions",
    "2_level_tree",
]

GRID_METHODS = [
    ["flat_memory_list", "partitions"],
    ["dynamic_partitions", "2_level_tree"],
]

OVERALL_AVERAGE_COLOR = "#d9e8f5"
OVERALL_AVERAGE_EDGE_COLOR = "#2f5f85"

DOMAIN_ORDER = [
    "Personal Beliefs (Political, Religious, and Social)",
    "Educational and Formative Experiences",
    "Financial and Legal Matters",
    "Health and Medical Information",
    "Self-Concept and Identity",
    "Private Thoughts and Journals",
    "Intimate and Romantic Relationships",
    "Social and Relational Information",
    "Professional and Work Life",
]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
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


def canonical_model(raw_model: str) -> str:
    return MODEL_LABELS.get(raw_model, raw_model)


def sort_model(label: str) -> tuple[int, str]:
    try:
        return MODEL_ORDER.index(label), label
    except ValueError:
        return len(MODEL_ORDER), label


def safe_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def domain_label(domain: str) -> str:
    return DOMAIN_LABELS.get(domain, domain)


def best_score(model_data: dict) -> int | None:
    scores: list[int] = []
    for generation in model_data.get("generations", []):
        if generation.get("error"):
            continue
        score = (generation.get("judge") or {}).get("score")
        if isinstance(score, int):
            scores.append(score)
    return max(scores) if scores else None


def load_domain_pairs(path: Path) -> dict[tuple[str, str], tuple[str, str]]:
    pairs: dict[tuple[str, str], tuple[str, str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            failure_type = normalize_failure_type(row.get("failure_type"))
            if failure_type != "cross_domain":
                continue
            key = (row["query"], failure_type)
            pairs[key] = (row["memory_domain"], row["query_domain"])
    return pairs


def available_domains(domain_pairs: dict[tuple[str, str], tuple[str, str]]) -> list[str]:
    seen = {domain for pair in domain_pairs.values() for domain in pair}
    return [domain for domain in DOMAIN_ORDER if domain in seen]


def collect_method_pair_scores(
    method: str,
    benchmark_path: Path,
) -> tuple[list[str], dict[str, dict[tuple[str, str], list[int]]]]:
    domain_pairs = load_domain_pairs(benchmark_path)
    scores: dict[str, dict[tuple[str, str], list[int]]] = defaultdict(lambda: defaultdict(list))

    for checkpoint_path in MEMORY_STRUCTURES[method]["checkpoints"]:
        with checkpoint_path.open(encoding="utf-8") as handle:
            checkpoint = json.load(handle)

        for entry in checkpoint.get("entries", {}).values():
            failure_type = normalize_failure_type(
                entry.get("failure_type") or entry.get("leakage_type")
            )
            if failure_type != "cross_domain":
                continue

            pair = domain_pairs.get((entry.get("query", ""), failure_type))
            if pair is None:
                continue

            for raw_model, model_data in entry.get("results", {}).items():
                score = best_score(model_data)
                if score is None:
                    continue
                scores[canonical_model(raw_model)][pair].append(score)

    return available_domains(domain_pairs), scores


def matrix_for_model(
    scores: dict[str, dict[tuple[str, str], list[int]]],
    domains: list[str],
    model: str,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.full((len(domains), len(domains)), np.nan)
    counts = np.zeros((len(domains), len(domains)), dtype=int)
    domain_index = {domain: index for index, domain in enumerate(domains)}

    for (memory_domain, query_domain), cell_scores in scores.get(model, {}).items():
        if memory_domain not in domain_index or query_domain not in domain_index:
            continue
        row = domain_index[memory_domain]
        col = domain_index[query_domain]
        counts[row, col] = len(cell_scores)
        if cell_scores:
            matrix[row, col] = float(np.mean(cell_scores))
    return matrix, counts


def average_model_matrix(
    scores: dict[str, dict[tuple[str, str], list[int]]],
    domains: list[str],
    models: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    matrices = []
    count_matrices = []
    for model in models:
        matrix, counts = matrix_for_model(scores, domains, model)
        matrices.append(matrix)
        count_matrices.append(counts)
    stacked = np.stack(matrices)
    valid = np.isfinite(stacked)
    totals = np.nansum(stacked, axis=0)
    model_counts = np.sum(valid, axis=0)
    average = np.divide(
        totals,
        model_counts,
        out=np.full_like(totals, np.nan, dtype=float),
        where=model_counts > 0,
    )
    counts = np.sum(np.stack(count_matrices), axis=0)
    return average, counts


def method_data(method: str, benchmark_path: Path) -> tuple[list[str], list[str], dict[str, tuple[np.ndarray, np.ndarray]], tuple[np.ndarray, np.ndarray]]:
    domains, scores = collect_method_pair_scores(method, benchmark_path)
    models = sorted(scores, key=sort_model)
    per_model = {
        model: matrix_for_model(scores, domains, model)
        for model in models
    }
    average = average_model_matrix(scores, domains, models)
    return domains, models, per_model, average


def add_domain_averages(
    matrix: np.ndarray,
    counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    with_average = np.full((matrix.shape[0] + 1, matrix.shape[1] + 1), np.nan)
    average_counts = np.zeros((counts.shape[0] + 1, counts.shape[1] + 1), dtype=int)

    with_average[1:, :-1] = matrix
    average_counts[1:, :-1] = counts

    row_valid = np.isfinite(matrix)
    weighted_counts = np.where(row_valid, counts, 0)
    row_totals = np.nansum(matrix * weighted_counts, axis=1)
    row_counts = np.sum(weighted_counts, axis=1)
    with_average[1:, -1] = np.divide(
        row_totals,
        row_counts,
        out=np.full(matrix.shape[0], np.nan, dtype=float),
        where=row_counts > 0,
    )
    average_counts[1:, -1] = np.sum(counts, axis=1)

    col_valid = np.isfinite(matrix)
    weighted_counts = np.where(col_valid, counts, 0)
    col_totals = np.nansum(matrix * weighted_counts, axis=0)
    col_counts = np.sum(weighted_counts, axis=0)
    with_average[0, :-1] = np.divide(
        col_totals,
        col_counts,
        out=np.full(matrix.shape[1], np.nan, dtype=float),
        where=col_counts > 0,
    )
    average_counts[0, :-1] = np.sum(counts, axis=0)

    valid = np.isfinite(matrix)
    weighted_counts = np.where(valid, counts, 0)
    overall_count = np.sum(weighted_counts)
    if overall_count > 0:
        with_average[0, -1] = np.nansum(matrix * weighted_counts) / overall_count
    average_counts[0, -1] = np.sum(counts)
    return with_average, average_counts


def reverse_query_domain_columns(
    matrix: np.ndarray,
    counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    query_order = list(range(matrix.shape[1] - 2, -1, -1)) + [matrix.shape[1] - 1]
    return matrix[:, query_order], counts[:, query_order]


def annotate_heatmap(ax, values: np.ndarray, counts: np.ndarray, *, fontsize: float) -> None:
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if counts[row, col] == 0 or not np.isfinite(value):
                continue
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=fontsize,
                fontweight="semibold",
                color="white" if value >= 3.6 else "#222222",
            )


def highlight_overall_average_cell(ax, values: np.ndarray) -> None:
    ax.add_patch(
        plt.Rectangle(
            (values.shape[1] - 1.5, -0.5),
            1,
            1,
            facecolor=OVERALL_AVERAGE_COLOR,
            edgecolor=OVERALL_AVERAGE_EDGE_COLOR,
            linewidth=1.4,
            zorder=2,
        )
    )


def style_axis(
    ax,
    domains: list[str],
    *,
    title: str,
    show_x: bool = True,
    show_y: bool = True,
    tick_size: float = 8,
    include_averages: bool = False,
) -> None:
    labels = [domain_label(domain) for domain in domains]
    x_labels = list(reversed(labels)) + ["Average"] if include_averages else list(reversed(labels))
    y_labels = ["Average"] + labels if include_averages else labels
    ax.set_title(title, fontsize=12.5, fontweight="semibold", pad=10)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels if show_x else [], rotation=35, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels if show_y else [])
    if include_averages:
        if show_x:
            ax.get_xticklabels()[-1].set_fontweight("bold")
        if show_y:
            ax.get_yticklabels()[0].set_fontweight("bold")
    ax.tick_params(axis="x", labelsize=tick_size)
    ax.tick_params(axis="y", labelsize=tick_size)
    ax.set_xticks(np.arange(-0.5, len(x_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(y_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.1)
    ax.tick_params(axis="both", which="major", length=0)
    ax.tick_params(which="minor", bottom=False, left=False)
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.85)
        ax.spines[side].set_color("#555555")


def save_single_heatmap(
    matrix: np.ndarray,
    counts: np.ndarray,
    domains: list[str],
    output_path: Path,
    *,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_matrix, plot_counts = add_domain_averages(matrix, counts)
    plot_matrix, plot_counts = reverse_query_domain_columns(plot_matrix, plot_counts)
    fig, ax = plt.subplots(figsize=(9.1, 8.0))
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#f0f0f0")
    image = ax.imshow(plot_matrix, cmap=cmap, vmin=0, vmax=5, aspect="equal")
    style_axis(ax, domains, title=title, include_averages=True)
    ax.set_xlabel("Query domain", fontweight="semibold", labelpad=8)
    ax.set_ylabel("Memory domain", fontweight="semibold", labelpad=8)
    highlight_overall_average_cell(ax, plot_matrix)
    annotate_heatmap(ax, plot_matrix, plot_counts, fontsize=7.0)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.025)
    colorbar.set_label("Average Score", fontsize=10.5, fontweight="semibold")
    colorbar.set_ticks([0, 1, 2, 3, 4, 5])
    fig.subplots_adjust(left=0.2, right=0.92, bottom=0.2, top=0.9)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_2x2(
    matrices: dict[str, tuple[list[str], np.ndarray, np.ndarray]],
    output_path: Path,
    *,
    title_suffix: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14.8, 11.6), constrained_layout=False)
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#f0f0f0")
    image = None

    for row_index, row in enumerate(GRID_METHODS):
        for col_index, method in enumerate(row):
            domains, matrix, counts = matrices[method]
            plot_matrix, plot_counts = add_domain_averages(matrix, counts)
            plot_matrix, plot_counts = reverse_query_domain_columns(plot_matrix, plot_counts)
            ax = axes[row_index, col_index]
            image = ax.imshow(plot_matrix, cmap=cmap, vmin=0, vmax=5, aspect="equal")
            style_axis(
                ax,
                domains,
                title=MEMORY_STRUCTURES[method]["label"],
                show_x=True,
                show_y=True,
                tick_size=5.8,
                include_averages=True,
            )
            ax.tick_params(axis="y", pad=1)
            ax.tick_params(axis="x", pad=1)
            highlight_overall_average_cell(ax, plot_matrix)
            annotate_heatmap(ax, plot_matrix, plot_counts, fontsize=5.1)

    colorbar_ax = fig.add_axes([0.865, 0.19, 0.018, 0.66])
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_label("Average Score", fontsize=10.5, fontweight="semibold")
    colorbar.set_ticks([0, 1, 2, 3, 4, 5])
    fig.supxlabel("Query domain", fontsize=12, fontweight="semibold", y=0.06)
    fig.supylabel("Memory domain", fontsize=12, fontweight="semibold", x=0.075)
    fig.suptitle(title_suffix, fontsize=14, fontweight="semibold", y=0.965)
    fig.subplots_adjust(left=0.12, right=0.845, bottom=0.12, top=0.91, wspace=0.06, hspace=0.18)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output-dir", type=Path, default=OUTDIR)
    args = parser.parse_args()

    by_method = {
        method: method_data(method, args.benchmark)
        for method in METHODS
    }

    models = sorted(
        {model for _, method_models, _, _ in by_method.values() for model in method_models},
        key=sort_model,
    )

    for model in models:
        model_dir = args.output_dir / safe_name(model)
        model_grid_data: dict[str, tuple[list[str], np.ndarray, np.ndarray]] = {}
        for method in METHODS:
            domains, _, per_model, _ = by_method[method]
            matrix, counts = per_model.get(
                model,
                (
                    np.full((len(domains), len(domains)), np.nan),
                    np.zeros((len(domains), len(domains)), dtype=int),
                ),
            )
            model_grid_data[method] = (domains, matrix, counts)
            save_single_heatmap(
                matrix,
                counts,
                domains,
                model_dir / f"{method}_pair_domain_score_heatmap.png",
                title=f"{MEMORY_STRUCTURES[method]['label']} - {model}",
            )
        save_2x2(
            model_grid_data,
            model_dir / "all_methods_2x2_pair_domain_score_heatmap.png",
            title_suffix=model,
        )

    average_dir = args.output_dir / "average"
    average_grid_data: dict[str, tuple[list[str], np.ndarray, np.ndarray]] = {}
    for method in METHODS:
        domains, _, _, average = by_method[method]
        matrix, counts = average
        average_grid_data[method] = (domains, matrix, counts)
        save_single_heatmap(
            matrix,
            counts,
            domains,
            average_dir / f"{method}_pair_domain_score_heatmap.png",
            title=f"{MEMORY_STRUCTURES[method]['label']} - Average Across Models",
        )
    save_2x2(
        average_grid_data,
        average_dir / "all_methods_2x2_pair_domain_score_heatmap.png",
        title_suffix="Average Across Models",
    )

    print(f"Figures saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
