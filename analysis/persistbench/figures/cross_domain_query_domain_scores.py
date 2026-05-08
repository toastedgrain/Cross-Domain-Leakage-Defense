#!/usr/bin/env python3
"""Cross-domain PersistBench scores by model and query domain.

For each cross-domain sample, this takes the highest judge score across that
model's generations, then aggregates those per-sample scores by query domain.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA_TEXT_SIZE = 20
AXIS_TEXT_SIZE = 24
AXIS_NAME_TEXT_SIZE = 24
TITLE_TEXT_SIZE = 26
SCALE_TEXT_SIZE = 24
# Suggested data text colors:
# - "white" for dark heat-map cells and high contrast on orange/red palettes.
# - "#222222" for light heat-map cells.
# - "#111111" for maximum dark text contrast.
DATA_TEXT_COLOR = "#353535"

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUTS = REPO_ROOT / "outputs" / "persistbench"
DEFAULT_BASELINE_DIR = OUTPUTS / "baseline"
DEFAULT_ALL_CONFIGS_BASELINE = OUTPUTS / "all_configs" / "baseline"
DEFAULT_ALL_CONFIGS_PARTITIONED = (
    OUTPUTS
    / "all_configs"
    / "partitioned"
    / "output_all_models_partitioned.json"
)
DEFAULT_PARTITIONED_DIR = OUTPUTS / "partitioned"
DEFAULT_ALL_CONFIGS_PARTITIONED_COS = OUTPUTS / "all_configs" / "partitioned_cos"
DEFAULT_PARTITIONED_COS_DIR = OUTPUTS / "partitioned_cos_similarity"
DEFAULT_ALL_CONFIGS_PARTITIONED_CUSTOM = OUTPUTS / "all_configs" / "partitioned_custom"
DEFAULT_PARTITIONED_CUSTOM_DIR = OUTPUTS / "partitioned_custom_categories"
DEFAULT_ALL_CONFIGS_TREE = OUTPUTS / "all_configs" / "tree"
DEFAULT_TREE_DIR = OUTPUTS / "tree"
DEFAULT_BENCHMARK = (
    REPO_ROOT
    / "benchmark_samples"
    / "persistbench"
    / "baseline"
    / "full_benchmark.jsonl"
)
OUTDIR = Path(__file__).resolve().parent / "cross_domain_query_domain_scores"

MEMORY_STRUCTURES = {
    "flat_memory_list": {
        "label": "Flat List",
        "checkpoints": [
            DEFAULT_ALL_CONFIGS_BASELINE / "output_all_models_baseline.json",
            DEFAULT_BASELINE_DIR / "persist_full_gemini3_pro.json",
            DEFAULT_BASELINE_DIR / "persist_full_llama3p3_70b_qwen3_235b.json",
        ],
    },
    "partitions": {
        "label": "Inference Fixed Partitions",
        "checkpoints": [
            DEFAULT_ALL_CONFIGS_PARTITIONED,
            DEFAULT_PARTITIONED_DIR / "persist_partitioned_gemini3_pro.json",
            DEFAULT_PARTITIONED_DIR
            / "with_empty_categories"
            / "cross_domain_partitioned_with_empty_categories_llama3p3_70b_qwen3_235b.json",
        ],
    },
    "cosine_similarity_partitions": {
        "label": "Cosine Similarity Partitions",
        "checkpoints": [
            DEFAULT_ALL_CONFIGS_PARTITIONED_COS / "output_all_models_partitioned_cos.json",
            DEFAULT_PARTITIONED_COS_DIR / "cross_domain_llama3p3_70b_qwen3_235b_cos_partitioned_with_empty.json",
        ],
    },
    "dynamic_partitions": {
        "label": "Inference Dynamic Partitions",
        "checkpoints": [
            DEFAULT_ALL_CONFIGS_PARTITIONED_CUSTOM / "output_all_models_partitioned_custom.json",
            DEFAULT_PARTITIONED_CUSTOM_DIR / "persist_partitioned_custom_categories_gemini3_pro.json",
            DEFAULT_PARTITIONED_CUSTOM_DIR
            / "with_empty_categories"
            / "cross_domain_partitioned_model_custom_with_empty_categories_llama3p3_70b_qwen3_235b.json",
        ],
    },
    "2_level_tree": {
        "label": "2-Level Tree",
        "checkpoints": [
            DEFAULT_ALL_CONFIGS_TREE / "output_all_models_tree_informed.json",
            DEFAULT_TREE_DIR / "output_persistbench_informed_tree_llama3p3_qwen3.json",
            DEFAULT_TREE_DIR / "persist_informed_tree_gemini3_pro.json",
        ],
    },
}

TRANSITION_STRUCTURES = [
    "flat_memory_list",
    "partitions",
    "dynamic_partitions",
]

COSINE_COMPARISON_STRUCTURES = [
    "partitions",
    "cosine_similarity_partitions",
]

GRID_STRUCTURES = [
    ["flat_memory_list", "partitions"],
    ["dynamic_partitions", "2_level_tree"],
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

CHECKPOINT_LABELS = {
    "output_all_models_baseline.json": "Flat Memory List",
    "output_all_models_partitioned.json": "Inference Fixed Partitions",
    "output_all_models_partitioned_cos.json": "Cosine Similarity Partitions",
    "output_all_models_partitioned_custom.json": "Inference Dynamic Partitions",
    "output_all_models_tree_informed.json": "2-Level Tree",
    "persist_full_gemini3_pro.json": "Flat Memory List",
    "persist_full_llama3p3_70b_qwen3_235b.json": "Flat Memory List",
    "persist_partitioned_gemini3_pro.json": "Inference Fixed Partitions",
    "cross_domain_llama3p3_70b_qwen3_235b_cos_partitioned_with_empty.json": (
        "Cosine Similarity Partitions + empty"
    ),
    "persist_partitioned_custom_categories_gemini3_pro.json": "Inference Dynamic Partitions",
    "cross_domain_partitioned_model_custom_with_empty_categories_llama3p3_70b_qwen3_235b.json": (
        "Inference Dynamic Partitions + empty"
    ),
    "output_persistbench_informed_tree_llama3p3_qwen3.json": "2-Level Tree",
    "persist_informed_tree_gemini3_pro.json": "2-Level Tree",
    "cross_domain_partitioned_with_empty_categories_llama3p3_70b_qwen3_235b.json": (
        "Inference Fixed Partitions + empty"
    ),
    "cross_domain_partition_informed_with_empty_categories_llama3p3_70b_qwen3_235b.json": (
        "Inference Fixed Partitions informed + empty"
    ),
}

DOMAIN_LABELS = {
    "Educational and Formative Experiences": "Education",
    "Financial and Legal Matters": "Finance/Legal",
    "Health and Medical Information": "Health",
    "Intimate and Romantic Relationships": "Romantic",
    "Personal Beliefs (Political, Religious, and Social)": "Beliefs",
    "Private Thoughts and Journals": "Journals",
    "Professional and Work Life": "Work",
    "Self-Concept and Identity": "Identity",
    "Social and Relational Information": "Social",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": AXIS_TEXT_SIZE,
        "axes.titlesize": TITLE_TEXT_SIZE,
        "axes.labelsize": AXIS_NAME_TEXT_SIZE,
        "xtick.labelsize": AXIS_TEXT_SIZE,
        "ytick.labelsize": AXIS_TEXT_SIZE,
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


def load_benchmark_domains(path: Path) -> dict[tuple[str, str], str]:
    domains: dict[tuple[str, str], str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (row["query"], normalize_failure_type(row.get("failure_type")))
            domains[key] = row.get("query_domain", "")
    return domains


def best_score(model_data: dict) -> int | None:
    scores: list[int] = []
    for generation in model_data.get("generations", []):
        if generation.get("error"):
            continue
        judge = generation.get("judge") or {}
        score = judge.get("score")
        if isinstance(score, int):
            scores.append(score)
    return max(scores) if scores else None


def model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def sort_model_row(label: str) -> tuple[int, str]:
    base_label = label.split(" (", 1)[0]
    try:
        return MODEL_ORDER.index(base_label), label
    except ValueError:
        return len(MODEL_ORDER), label


def aggregate_scores(
    checkpoint_paths: list[Path], benchmark_path: Path
) -> tuple[list[str], list[str], dict[str, dict[str, list[int]]]]:
    benchmark_domains = load_benchmark_domains(benchmark_path)

    scores: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    domains_seen: set[str] = set()
    rows_seen: set[str] = set()
    missing_domains: list[str] = []
    checkpoint_data: list[tuple[Path, dict]] = []

    for checkpoint_path in checkpoint_paths:
        with checkpoint_path.open(encoding="utf-8") as handle:
            checkpoint = json.load(handle)
        checkpoint_data.append((checkpoint_path, checkpoint))

    for checkpoint_path, checkpoint in checkpoint_data:
        for entry_id, entry in checkpoint.get("entries", {}).items():
            failure_type = normalize_failure_type(
                entry.get("failure_type") or entry.get("leakage_type")
            )
            if failure_type != "cross_domain":
                continue

            domain = benchmark_domains.get((entry.get("query", ""), failure_type))
            if not domain:
                missing_domains.append(f"{checkpoint_path.name}:{entry_id}")
                continue

            domains_seen.add(domain)
            for model, model_data in entry.get("results", {}).items():
                score = best_score(model_data)
                if score is None:
                    continue
                row = model_label(model)
                rows_seen.add(row)
                scores[row][domain].append(score)

    if missing_domains:
        raise RuntimeError(
            f"Could not find query domains for {len(missing_domains)} checkpoint entries"
        )

    models = sorted(rows_seen, key=sort_model_row)
    domains = sorted(domains_seen, key=lambda domain: DOMAIN_LABELS.get(domain, domain).lower())
    return models, domains, scores


def build_matrices(
    checkpoint_paths: list[Path],
    benchmark_path: Path,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    models, domains, scores = aggregate_scores(checkpoint_paths, benchmark_path)
    averages = np.full((len(models), len(domains)), np.nan)
    counts = np.zeros((len(models), len(domains)), dtype=int)

    for row, model in enumerate(models):
        for col, domain in enumerate(domains):
            cell_scores = scores[model].get(domain, [])
            counts[row, col] = len(cell_scores)
            if cell_scores:
                averages[row, col] = float(np.mean(cell_scores))

    if len(models) > 0:
        average_row = np.nanmean(averages, axis=0, keepdims=True)
        count_row = np.sum(counts, axis=0, keepdims=True)
        models = [*models, "Average"]
        averages = np.vstack([averages, average_row])
        counts = np.vstack([counts, count_row])

    return models, domains, averages, counts


def annotate_heatmap(
    ax,
    averages: np.ndarray,
    counts: np.ndarray,
    *,
    fontsize: float,
) -> None:
    for row in range(averages.shape[0]):
        for col in range(averages.shape[1]):
            if counts[row, col] == 0:
                label = ""
            else:
                label = f"{averages[row, col]:.2f}"
            ax.text(
                col,
                row,
                label,
                ha="center",
                va="center",
                fontsize=fontsize,
                fontweight="bold",
                color=DATA_TEXT_COLOR,
            )


def style_heatmap_axis(
    ax,
    models: list[str],
    domains: list[str],
    *,
    title: str,
    show_y_labels: bool,
    show_x_labels: bool,
    title_size: float,
    x_tick_size: float | None = None,
    y_tick_size: float | None = None,
) -> None:
    ax.set_title(
        f"{title}",
        fontsize=title_size,
        fontweight="semibold",
        pad=10,
        color="#222222",
    )
    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels(
        [DOMAIN_LABELS.get(domain, domain) for domain in domains] if show_x_labels else [],
        rotation=35,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models if show_y_labels else [])
    if show_y_labels:
        for tick in ax.get_yticklabels():
            if tick.get_text() == "Average":
                tick.set_fontweight("semibold")
    if x_tick_size is not None:
        ax.tick_params(axis="x", labelsize=x_tick_size)
    if y_tick_size is not None:
        ax.tick_params(axis="y", labelsize=y_tick_size)

    ax.set_xticks(np.arange(-0.5, len(domains), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.4)
    ax.tick_params(axis="both", which="major", length=0)
    ax.tick_params(which="minor", bottom=False, left=False)

    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.85)
        ax.spines[side].set_color("#555555")


def draw_heatmap(
    checkpoint_paths: list[Path],
    benchmark_path: Path,
    output_dir: Path,
    *,
    title: str,
    output_stem: str,
) -> None:
    models, domains, averages, counts = build_matrices(checkpoint_paths, benchmark_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    fig_width = max(12.4, 1.18 * len(domains) + 2.8)
    fig_height = max(9.6, 1.18 * len(models) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#f0f0f0")
    image = ax.imshow(averages, cmap=cmap, vmin=1, vmax=5, aspect="equal")
    style_heatmap_axis(
        ax,
        models,
        domains,
        title=title,
        show_y_labels=True,
        show_x_labels=True,
        title_size=TITLE_TEXT_SIZE,
    )
    ax.set_xlabel("Sample query domain", fontsize=AXIS_NAME_TEXT_SIZE, fontweight="semibold", labelpad=10)
    annotate_heatmap(ax, averages, counts, fontsize=DATA_TEXT_SIZE)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    colorbar.set_label("Average Score", fontsize=SCALE_TEXT_SIZE, fontweight="semibold")
    colorbar.set_ticks([1, 2, 3, 4, 5])
    colorbar.ax.tick_params(labelsize=SCALE_TEXT_SIZE, length=3)
    
    fig.subplots_adjust(left=0.20, right=0.9, bottom=0.29, top=0.84)

    fig.savefig(output_dir / f"{output_stem}.png", bbox_inches="tight")
    # fig.savefig(output_dir / f"{output_stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def draw_transition_figure(benchmark_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    figure_data = []
    for structure in TRANSITION_STRUCTURES:
        config = MEMORY_STRUCTURES[structure]
        models, domains, averages, counts = build_matrices(
            config["checkpoints"],
            benchmark_path,
        )
        figure_data.append((config["label"], models, domains, averages, counts))

    fig = plt.figure(figsize=(29.0, 10.2))
    gridspec = fig.add_gridspec(
        nrows=1,
        ncols=6,
        width_ratios=[1.0, 0.08, 1.0, 0.08, 1.0, 0.05],
        wspace=0.1,
    )
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#f0f0f0")
    image = None

    for index, (title, models, domains, averages, counts) in enumerate(figure_data):
        ax = fig.add_subplot(gridspec[0, index * 2])
        image = ax.imshow(averages, cmap=cmap, vmin=1, vmax=5, aspect="auto")
        style_heatmap_axis(
            ax,
            models,
            domains,
            title=title,
            show_y_labels=(index == 0),
            show_x_labels=True,
            title_size=TITLE_TEXT_SIZE,
            x_tick_size=AXIS_TEXT_SIZE,
            y_tick_size=AXIS_TEXT_SIZE,
        )
        annotate_heatmap(ax, averages, counts, fontsize=DATA_TEXT_SIZE)

        if index < len(figure_data) - 1:
            arrow_ax = fig.add_subplot(gridspec[0, index * 2 + 1])
            arrow_ax.axis("off")
            arrow_ax.text(
                0.5,
                0.53,
                "→",
                ha="center",
                va="center",
                fontsize=TITLE_TEXT_SIZE,
                fontweight="semibold",
                color="#333333",
                transform=arrow_ax.transAxes,
            )

    cax = fig.add_subplot(gridspec[0, 5])
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_label("Average Score", fontsize=SCALE_TEXT_SIZE, fontweight="semibold")
    colorbar.set_ticks([1, 2, 3, 4, 5])
    colorbar.ax.tick_params(labelsize=SCALE_TEXT_SIZE, length=3)

    fig.subplots_adjust(left=0.075, right=0.95, bottom=0.34, top=0.78)

    output_stem = "flat_to_dynamic_transition_cross_domain_best_score_by_query_domain"
    fig.savefig(output_dir / f"{output_stem}.png", bbox_inches="tight")
    # fig.savefig(output_dir / f"{output_stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def draw_four_method_grid(benchmark_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(27.0, 20.5),
        constrained_layout=False,
    )
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#f0f0f0")
    image = None

    for row_index, row in enumerate(GRID_STRUCTURES):
        for col_index, structure in enumerate(row):
            config = MEMORY_STRUCTURES[structure]
            models, domains, averages, counts = build_matrices(
                config["checkpoints"],
                benchmark_path,
            )
            ax = axes[row_index, col_index]
            image = ax.imshow(averages, cmap=cmap, vmin=0, vmax=5, aspect="auto")
            style_heatmap_axis(
                ax,
                models,
                domains,
                title=config["label"],
                show_y_labels=(col_index == 0),
                show_x_labels=True,
                title_size=TITLE_TEXT_SIZE,
                x_tick_size=AXIS_TEXT_SIZE,
                y_tick_size=AXIS_TEXT_SIZE,
            )
            annotate_heatmap(ax, averages, counts, fontsize=DATA_TEXT_SIZE)

    colorbar = fig.colorbar(
        image,
        ax=axes.ravel().tolist(),
        fraction=0.025,
        pad=0.045,
    )
    colorbar.set_label("Average Score", fontsize=SCALE_TEXT_SIZE, fontweight="semibold")
    colorbar.set_ticks([0, 1, 2, 3, 4, 5])
    colorbar.ax.tick_params(labelsize=SCALE_TEXT_SIZE, length=3)

    fig.supxlabel("Sample query domain", fontsize=AXIS_NAME_TEXT_SIZE, fontweight="semibold", y=0.035)
    fig.subplots_adjust(left=0.10, right=0.86, bottom=0.18, top=0.88, wspace=0.12, hspace=0.36)

    output_stem = "all_methods_2x2_cross_domain_best_score_by_query_domain"
    fig.savefig(output_dir / f"{output_stem}.png", bbox_inches="tight")
    # fig.savefig(output_dir / f"{output_stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def draw_cosine_vs_fixed_figure(benchmark_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(25.0, 10.2),
        constrained_layout=False,
    )
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#f0f0f0")
    image = None

    for index, structure in enumerate(COSINE_COMPARISON_STRUCTURES):
        config = MEMORY_STRUCTURES[structure]
        models, domains, averages, counts = build_matrices(
            config["checkpoints"],
            benchmark_path,
        )
        ax = axes[index]
        image = ax.imshow(averages, cmap=cmap, vmin=0, vmax=5, aspect="auto")
        style_heatmap_axis(
            ax,
            models,
            domains,
            title=config["label"],
            show_y_labels=(index == 0),
            show_x_labels=True,
            title_size=TITLE_TEXT_SIZE,
            x_tick_size=AXIS_TEXT_SIZE,
            y_tick_size=AXIS_TEXT_SIZE,
        )
        annotate_heatmap(ax, averages, counts, fontsize=DATA_TEXT_SIZE)

    colorbar = fig.colorbar(
        image,
        ax=axes.ravel().tolist(),
        fraction=0.03,
        pad=0.045,
    )
    colorbar.set_label("Average Score", fontsize=SCALE_TEXT_SIZE, fontweight="semibold")
    colorbar.set_ticks([0, 1, 2, 3, 4, 5])
    colorbar.ax.tick_params(labelsize=SCALE_TEXT_SIZE, length=3)

    fig.supxlabel("Sample query domain", fontsize=AXIS_NAME_TEXT_SIZE, fontweight="semibold", y=0.04)
    fig.subplots_adjust(left=0.10, right=0.86, bottom=0.30, top=0.82, wspace=0.12)

    output_stem = "cosine_vs_inference_fixed_cross_domain_best_score_by_query_domain"
    fig.savefig(output_dir / f"{output_stem}.png", bbox_inches="tight")
    # fig.savefig(output_dir / f"{output_stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def draw_memory_structure(name: str, benchmark_path: Path, output_dir: Path) -> None:
    if name not in MEMORY_STRUCTURES:
        valid = ", ".join(MEMORY_STRUCTURES)
        raise ValueError(f"Unknown memory structure {name!r}. Valid values: {valid}")
    config = MEMORY_STRUCTURES[name]
    draw_heatmap(
        config["checkpoints"],
        benchmark_path,
        output_dir,
        title=config["label"],
        output_stem=f"{name}_cross_domain_best_score_by_query_domain",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        default=None,
        help=(
            "Checkpoint JSON to include. Repeat to include multiple files. "
            "When provided, generates one custom plot instead of the default structure plots."
        ),
    )
    parser.add_argument(
        "--structure",
        action="append",
        choices=sorted(MEMORY_STRUCTURES),
        help="Memory structure plot to generate. Repeat for multiple. Default: all structures.",
    )
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output-dir", type=Path, default=OUTDIR)
    args = parser.parse_args()

    if args.checkpoint:
        draw_heatmap(
            args.checkpoint,
            args.benchmark,
            args.output_dir,
            title="Custom Checkpoints",
            output_stem="custom_cross_domain_best_score_by_query_domain",
        )
    else:
        structures = args.structure or list(MEMORY_STRUCTURES)
        for structure in structures:
            draw_memory_structure(structure, args.benchmark, args.output_dir)
        if args.structure is None:
            draw_transition_figure(args.benchmark, args.output_dir)
            draw_four_method_grid(args.benchmark, args.output_dir)
            draw_cosine_vs_fixed_figure(args.benchmark, args.output_dir)
    print(f"Figure saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
