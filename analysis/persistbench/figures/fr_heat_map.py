#!/usr/bin/env python3
"""Failure-rate heatmaps for PersistBench memory structures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

OUTDIR = Path(__file__).resolve().parent / "fr_heat_maps"
OUTDIR.mkdir(exist_ok=True)

MODELS = [
    "Llama 3.3-70B",
    "Qwen 3-235B",
    "DeepSeek V3.2",
    "gpt-oss 120B",
    "GLM-4.7",
    "Grok 4.1 Fast",
    "Gemini 3.1 Pro",
]

DEFENCES = [
    "Standard Prompt",
    "Permissive",
    "Restrictive",
    "Rubric Informed",
    "GEPA",
]

METRICS = [
    ("CD", "Cross-Domain Leakage"),
    ("Sycph", "Sycophancy"),
    ("Benef", "Beneficial"),
]

MEMORY_STRUCTURES = [
    ("Flat Memory List", "Flat Memory List"),
    ("Partitions", "Fixed Partitions"),
    ("Dynamic Partitions", "Dynamic Partitions"),
]

# Values are failure rates from the provided table.
# Shape: method -> defence -> model -> metric.
DATA = {
    "Flat Memory List": {
        "Standard Prompt": {
            "Llama 3.3-70B": {"CD": 13.0, "Sycph": 81.0, "Benef": 59.0},
            "Qwen 3-235B": {"CD": 80.0, "Sycph": 99.5, "Benef": 22.0},
            "DeepSeek V3.2": {"CD": 69.0, "Sycph": 99.5, "Benef": 21.0},
            "gpt-oss 120B": {"CD": 51.0, "Sycph": 97.0, "Benef": 21.0},
            "GLM-4.7": {"CD": 58.5, "Sycph": 100.0, "Benef": 19.0},
            "Grok 4.1 Fast": {"CD": 57.5, "Sycph": 99.0, "Benef": 21.0},
            "Gemini 3.1 Pro": {"CD": 64.5, "Sycph": 99.5, "Benef": 0.0},
        },
        "Permissive": {
            "Llama 3.3-70B": {"CD": 37.5, "Sycph": 93.0, "Benef": 53.0},
            "Qwen 3-235B": {"CD": 84.0, "Sycph": 99.5, "Benef": 20.0},
            "DeepSeek V3.2": {"CD": 85.0, "Sycph": 100.0, "Benef": 15.0},
            "gpt-oss 120B": {"CD": 80.0, "Sycph": 100.0, "Benef": 14.0},
            "GLM-4.7": {"CD": 73.5, "Sycph": 100.0, "Benef": 17.0},
            "Grok 4.1 Fast": {"CD": 70.5, "Sycph": 100.0, "Benef": 11.0},
            "Gemini 3.1 Pro": {"CD": 90.0, "Sycph": 100.0, "Benef": 1.0},
        },
        "Restrictive": {
            "Llama 3.3-70B": {"CD": 8.5, "Sycph": 66.5, "Benef": 64.0},
            "Qwen 3-235B": {"CD": 65.5, "Sycph": 98.0, "Benef": 20.0},
            "DeepSeek V3.2": {"CD": 46.0, "Sycph": 98.0, "Benef": 23.0},
            "gpt-oss 120B": {"CD": 22.0, "Sycph": 94.0, "Benef": 24.0},
            "GLM-4.7": {"CD": 16.5, "Sycph": 92.5, "Benef": 22.0},
            "Grok 4.1 Fast": {"CD": 34.5, "Sycph": 95.5, "Benef": 21.0},
            "Gemini 3.1 Pro": {"CD": 3.5, "Sycph": 47.5, "Benef": 47.0},
        },
        "Rubric Informed": {
            "Llama 3.3-70B": {"CD": 15.0, "Sycph": 72.5, "Benef": 61.0},
            "Qwen 3-235B": {"CD": 64.5, "Sycph": 98.5, "Benef": 16.0},
            "DeepSeek V3.2": {"CD": 58.5, "Sycph": 99.5, "Benef": 23.0},
            "gpt-oss 120B": {"CD": 46.0, "Sycph": 97.5, "Benef": 20.0},
            "GLM-4.7": {"CD": 41.0, "Sycph": 98.5, "Benef": 16.0},
            "Grok 4.1 Fast": {"CD": 39.5, "Sycph": 97.5, "Benef": 18.0},
            "Gemini 3.1 Pro": {"CD": 33.5, "Sycph": 73.5, "Benef": 0.0},
        },
        "GEPA": {
            "Llama 3.3-70B": {"CD": 22.5, "Sycph": 78.5, "Benef": 55.0},
            "Qwen 3-235B": {"CD": 77.0, "Sycph": 98.5, "Benef": 13.0},
            "DeepSeek V3.2": {"CD": 73.0, "Sycph": 99.0, "Benef": 15.0},
            "gpt-oss 120B": {"CD": 64.0, "Sycph": 95.0, "Benef": 18.0},
            "GLM-4.7": {"CD": 55.5, "Sycph": 96.5, "Benef": 14.0},
            "Grok 4.1 Fast": {"CD": 48.0, "Sycph": 96.5, "Benef": 20.0},
            "Gemini 3.1 Pro": {"CD": 53.5, "Sycph": 79.0, "Benef": 1.0},
        },
    },
    "Partitions": {
        "Standard Prompt": {
            "Llama 3.3-70B": {"CD": 9.0, "Sycph": 78.0, "Benef": 61.0},
            "Qwen 3-235B": {"CD": 71.0, "Sycph": 99.5, "Benef": 17.0},
            "DeepSeek V3.2": {"CD": 60.5, "Sycph": 98.0, "Benef": 35.0},
            "gpt-oss 120B": {"CD": 44.5, "Sycph": 95.5, "Benef": 19.0},
            "GLM-4.7": {"CD": 47.5, "Sycph": 99.5, "Benef": 15.0},
            "Grok 4.1 Fast": {"CD": 49.0, "Sycph": 97.5, "Benef": 24.0},
            "Gemini 3.1 Pro": {"CD": 70.5, "Sycph": 100.0, "Benef": 0.0},
        },
        "Permissive": {
            "Llama 3.3-70B": {"CD": 36.0, "Sycph": 86.0, "Benef": 56.0},
            "Qwen 3-235B": {"CD": 83.0, "Sycph": 100.0, "Benef": 12.0},
            "DeepSeek V3.2": {"CD": 87.0, "Sycph": 100.0, "Benef": 20.0},
            "gpt-oss 120B": {"CD": 74.5, "Sycph": 99.5, "Benef": 17.0},
            "GLM-4.7": {"CD": 75.5, "Sycph": 100.0, "Benef": 22.0},
            "Grok 4.1 Fast": {"CD": 71.9, "Sycph": 99.5, "Benef": 11.0},
            "Gemini 3.1 Pro": {"CD": 90.5, "Sycph": 100.0, "Benef": 1.0},
        },
        "Restrictive": {
            "Llama 3.3-70B": {"CD": 3.0, "Sycph": 68.0, "Benef": 68.0},
            "Qwen 3-235B": {"CD": 58.5, "Sycph": 98.0, "Benef": 18.0},
            "DeepSeek V3.2": {"CD": 40.0, "Sycph": 97.0, "Benef": 30.0},
            "gpt-oss 120B": {"CD": 18.5, "Sycph": 86.5, "Benef": 22.0},
            "GLM-4.7": {"CD": 19.5, "Sycph": 92.5, "Benef": 24.0},
            "Grok 4.1 Fast": {"CD": 31.7, "Sycph": 96.0, "Benef": 20.0},
            "Gemini 3.1 Pro": {"CD": 5.5, "Sycph": 53.0, "Benef": 36.0},
        },
        "Rubric Informed": {
            "Llama 3.3-70B": {"CD": 9.5, "Sycph": 66.0, "Benef": 51.0},
            "Qwen 3-235B": {"CD": 65.5, "Sycph": 98.0, "Benef": 18.0},
            "DeepSeek V3.2": {"CD": 63.0, "Sycph": 99.0, "Benef": 26.0},
            "gpt-oss 120B": {"CD": 43.0, "Sycph": 94.5, "Benef": 22.0},
            "GLM-4.7": {"CD": 40.5, "Sycph": 95.0, "Benef": 20.0},
            "Grok 4.1 Fast": {"CD": 39.7, "Sycph": 95.0, "Benef": 13.0},
            "Gemini 3.1 Pro": {"CD": 37.5, "Sycph": 82.0, "Benef": 0.0},
        },
        "GEPA": {
            "Llama 3.3-70B": {"CD": 22.0, "Sycph": 73.0, "Benef": 57.0},
            "Qwen 3-235B": {"CD": 73.5, "Sycph": 99.0, "Benef": 13.0},
            "DeepSeek V3.2": {"CD": 70.5, "Sycph": 98.0, "Benef": 13.0},
            "gpt-oss 120B": {"CD": 63.5, "Sycph": 93.0, "Benef": 14.0},
            "GLM-4.7": {"CD": 50.0, "Sycph": 95.0, "Benef": 15.0},
            "Grok 4.1 Fast": {"CD": 48.7, "Sycph": 97.0, "Benef": 11.0},
            "Gemini 3.1 Pro": {"CD": 46.0, "Sycph": 78.5, "Benef": 0.0},
        },
    },
    "Dynamic Partitions": {
        "Standard Prompt": {
            "Llama 3.3-70B": {"CD": 10.5, "Sycph": 76.0, "Benef": 56.0},
            "Qwen 3-235B": {"CD": 66.5, "Sycph": 99.0, "Benef": 22.0},
            "DeepSeek V3.2": {"CD": 55.0, "Sycph": 98.5, "Benef": 16.0},
            "gpt-oss 120B": {"CD": 41.0, "Sycph": 94.0, "Benef": 16.0},
            "GLM-4.7": {"CD": 48.0, "Sycph": 99.5, "Benef": 14.0},
            "Grok 4.1 Fast": {"CD": 51.5, "Sycph": 98.0, "Benef": 19.0},
            "Gemini 3.1 Pro": {"CD": 59.5, "Sycph": 100.0, "Benef": 2.0},
        },
        "Permissive": {
            "Llama 3.3-70B": {"CD": 37.0, "Sycph": 86.0, "Benef": 58.0},
            "Qwen 3-235B": {"CD": 82.0, "Sycph": 100.0, "Benef": 13.0},
            "DeepSeek V3.2": {"CD": 82.0, "Sycph": 100.0, "Benef": 15.0},
            "gpt-oss 120B": {"CD": 72.5, "Sycph": 99.5, "Benef": 11.0},
            "GLM-4.7": {"CD": 73.5, "Sycph": 100.0, "Benef": 18.0},
            "Grok 4.1 Fast": {"CD": 68.3, "Sycph": 100.0, "Benef": 18.0},
            "Gemini 3.1 Pro": {"CD": 90.5, "Sycph": 100.0, "Benef": 2.0},
        },
        "Restrictive": {
            "Llama 3.3-70B": {"CD": 10.0, "Sycph": 69.0, "Benef": 69.0},
            "Qwen 3-235B": {"CD": 62.0, "Sycph": 97.0, "Benef": 24.0},
            "DeepSeek V3.2": {"CD": 41.0, "Sycph": 98.5, "Benef": 23.0},
            "gpt-oss 120B": {"CD": 19.0, "Sycph": 88.5, "Benef": 24.0},
            "GLM-4.7": {"CD": 9.5, "Sycph": 91.5, "Benef": 21.0},
            "Grok 4.1 Fast": {"CD": 30.7, "Sycph": 97.0, "Benef": 21.0},
            "Gemini 3.1 Pro": {"CD": 5.5, "Sycph": 53.5, "Benef": 33.0},
        },
        "Rubric Informed": {
            "Llama 3.3-70B": {"CD": 8.5, "Sycph": 68.0, "Benef": 65.0},
            "Qwen 3-235B": {"CD": 61.5, "Sycph": 98.0, "Benef": 13.0},
            "DeepSeek V3.2": {"CD": 61.0, "Sycph": 98.0, "Benef": 22.0},
            "gpt-oss 120B": {"CD": 44.5, "Sycph": 96.0, "Benef": 14.0},
            "GLM-4.7": {"CD": 38.0, "Sycph": 97.5, "Benef": 19.0},
            "Grok 4.1 Fast": {"CD": 41.2, "Sycph": 95.5, "Benef": 13.0},
            "Gemini 3.1 Pro": {"CD": 37.5, "Sycph": 78.4, "Benef": 0.0},
        },
        "GEPA": {
            "Llama 3.3-70B": {"CD": 20.5, "Sycph": 76.0, "Benef": 52.0},
            "Qwen 3-235B": {"CD": 73.5, "Sycph": 98.0, "Benef": 16.0},
            "DeepSeek V3.2": {"CD": 69.5, "Sycph": 100.0, "Benef": 16.0},
            "gpt-oss 120B": {"CD": 55.5, "Sycph": 95.0, "Benef": 13.0},
            "GLM-4.7": {"CD": 57.5, "Sycph": 97.0, "Benef": 11.0},
            "Grok 4.1 Fast": {"CD": 43.2, "Sycph": 96.5, "Benef": 10.0},
            "Gemini 3.1 Pro": {"CD": 47.5, "Sycph": 77.5, "Benef": 2.0},
        },
    },
}


def filename_safe(name: str) -> str:
    return name.lower().replace(" ", "_").replace(".", "").replace("-", "_")


def metric_matrix(memory_structure: str, metric: str) -> np.ndarray:
    return np.array([
        [
            DATA[memory_structure][defence][model][metric]
            for defence in DEFENCES
        ]
        for model in MODELS
    ])


def format_cell(value: float) -> str:
    if abs(value - round(value)) < 0.05:
        return f"{value:.0f}"
    return f"{value:.1f}"


def style_heatmap_axis(ax, title: str) -> None:
    ax.set_title(title, fontsize=13, fontweight="semibold", pad=10, color="#222222")
    ax.set_xticks(range(len(DEFENCES)))
    ax.set_xticklabels(
        DEFENCES,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODELS)
    ax.tick_params(axis="both", which="major", length=0)
    ax.set_xticks(np.arange(-0.5, len(DEFENCES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(MODELS), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)

    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.85)
        ax.spines[side].set_color("#555555")


def annotate_heatmap(ax, matrix: np.ndarray) -> None:
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            color = "white" if value >= 62 else "#222222"
            ax.text(
                col,
                row,
                format_cell(value),
                ha="center",
                va="center",
                fontsize=9.3,
                fontweight="semibold",
                color=color,
            )


def draw_metric(metric_key: str, metric_label: str) -> None:
    fig = plt.figure(figsize=(14.2, 5.9))
    gridspec = fig.add_gridspec(
        nrows=1,
        ncols=6,
        width_ratios=[1, 0.055, 1, 0.055, 1, 0.055],
        wspace=0.12,
    )

    cmap = plt.get_cmap("RdYlGn_r").copy()
    image = None

    for index, (memory_structure, display_label) in enumerate(MEMORY_STRUCTURES):
        ax = fig.add_subplot(gridspec[0, index * 2])
        matrix = metric_matrix(memory_structure, metric_key)
        image = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=100, aspect="equal")

        style_heatmap_axis(ax, display_label)
        annotate_heatmap(ax, matrix)

        if index != 0:
            ax.set_yticklabels([])

        if index < len(MEMORY_STRUCTURES) - 1:
            arrow_ax = fig.add_subplot(gridspec[0, index * 2 + 1])
            arrow_ax.axis("off")
            arrow_ax.text(
                0.5,
                0.52,
                "→",
                ha="center",
                va="center",
                fontsize=26,
                fontweight="semibold",
                color="#444444",
                transform=arrow_ax.transAxes,
            )

    cax = fig.add_subplot(gridspec[0, 5])
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_label("Failure rate (%)", fontsize=11, fontweight="semibold")
    colorbar.set_ticks([0, 20, 40, 60, 80, 100])
    colorbar.ax.tick_params(labelsize=10, length=3)

    fig.suptitle(
        metric_label,
        fontsize=16,
        fontweight="semibold",
        y=0.98,
        color="#222222",
    )
    fig.subplots_adjust(left=0.095, right=0.94, bottom=0.19, top=0.86)

    stem = f"{filename_safe(metric_label)}_failure_rate_heat_map"
    fig.savefig(OUTDIR / f"{stem}.png", bbox_inches="tight")
    fig.savefig(OUTDIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    for metric_key, metric_label in METRICS:
        draw_metric(metric_key, metric_label)

    print(f"Figures saved to: {OUTDIR.resolve()}")


if __name__ == "__main__":
    main()
