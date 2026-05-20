#!/usr/bin/env python3
"""Per-model memory distribution heatmap for PersistBench partition data."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA_TEXT_SIZE = 18
AXIS_TEXT_SIZE = 22
AXIS_NAME_TEXT_SIZE = 22
TITLE_TEXT_SIZE = 24
SCALE_TEXT_SIZE = 18

plt.rcParams.update({
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
})

OUTDIR = Path(__file__).resolve().parent / "category_fill_heat_maps"
OUTDIR.mkdir(exist_ok=True)

CATEGORIES = [
    "personal",
    "education",
    "employment",
    "finance",
    "housing",
    "legal",
    "health",
    "schedule",
    "identity",
    "social",
    "romantic",
]

MODELS = [
    "Cosine Similarity",
    "Llama 3.3 70B",
    "Qwen 3 235B",
    "DeepSeek V3.2",
    "GPT OSS 120B",
    "GLM-4.7",
    "Grok 4.1 Fast",
    "Gemini 3.1 Pro"
]

# Values are total memory items in each category.
DATA = {
    "Cosine Similarity": {
        "personal": 2539,
        "education": 155,
        "employment": 478,
        "finance": 72,
        "housing": 310,
        "legal": 108,
        "health": 184,
        "schedule": 576,
        "identity": 59,
        "social": 750,
        "romantic": 79,
    },
    "DeepSeek V3.2": {
        "personal": 2129,
        "education": 167,
        "employment": 531,
        "finance": 74,
        "housing": 187,
        "legal": 40,
        "health": 476,
        "schedule": 160,
        "identity": 621,
        "social": 786,
        "romantic": 139,
    },
    "Gemini 3.1 Pro": {
        "personal": 2385,
        "education": 145,
        "employment": 665,
        "finance": 78,
        "housing": 274,
        "legal": 32,
        "health": 377,
        "schedule": 274,
        "identity": 320,
        "social": 622,
        "romantic": 138,
    },
    "GPT OSS 120B": {
        "personal": 2077,
        "education": 200,
        "employment": 649,
        "finance": 98,
        "housing": 262,
        "legal": 37,
        "health": 511,
        "schedule": 320,
        "identity": 423,
        "social": 612,
        "romantic": 121,
    },
    "Llama 3.3 70B": {
        "personal": 2477,
        "education": 220,
        "employment": 518,
        "finance": 86,
        "housing": 212,
        "legal": 36,
        "health": 481,
        "schedule": 225,
        "identity": 352,
        "social": 594,
        "romantic": 109,
    },
    "Qwen 3 235B": {
        "personal": 2122,
        "education": 255,
        "employment": 566,
        "finance": 99,
        "housing": 249,
        "legal": 34,
        "health": 462,
        "schedule": 232,
        "identity": 473,
        "social": 668,
        "romantic": 150,
    },
    "Grok 4.1 Fast": {
        "personal": 2364,
        "education": 166,
        "employment": 482,
        "finance": 61,
        "housing": 210,
        "legal": 26,
        "health": 373,
        "schedule": 233,
        "identity": 539,
        "social": 716,
        "romantic": 140,
    },
    "GLM-4.7": {
        "personal": 2393,
        "education": 138,
        "employment": 682,
        "finance": 71,
        "housing": 263,
        "legal": 33,
        "health": 387,
        "schedule": 246,
        "identity": 369,
        "social": 597,
        "romantic": 131,
    },
}


def matrix() -> np.ndarray:
    return np.array([
        [DATA[model][category] for category in CATEGORIES]
        for model in MODELS
    ])


def style_axis(ax) -> None:
    ax.set_xticks(range(len(CATEGORIES)))
    ax.set_xticklabels(CATEGORIES, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODELS)
    ax.tick_params(axis="both", which="major", length=0)

    ax.set_xticks(np.arange(-0.5, len(CATEGORIES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(MODELS), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)

    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.85)
        ax.spines[side].set_color("#555555")


def annotate_heatmap(ax, values: np.ndarray) -> None:
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            color = "white" if value >= 900 else "#222222"
            ax.text(
                col,
                row,
                f"{value:.0f}",
                ha="center",
                va="center",
                fontsize=DATA_TEXT_SIZE,
                fontweight="bold",
                color=color,
            )


def draw_heatmap() -> None:
    values = matrix()

    fig, ax = plt.subplots(figsize=(17.4, 8.4))
    image = ax.imshow(values, cmap=plt.get_cmap("YlGnBu"), vmin=0, vmax=float(values.max()), aspect="auto")

    style_axis(ax)
    annotate_heatmap(ax, values)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("Memory items", fontsize=SCALE_TEXT_SIZE, fontweight="semibold")
    colorbar.ax.tick_params(labelsize=SCALE_TEXT_SIZE, length=3)

    ax.set_title(
        "Memory Distribution of Inferance and Cosine Similarity Fixed Partitions",
        fontsize=TITLE_TEXT_SIZE,
        fontweight="semibold",
        pad=14,
        color="#222222",
    )
    fig.subplots_adjust(left=0.20, right=0.94, bottom=0.28, top=0.86)

    stem = "memory_distribution_heat_map"
    fig.savefig(OUTDIR / f"{stem}.png", bbox_inches="tight")
    fig.savefig(OUTDIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    draw_heatmap()
    print(f"Figures saved to: {OUTDIR.resolve()}")


if __name__ == "__main__":
    main()
