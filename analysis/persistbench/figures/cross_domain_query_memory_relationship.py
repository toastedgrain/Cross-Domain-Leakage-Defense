#!/usr/bin/env python3
"""Visualize PersistBench cross-domain memory-domain/query-domain relationships."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BENCHMARK = (
    REPO_ROOT
    / "benchmark_samples"
    / "persistbench"
    / "baseline"
    / "full_benchmark.jsonl"
)
OUTDIR = Path(__file__).resolve().parent / "cross_domain_query_memory_relationship"

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


def label(domain: str) -> str:
    return DOMAIN_LABELS.get(domain, domain)


def load_cross_domain_pairs(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("failure_type") != "cross_domain":
                continue
            memory_domain = row.get("memory_domain")
            query_domain = row.get("query_domain")
            if memory_domain and query_domain:
                pairs.append((memory_domain, query_domain))
    return pairs


def build_matrix(pairs: list[tuple[str, str]]) -> tuple[list[str], np.ndarray]:
    domains = [domain for domain in DOMAIN_ORDER if domain in {d for pair in pairs for d in pair}]
    index = {domain: idx for idx, domain in enumerate(domains)}
    matrix = np.zeros((len(domains), len(domains)), dtype=float)
    for memory_domain, query_domain in pairs:
        if memory_domain in index and query_domain in index:
            matrix[index[memory_domain], index[query_domain]] += 1
    return domains, matrix


def cramers_v(matrix: np.ndarray) -> tuple[float, float]:
    total = float(matrix.sum())
    if total == 0:
        return 0.0, 0.0
    row_totals = matrix.sum(axis=1, keepdims=True)
    col_totals = matrix.sum(axis=0, keepdims=True)
    expected = row_totals @ col_totals / total
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2_terms = np.where(expected > 0, (matrix - expected) ** 2 / expected, 0)
    chi2 = float(chi2_terms.sum())
    rows, cols = matrix.shape
    v = math.sqrt(chi2 / (total * max(1, min(rows - 1, cols - 1))))
    return chi2, v


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    row_totals = matrix.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(matrix, row_totals, out=np.zeros_like(matrix), where=row_totals > 0)


def lift_matrix(matrix: np.ndarray) -> np.ndarray:
    total = matrix.sum()
    row_totals = matrix.sum(axis=1, keepdims=True)
    col_totals = matrix.sum(axis=0, keepdims=True)
    expected = row_totals @ col_totals / total
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(matrix, expected, out=np.zeros_like(matrix), where=expected > 0)


def annotate(ax, values: np.ndarray, formatter) -> None:
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if not np.isfinite(value) or value == 0:
                continue
            ax.text(
                col,
                row,
                formatter(value),
                ha="center",
                va="center",
                fontsize=7.5,
                fontweight="semibold",
                color="white" if value > np.nanmax(values) * 0.62 else "#222222",
            )


def style_domain_axes(ax, domains: list[str]) -> None:
    labels = [label(domain) for domain in domains]
    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels(labels, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(domains)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Query domain", fontweight="semibold", labelpad=8)
    ax.set_ylabel("Memory domain", fontweight="semibold", labelpad=8)
    ax.set_xticks(np.arange(-0.5, len(domains), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(domains), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(axis="both", which="major", length=0)
    ax.tick_params(which="minor", bottom=False, left=False)
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.85)
        ax.spines[side].set_color("#555555")


def save_heatmap(
    values: np.ndarray,
    domains: list[str],
    *,
    title: str,
    colorbar_label: str,
    output_path: Path,
    cmap_name: str,
    vmin: float | None = None,
    vmax: float | None = None,
    formatter=lambda value: f"{value:.0f}",
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 7.1))
    cmap = plt.get_cmap(cmap_name).copy()
    image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_title(title, fontsize=13, fontweight="semibold", pad=12)
    style_domain_axes(ax, domains)
    annotate(ax, values, formatter)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.025)
    colorbar.set_label(colorbar_label, fontsize=10.5, fontweight="semibold")
    fig.subplots_adjust(left=0.19, right=0.92, bottom=0.21, top=0.9)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def top_pairs(matrix: np.ndarray, domains: list[str], *, k: int = 8) -> list[tuple[float, float, str, str]]:
    lifts = lift_matrix(matrix)
    pairs: list[tuple[float, float, str, str]] = []
    for row, memory_domain in enumerate(domains):
        for col, query_domain in enumerate(domains):
            count = matrix[row, col]
            lift = lifts[row, col]
            if count > 0 and np.isfinite(lift):
                pairs.append((float(lift), float(count), memory_domain, query_domain))
    return sorted(pairs, reverse=True)[:k]


def save_summary(path: Path, matrix: np.ndarray, domains: list[str]) -> None:
    chi2, v = cramers_v(matrix)
    rows = []
    rows.append("# Cross-Domain Query/Memory Domain Relationship\n")
    rows.append(f"- Samples: {int(matrix.sum())}")
    rows.append(f"- Chi-square statistic: {chi2:.2f}")
    rows.append(f"- Cramer's V: {v:.3f}")
    rows.append(
        "- Interpretation: Cramer's V near 0 means weak association; near 1 means strong association.\n"
    )
    rows.append("## Most Overrepresented Domain Pairs\n")
    for lift, count, memory_domain, query_domain in top_pairs(matrix, domains):
        rows.append(
            f"- {label(memory_domain)} memory -> {label(query_domain)} query: "
            f"n={count:.0f}, lift={lift:.2f}x"
        )
    rows.append("\n## Suggested Visualizations\n")
    rows.extend(
        [
            "- Count heatmap: best first view for absolute dataset composition.",
            "- Row-normalized heatmap: best for asking where each memory domain tends to be queried.",
            "- Lift heatmap: best for finding relationships beyond marginal domain frequency.",
            "- Bipartite/alluvial diagram: useful in a paper or slide when emphasizing flows from memory domains to query domains.",
            "- Clustered heatmap: useful if domain order should be data-driven rather than semantic.",
        ]
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output-dir", type=Path, default=OUTDIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs = load_cross_domain_pairs(args.benchmark)
    domains, matrix = build_matrix(pairs)

    save_heatmap(
        matrix,
        domains,
        title="Cross-Domain Samples: Memory Domain vs Query Domain",
        colorbar_label="Sample count",
        output_path=args.output_dir / "memory_query_domain_count_heatmap.png",
        cmap_name="Blues",
        vmin=0,
        formatter=lambda value: f"{value:.0f}",
    )
    save_heatmap(
        row_normalize(matrix) * 100,
        domains,
        title="Cross-Domain Samples: Query Share Within Each Memory Domain",
        colorbar_label="Row share (%)",
        output_path=args.output_dir / "memory_query_domain_row_share_heatmap.png",
        cmap_name="YlGnBu",
        vmin=0,
        vmax=100,
        formatter=lambda value: f"{value:.0f}",
    )
    save_heatmap(
        lift_matrix(matrix),
        domains,
        title="Cross-Domain Samples: Observed / Expected Pairing",
        colorbar_label="Lift",
        output_path=args.output_dir / "memory_query_domain_lift_heatmap.png",
        cmap_name="RdBu_r",
        vmin=0,
        vmax=max(2.0, float(np.nanmax(lift_matrix(matrix)))),
        formatter=lambda value: f"{value:.1f}",
    )
    save_summary(args.output_dir / "memory_query_domain_relationship_summary.md", matrix, domains)

    chi2, v = cramers_v(matrix)
    print(f"Saved figures to: {args.output_dir.resolve()}")
    print(f"Samples: {int(matrix.sum())}")
    print(f"Chi-square statistic: {chi2:.2f}")
    print(f"Cramer's V: {v:.3f}")


if __name__ == "__main__":
    main()
