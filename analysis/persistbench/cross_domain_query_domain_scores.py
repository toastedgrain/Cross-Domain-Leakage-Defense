#!/usr/bin/env python3
"""Print cross-domain PersistBench score summaries by model and query domain."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.persistbench.figures.cross_domain_domain_scores import (
    DEFAULT_BENCHMARK,
    MEMORY_STRUCTURES,
    aggregate_scores,
)


def format_avg(scores: list[int]) -> str:
    return "-" if not scores else f"{float(np.mean(scores)):.2f}"


def print_structure(name: str, benchmark_path: Path) -> None:
    config = MEMORY_STRUCTURES[name]
    models, domains, scores = aggregate_scores(config["checkpoints"], benchmark_path)

    print(f"\n{config['label']}")
    print("=" * len(config["label"]))
    print("Model\tDomain\tN\tAverage best judge score")
    for model in models:
        for domain in domains:
            cell_scores = scores[model].get(domain, [])
            print(f"{model}\t{domain}\t{len(cell_scores)}\t{format_avg(cell_scores)}")

    print("\nOverall by model")
    print("Model\tN\tAverage best judge score")
    for model in models:
        all_scores = [
            score
            for domain_scores in scores[model].values()
            for score in domain_scores
        ]
        print(f"{model}\t{len(all_scores)}\t{format_avg(all_scores)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument(
        "--structure",
        action="append",
        choices=sorted(MEMORY_STRUCTURES),
        help="Memory structure to print. Repeat for multiple. Default: all structures.",
    )
    args = parser.parse_args()

    for structure in args.structure or MEMORY_STRUCTURES:
        print_structure(structure, args.benchmark)


if __name__ == "__main__":
    main()
