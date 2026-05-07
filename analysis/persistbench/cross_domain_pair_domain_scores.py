#!/usr/bin/env python3
"""Print pair-level PersistBench cross-domain scores by memory/query domain."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.persistbench.figures.cross_domain_pair_domain_scores import (
    DEFAULT_BENCHMARK,
    MEMORY_STRUCTURES,
    METHODS,
    domain_label,
    method_data,
    sort_model,
)


def format_value(value: float) -> str:
    if value != value:
        return ""
    return f"{value:.3f}"


def print_matrix_rows(
    *,
    method: str,
    model: str,
    domains: list[str],
    matrix,
    counts,
) -> None:
    method_label = MEMORY_STRUCTURES[method]["label"]
    for row_index, memory_domain in enumerate(domains):
        for col_index, query_domain in enumerate(domains):
            count = int(counts[row_index, col_index])
            if count == 0:
                continue
            print(
                "\t".join(
                    [
                        method_label,
                        model,
                        domain_label(memory_domain),
                        domain_label(query_domain),
                        str(count),
                        format_value(float(matrix[row_index, col_index])),
                    ]
                )
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument(
        "--method",
        choices=METHODS,
        action="append",
        help="Method to print. Repeat to print multiple methods. Defaults to all methods.",
    )
    parser.add_argument(
        "--model",
        action="append",
        help="Canonical model label to print. Repeat to print multiple models. Defaults to all models.",
    )
    parser.add_argument(
        "--average-only",
        action="store_true",
        help="Only print the average-across-models rows.",
    )
    args = parser.parse_args()

    methods = args.method or METHODS
    requested_models = set(args.model or [])

    print(
        "\t".join(
            [
                "Method",
                "Model",
                "Memory Domain",
                "Query Domain",
                "Samples",
                "Average Best Judge Score",
            ]
        )
    )

    for method in methods:
        domains, models, per_model, average = method_data(method, args.benchmark)
        selected_models = [
            model
            for model in sorted(models, key=sort_model)
            if not requested_models or model in requested_models
        ]

        if not args.average_only:
            for model in selected_models:
                matrix, counts = per_model[model]
                print_matrix_rows(
                    method=method,
                    model=model,
                    domains=domains,
                    matrix=matrix,
                    counts=counts,
                )

        matrix, counts = average
        print_matrix_rows(
            method=method,
            model="Average",
            domains=domains,
            matrix=matrix,
            counts=counts,
        )


if __name__ == "__main__":
    main()
