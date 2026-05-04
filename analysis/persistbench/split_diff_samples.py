#!/usr/bin/env python3
"""Split diff_samples.json into compact per-model/per-comparison JSON files."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from typing import Any


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def slug(value: str) -> str:
    value = value.lower().replace("->", "to")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def compact_memories(value: Any) -> Any:
    """Remove empty memory categories while preserving list/dict structure."""
    if isinstance(value, list):
        return [item for item in value if item not in (None, "", [], {})]

    if isinstance(value, dict):
        compacted: dict[str, Any] = {}
        for key, child in value.items():
            child_compact = compact_memories(child)
            if child_compact not in (None, "", [], {}):
                compacted[key] = child_compact
        return compacted

    return value


def compact_method(method: dict[str, Any]) -> dict[str, Any]:
    judge = method.get("judge") or {}
    judge_text = (
        judge.get("explanation")
        or judge.get("reasoning")
        or judge.get("rationale")
        or judge.get("summary")
    )

    compacted = {
        "method": method.get("method"),
        "score": method.get("score"),
        "generation_index": method.get("generation_index"),
        "all_scores": method.get("all_scores"),
        "memory_format": method.get("memory_format"),
        "memories": compact_memories(method.get("memories")),
        "highest_scored_output": method.get("output"),
    }

    if judge_text:
        compacted["judge_explanation"] = judge_text
    elif judge:
        compacted["judge"] = judge

    return compacted


def compact_sample(sample: dict[str, Any], comparison: dict[str, Any]) -> dict[str, Any]:
    first = comparison["first_method"]
    second = comparison["second_method"]

    return {
        "rank": sample.get("rank"),
        "hash_id": sample.get("hash_id"),
        "score_change": {
            "from_method": first,
            "to_method": second,
            "from_score": sample.get("first_score"),
            "to_score": sample.get("second_score"),
            "improvement": sample.get("improvement"),
            "note": "Higher scores are worse; improvement means the second method has a lower worst-case score.",
        },
        "query": sample.get("query"),
        "memory_domain": sample.get("memory_domain"),
        "query_domain": sample.get("query_domain"),
        "methods": {
            first: compact_method(sample["methods"][first]),
            second: compact_method(sample["methods"][second]),
        },
    }


def split(input_path: pathlib.Path, output_dir: pathlib.Path) -> list[pathlib.Path]:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    output_paths: list[pathlib.Path] = []

    output_dir.mkdir(parents=True, exist_ok=True)

    for model in data["models"]:
        model_label = model["label"]
        model_dir = output_dir / slug(model_label)
        model_dir.mkdir(parents=True, exist_ok=True)

        for comparison in model["comparisons"]:
            payload = {
                "model": {
                    "label": model_label,
                    "model_id": model["model_id"],
                },
                "comparison": {
                    "name": comparison["name"],
                    "first_method": comparison["first_method"],
                    "second_method": comparison["second_method"],
                },
                "score_rule": "Each method score is max(valid judge scores across 3 runs). Scores are 1-5 and higher is worse.",
                "samples": [
                    compact_sample(sample, comparison)
                    for sample in comparison["samples"]
                ],
            }

            path = model_dir / f"{slug(comparison['name'])}.json"
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            output_paths.append(path)

    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, default=REPO_ROOT / "diff_samples.json")
    parser.add_argument("--output-dir", type=pathlib.Path, default=REPO_ROOT / "diff_samples_split")
    args = parser.parse_args()

    paths = split(args.input, args.output_dir)
    for path in paths:
        print(path.relative_to(REPO_ROOT))
    print(f"Wrote {len(paths)} files under {args.output_dir}")


if __name__ == "__main__":
    main()
