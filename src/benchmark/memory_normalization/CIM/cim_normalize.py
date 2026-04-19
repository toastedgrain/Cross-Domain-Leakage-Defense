#!/usr/bin/env python3
"""Convert raw CIM profiles into a flat PersistBench-style JSONL file.

Each output row has `query` (template), `memories`, `task`, `recipient`,
`attribute_memory_map`, and `hash_id`. When a labels file is present, rows are
also annotated with `required_attributes` and `forbidden_attributes`.

Context hashes are computed with the same formula as cim_labeler_deepseek_2majority.py
(md5 of "name|task|recipient"), so labels look up correctly without any
HuggingFace dependency.

Usage:
  python src/benchmark/memory_normalization/CIM/cim_normalize.py
  python src/benchmark/memory_normalization/CIM/cim_normalize.py \\
      --input benchmark_samples/CIM/raw/cim_custom_raw.json \\
      --output benchmark_samples/CIM/normalized/cim_custom_normalized.jsonl \\
      --labels outputs/CIM/cim_labels_custom_deepseek.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT_FILE = _PROJECT_ROOT / "benchmark_samples" / "CIM" / "raw" / "cim_custom_raw.json"
DEFAULT_OUTPUT_FILE = _PROJECT_ROOT / "benchmark_samples" / "CIM" / "normalized" / "cim_custom_normalized.jsonl"
DEFAULT_PROMPT_TEMPLATE_FILE = _PROJECT_ROOT / "prompts" / "cim_paper.txt"
DEFAULT_LABELS_FILE = _PROJECT_ROOT / "outputs" / "CIM" / "cim_labels_custom_deepseek.json"

_LABEL_NORMALIZE = {
    "necessary": "necessary",
    "inappropriate": "inappropriate",
    "share": "necessary",
    "private": "inappropriate",
}


def _ctx_hash(name: str, task: str, recipient: str) -> str:
    """Context hash matching the formula used by cim_labeler_deepseek_2majority."""
    return hashlib.md5(f"{name.lower()}|{task.lower()}|{recipient.lower()}".encode()).hexdigest()


def _load_labels(labels_path: Path) -> dict[tuple[str, str], dict[str, str | None]]:
    """Load labels file and return index keyed by (name, context_hash)."""
    with open(labels_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    index: dict[tuple[str, str], dict[str, str | None]] = {}
    for key, value in data.get("labels", {}).items():
        parts = key.split("|", 2)
        if len(parts) == 3:
            name, ctx_hash, attr_name = parts
            index.setdefault((name, ctx_hash), {})[attr_name] = _LABEL_NORMALIZE.get(value) if value is not None else None
    return index


def _build_memory_map(profile: dict[str, Any]) -> dict[str, str]:
    """Return {attribute_name: memory_statement} for all attributes in a profile."""
    bio = profile.get("bio", {})
    try:
        name, sex, address, age = bio["name"], bio["sex"], bio["address"], bio["age"]
    except KeyError as exc:
        raise ValueError(f"CIM profile is missing required bio field: {exc.args[0]}") from exc

    memory_map: dict[str, str] = {
        "name": f"{name} is the person's name.",
        "sex": f"The person's sex is {sex}.",
        "address": f"The person's address is {address}.",
        "age": f"The person is {age} years old.",
    }
    for attr_name, attr_data in profile.get("information_attributes", {}).items():
        if attr_name in memory_map:
            continue
        memory_statement = attr_data.get("memory_statement")
        if not isinstance(memory_statement, str):
            raise ValueError(f"Invalid memory_statement for attribute {attr_name!r}")
        memory_map[attr_name] = memory_statement

    return memory_map


def convert_cim_raw_to_persistbench(
    input_file: Path,
    output_file: Path,
    prompt_template_file: Path = DEFAULT_PROMPT_TEMPLATE_FILE,
    labels_file: Path | None = None,
) -> int:
    """Convert CIM raw profiles to PersistBench-style JSONL rows."""
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    if not prompt_template_file.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_template_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        profiles = json.load(f)
    if not isinstance(profiles, list):
        raise ValueError("CIM raw file must contain a top-level JSON array")

    query_template = prompt_template_file.read_text(encoding="utf-8").strip()

    resolved_labels = labels_file if labels_file is not None else DEFAULT_LABELS_FILE
    labels_index: dict[tuple[str, str], dict[str, str | None]] = {}
    if resolved_labels.exists():
        labels_index = _load_labels(resolved_labels)
        print(f"Loaded labels from {resolved_labels} ({len(labels_index)} contexts)")
    else:
        print(f"No labels file at {resolved_labels} — skipping attribute annotation")

    rows: list[dict[str, Any]] = []

    for profile in profiles:
        if not isinstance(profile, dict):
            raise ValueError("Each CIM profile must be a JSON object")

        bio = profile.get("bio", {})
        contexts = profile.get("contexts", [])
        if not isinstance(contexts, list):
            raise ValueError("CIM profile must contain a 'contexts' list")

        try:
            name = bio["name"]
        except KeyError as exc:
            raise ValueError(f"CIM profile is missing required bio field: {exc.args[0]}") from exc

        attribute_memory_map = _build_memory_map(profile)
        memories = list(attribute_memory_map.values())

        for context in contexts:
            if not isinstance(context, dict):
                raise ValueError("Each CIM context must be a JSON object")

            raw_recipient = context.get("recipient")
            raw_task = context.get("task")
            if not isinstance(raw_recipient, str) or not raw_recipient.strip():
                raise ValueError("CIM context is missing a valid recipient")
            if not isinstance(raw_task, str) or not raw_task.strip():
                raise ValueError("CIM context is missing a valid task")

            ctx_hash = _ctx_hash(name, raw_task, raw_recipient)
            row_hash = hashlib.md5(
                json.dumps({"name": name, "task": raw_task, "recipient": raw_recipient}, ensure_ascii=False, sort_keys=True).encode()
            ).hexdigest()

            row: dict[str, Any] = {
                "name": name,
                "query": query_template,
                "memories": memories,
                "task": raw_task,
                "recipient": raw_recipient,
                "attribute_memory_map": attribute_memory_map,
                "hash_id": row_hash,
            }

            if labels_index:
                attr_labels = labels_index.get((name, ctx_hash), {})
                row["required_attributes"] = [a for a, lbl in attr_labels.items() if lbl == "necessary"]
                row["forbidden_attributes"] = [a for a, lbl in attr_labels.items() if lbl == "inappropriate"]

            rows.append(row)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")

    print(f"Wrote {len(rows)} CIM sample(s) to {output_file}")
    return len(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw CIM profiles into flat PersistBench-style JSONL.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_FILE,
                        help=f"Input raw CIM JSON file (default: {DEFAULT_INPUT_FILE})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE,
                        help=f"Output JSONL file (default: {DEFAULT_OUTPUT_FILE})")
    parser.add_argument("--prompt-template", type=Path, default=DEFAULT_PROMPT_TEMPLATE_FILE,
                        help=f"Prompt template file (default: {DEFAULT_PROMPT_TEMPLATE_FILE})")
    parser.add_argument("--labels", type=Path, default=None,
                        help=f"Labels JSON file (default: {DEFAULT_LABELS_FILE})")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    convert_cim_raw_to_persistbench(
        input_file=args.input,
        output_file=args.output,
        prompt_template_file=args.prompt_template,
        labels_file=args.labels,
    )


if __name__ == "__main__":
    main()