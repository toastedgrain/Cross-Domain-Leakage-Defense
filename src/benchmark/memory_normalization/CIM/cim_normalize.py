#!/usr/bin/env python3
"""Convert raw CIM profiles into a flat PersistBench-style JSONL file.

The output keeps the usual top-level `query` and `memories` fields, but the
query is written as a template string with placeholders:

  {memories}
  {task}
  {recipient}

The actual task and recipient are stored separately in the row so the file can
be reused for later formatting or analysis.

Task/recipient text is sourced from the HuggingFace dataset (same source used
by the labeler) so context hashes in the JSONL always match those in the labels
file.  Bio memory lines (name, sex, address, age) are sourced from cim_raw.json.

Usage:
  python src/benchmark/memory_partitioning/cim_normalize.py
  python src/benchmark/memory_partitioning/cim_normalize.py \\
      --input benchmark_samples/CIM/baseline/cim_raw.json \\
      --output benchmark_samples/CIM/baseline/cim_normalized.jsonl \\
      --hf-dataset facebook/CIMemories
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "VERTEXAI_SERVICE_ACCOUNT_PATH",
    str(Path.home() / "Downloads" / "VERTEXAI_SERVICE_ACCOUNT.json"),
)

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT_FILE = _PROJECT_ROOT / "benchmark_samples" / "CIM" / "baseline" / "cim_raw.json"
DEFAULT_OUTPUT_FILE = _PROJECT_ROOT / "benchmark_samples" / "CIM" / "baseline" / "cim_normalized.jsonl"
DEFAULT_PROMPT_TEMPLATE_FILE = _PROJECT_ROOT / "prompts" / "cim_paper.txt"
DEFAULT_LABELS_FILE = _PROJECT_ROOT / "outputs" / "CIM" / "cim_labels_deepseek.json"
DEFAULT_HF_DATASET = "facebook/CIMemories"


def _extract_task_info(prompt: str) -> tuple[str, str]:
    """Extract recipient and purpose from CIM prompt text.

    CIM prompts end with:
      'Write a complete message to {recipient} to achieve the following purpose: {purpose}.'
    Matches the logic in cim_labeler_deepseek.py exactly.
    """
    pattern = r"Write a complete message to (.+?) to achieve the following purpose: (.+?)\."
    match = re.search(pattern, prompt)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return "the recipient", prompt[-200:].strip()


def _load_hf_context_map(
    dataset_id: str = DEFAULT_HF_DATASET, split: str = "test"
) -> dict[str, list[dict[str, str]]]:
    """Load HuggingFace CIM dataset and return name -> list of context dicts.

    Each context dict has: recipient, purpose (=task), prompt_hash.
    The prompt_hash is md5(full_hf_prompt)[:12] — the formula used by the
    labeler when it generated cim_labels_deepseek.json.
    """
    from collections import defaultdict
    from datasets import load_dataset

    print(f"Loading HuggingFace dataset {dataset_id!r} (split={split})...")
    ds = load_dataset(dataset_id, split=split)

    seen: set[tuple[str, str]] = set()
    by_name: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in ds:
        key = (row["name"], row["prompt"])
        if key in seen:
            continue
        seen.add(key)
        name, prompt = row["name"], row["prompt"]
        recipient, purpose = _extract_task_info(prompt)
        # Use full-prompt hash — matches the formula used during label generation
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:12]
        by_name[name].append({
            "purpose": purpose,
            "recipient": recipient,
            "prompt_hash": prompt_hash,
        })

    total = sum(len(v) for v in by_name.values())
    print(f"  Found {len(by_name)} unique names, {total} total contexts from HuggingFace")
    return dict(by_name)


def _match_hf_context(
    name: str,
    raw_task: str,
    raw_recipient: str,
    hf_by_name: dict[str, list[dict[str, str]]],
) -> dict[str, str] | None:
    """Find the HuggingFace context dict that best matches a cim_raw.json context.

    Returns the context dict (with purpose, recipient, prompt_hash) if found, else None.
    Matching is case-insensitive exact, then falls back to substring check.
    """
    candidates = hf_by_name.get(name, [])
    if not candidates:
        return None

    raw_task_l = raw_task.lower().strip()
    raw_recip_l = raw_recipient.lower().strip()

    # Exact lowercase match
    for c in candidates:
        if c["purpose"].lower() == raw_task_l and c["recipient"].lower() == raw_recip_l:
            return c

    # Recipient exact + task substring (handles minor wording differences)
    recip_matches = [c for c in candidates if c["recipient"].lower() == raw_recip_l]
    for c in recip_matches:
        hf_l = c["purpose"].lower()
        if raw_task_l in hf_l or hf_l in raw_task_l:
            return c

    # Recipient substring + task substring
    for c in candidates:
        r_l = c["recipient"].lower()
        if raw_recip_l in r_l or r_l in raw_recip_l:
            hf_l = c["purpose"].lower()
            if raw_task_l in hf_l or hf_l in raw_task_l:
                return c

    return None


_LABEL_NORMALIZE = {
    # DeepSeek labeler vocabulary
    "necessary": "necessary",
    "inappropriate": "inappropriate",
    # Llama labeler vocabulary
    "share": "necessary",
    "private": "inappropriate",
}


def _load_labels(
    labels_path: Path,
) -> dict[tuple[str, str], dict[str, str | None]]:
    """Load labels file and return index keyed by (name, context_hash).

    Accepts both label vocabularies:
      - necessary / inappropriate  (DeepSeek labeler)
      - share / private            (Llama labeler)
    Both are normalised to necessary / inappropriate internally.
    """
    with open(labels_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw_labels: dict[str, str | None] = data.get("labels", {})
    index: dict[tuple[str, str], dict[str, str | None]] = {}
    for key, value in raw_labels.items():
        parts = key.split("|", 2)
        if len(parts) == 3:
            name, ctx_hash, attr_name = parts
            normalized = _LABEL_NORMALIZE.get(value) if value is not None else None
            index.setdefault((name, ctx_hash), {})[attr_name] = normalized
    return index


def _build_memory_lines(
    profile: dict[str, Any],
) -> tuple[list[str], dict[str, str]]:
    """Render a CIM profile into the flat memory list and attribute_memory_map.

    Returns:
        memories: flat list of memory strings (one per attribute)
        attribute_memory_map: {attribute_name: memory_statement} for all attributes
    """
    bio = profile.get("bio", {})
    attributes = profile.get("information_attributes", {})

    try:
        name = bio["name"]
        sex = bio["sex"]
        address = bio["address"]
        age = bio["age"]
    except KeyError as exc:
        raise ValueError(
            f"CIM profile is missing required bio field: {exc.args[0]}"
        ) from exc

    bio_statements = {
        "name": f"{name} is the person's name.",
        "sex": f"The person's sex is {sex}.",
        "address": f"The person's address is {address}.",
        "age": f"The person is {age} years old.",
    }

    attribute_memory_map: dict[str, str] = dict(bio_statements)

    for attr_name, attr_data in attributes.items():
        if attr_name in bio_statements:
            continue
        memory_statement = attr_data.get("memory_statement")
        if not isinstance(memory_statement, str):
            raise ValueError(
                f"Invalid memory_statement for attribute {attr_name!r}"
            )
        attribute_memory_map[attr_name] = memory_statement

    memories = list(attribute_memory_map.values())
    return memories, attribute_memory_map


def _load_prompt_template(prompt_template_file: Path) -> str:
    if not prompt_template_file.exists():
        raise FileNotFoundError(
            f"Prompt template file not found: {prompt_template_file}"
        )
    return prompt_template_file.read_text(encoding="utf-8")


def convert_cim_raw_to_persistbench(
    input_file: Path,
    output_file: Path,
    prompt_template_file: Path = DEFAULT_PROMPT_TEMPLATE_FILE,
    labels_file: Path | None = None,
    hf_dataset: str | None = DEFAULT_HF_DATASET,
    hf_split: str = "test",
) -> int:
    """Convert CIM raw profiles to JSONL rows that resemble PersistBench rows.

    Task/recipient text comes from the HuggingFace dataset (same source as the
    labeler) so context hashes match the labels file.  When hf_dataset is None,
    falls back to cim_raw.json task/recipient values (hashes may not match
    existing labels).

    When labels_file is provided (or the default exists), each row is annotated
    with required_attributes and forbidden_attributes.
    """
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        profiles = json.load(f)

    if not isinstance(profiles, list):
        raise ValueError("CIM raw file must contain a top-level JSON array")

    query_template = _load_prompt_template(prompt_template_file).strip()

    # Load HuggingFace context map for correct task/recipient text
    hf_by_name: dict[str, list[dict[str, str]]] = {}
    if hf_dataset:
        try:
            hf_by_name = _load_hf_context_map(hf_dataset, hf_split)
        except Exception as e:
            print(
                f"Warning: Could not load HuggingFace dataset ({e}). "
                "Falling back to cim_raw.json task/recipient — "
                "context hashes may not match existing labels."
            )

    # Load labels index if available
    resolved_labels = labels_file if labels_file is not None else DEFAULT_LABELS_FILE
    labels_index: dict[tuple[str, str], dict[str, str | None]] = {}
    if resolved_labels.exists():
        labels_index = _load_labels(resolved_labels)
        print(f"Loaded labels from {resolved_labels} ({len(labels_index)} contexts)")
    else:
        print(f"No labels file found at {resolved_labels} — skipping attribute annotation")

    rows: list[dict[str, Any]] = []
    unmatched = 0

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
            raise ValueError(
                f"CIM profile is missing required bio field: {exc.args[0]}"
            ) from exc

        memories, attribute_memory_map = _build_memory_lines(profile)

        for context in contexts:
            if not isinstance(context, dict):
                raise ValueError("Each CIM context must be a JSON object")

            raw_recipient = context.get("recipient")
            raw_task = context.get("task")
            if not isinstance(raw_recipient, str) or not raw_recipient.strip():
                raise ValueError("CIM context is missing a valid recipient")
            if not isinstance(raw_task, str) or not raw_task.strip():
                raise ValueError("CIM context is missing a valid task")

            # Match to HuggingFace context to get the prompt_hash used during labeling
            prompt_hash: str | None = None
            if hf_by_name:
                hf_ctx = _match_hf_context(name, raw_task, raw_recipient, hf_by_name)
                if hf_ctx:
                    prompt_hash = hf_ctx["prompt_hash"]
                else:
                    unmatched += 1

            # Hash encodes memories + task + recipient so each context row is
            # unique even though query template and memories are the same for
            # all rows of the same persona.
            row_hash = hashlib.md5(
                json.dumps(
                    {"name": name, "task": raw_task, "recipient": raw_recipient},
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode()
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

            if labels_index and prompt_hash is not None:
                attr_labels = labels_index.get((name, prompt_hash), {})
                row["required_attributes"] = [
                    a for a, lbl in attr_labels.items() if lbl == "necessary"
                ]
                row["forbidden_attributes"] = [
                    a for a, lbl in attr_labels.items() if lbl == "inappropriate"
                ]

            rows.append(row)

    if unmatched:
        print(
            f"Warning: {unmatched} context(s) had no HuggingFace match — "
            "used raw task/recipient values (hashes may not match labels)"
        )

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
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help=f"Input raw CIM JSON file (default: {DEFAULT_INPUT_FILE})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output JSONL file (default: {DEFAULT_OUTPUT_FILE})",
    )
    parser.add_argument(
        "--prompt-template",
        type=Path,
        default=DEFAULT_PROMPT_TEMPLATE_FILE,
        help=(
            "Prompt template file to store in the query field "
            f"(default: {DEFAULT_PROMPT_TEMPLATE_FILE})"
        ),
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help=(
            "DeepSeek labels JSON file for annotating required/forbidden attributes "
            f"(default: {DEFAULT_LABELS_FILE})"
        ),
    )
    parser.add_argument(
        "--hf-dataset",
        default=DEFAULT_HF_DATASET,
        help=(
            "HuggingFace dataset ID for sourcing task/recipient text that matches "
            f"the labeler's context hashes (default: {DEFAULT_HF_DATASET}). "
        ),
    )
    parser.add_argument(
        "--no-hf",
        action="store_true",
        help=(
            "Skip HuggingFace loading and use cim_raw.json task/recipient directly. "
            "Context hashes may not match existing labels."
        ),
    )
    parser.add_argument(
        "--hf-split",
        default="test",
        help="HuggingFace dataset split (default: test)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    convert_cim_raw_to_persistbench(
        input_file=args.input,
        output_file=args.output,
        prompt_template_file=args.prompt_template,
        labels_file=args.labels,
        hf_dataset=None if args.no_hf else args.hf_dataset,
        hf_split=args.hf_split,
    )


if __name__ == "__main__":
    main()
