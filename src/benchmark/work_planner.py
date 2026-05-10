"""Utilities for loading benchmark inputs and building generation work queues."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

from benchmark.checkpoint import (
    Checkpoint,
    GenerationStatus,
    get_generation_status,
    initialize_checkpoint,
    save_checkpoint,
)
from benchmark.config import (
    FAILURE_TYPE_CROSS_DOMAIN,
    BenchmarkConfig,
    ModelEntry,
    get_generations_for_failure_type,
    load_benchmark_config_data,
    resolve_entry_configuration,
    validate_failure_type,
)
from benchmark.exceptions import FatalBenchmarkError
from benchmark.utils import generate_hash_id

# Normalized input row and queued generation triple (entry, model, gen_idx)
InputEntry: TypeAlias = dict[str, Any]
WorkItem: TypeAlias = tuple[InputEntry, ModelEntry, int]


def _normalize_memories(memories: list[str] | dict) -> list[str]:
    """Convert memories to a plain list for prompt consumption.

    - list                        → returned as-is
    - dict[str, list[str]]        → each category becomes one string: "category: mem1, mem2, …"
    - dict[str, dict[str, list]]  → two-level tree: all leaf memories flattened into a plain list
    """
    if isinstance(memories, dict):
        first = next(iter(memories.values()), None)
        if isinstance(first, dict):
            # Two-level tree: flatten every leaf memory into a plain list
            return [
                mem
                for cat_node in memories.values()
                for mems in cat_node.values()
                for mem in mems
            ]
        return [
            f"{category}: {', '.join(mems)}"
            for category, mems in memories.items()
        ]
    return list(memories)


@dataclass(slots=True)
class WorkPlan:
    """Structured view of pending benchmark work."""

    checkpoint: Checkpoint
    pending_work: list[WorkItem]
    completed: int
    total: int


def load_input_file(file_path: Path) -> list[InputEntry]:
    """Load entries from JSON or JSONL file."""
    if not file_path.exists():
        raise ValueError(f"Input file {file_path} does not exist")

    suffix = file_path.suffix.lower()
    if suffix not in (".json", ".jsonl"):
        raise ValueError(f"Input file must be JSON or JSONL (got {suffix})")

    with open(file_path, "r", encoding="utf-8") as f:
        if suffix == ".jsonl":
            entries = [json.loads(line) for line in f if line.strip()]
        else:
            entries = json.load(f)

    print(f"Loaded {len(entries)} rows from {file_path}")
    return entries


def load_and_validate_entries(input_file: Path) -> list[InputEntry]:
    """Load, validate, and deduplicate PersistBench entries from an input file."""
    raw_entries = load_input_file(input_file)
    input_entries: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()

    for i, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, dict):
            raise FatalBenchmarkError("Entry must be a dict")
        if "memories" not in raw_entry or "query" not in raw_entry:
            raise FatalBenchmarkError("Entry must have 'memories' and 'query' fields")

        raw_memories = raw_entry["memories"]
        query = raw_entry["query"]

        if not isinstance(raw_memories, (list, dict)):
            raise FatalBenchmarkError("'memories' must be a list or dict")
        if not isinstance(query, str) or not query.strip():
            raise FatalBenchmarkError("'query' must be a non-empty string")

        memories = _normalize_memories(raw_memories)
        hash_id = raw_entry.get("hash_id") or generate_hash_id(memories, query)

        failure_type = resolve_entry_configuration(raw_entry)
        validate_failure_type(failure_type)
        if hash_id in seen_hashes:
            continue

        seen_hashes.add(hash_id)
        entry_data = {
            "memories": memories,
            "query": query,
            "hash_id": hash_id,
            "original_index": i,
            "failure_type": failure_type,
        }
        if "full_memories" in raw_entry:
            entry_data["full_memories"] = raw_entry["full_memories"]
        if isinstance(raw_memories, dict):
            first = next(iter(raw_memories.values()), None)
            if isinstance(first, dict):
                # Two-level tree: store the full structure for tree-mode loading
                entry_data["tree_memories"] = raw_memories
            else:
                # One-level partitioned dict: store for partitioned_labeled mode
                entry_data["categorized_memories"] = {
                    category: list(items) for category, items in raw_memories.items()
                }
        input_entries.append(entry_data)

    if not input_entries:
        raise ValueError("No valid entries found")

    return input_entries


def samples_to_input_entries(samples: Iterable[Any]) -> list[InputEntry]:
    """Convert PersistBench Samples to InputEntry dicts for the benchmark pipeline."""
    from benchmark.dataset_loaders import Sample

    entries: list[InputEntry] = []
    for i, sample in enumerate(samples):
        assert isinstance(sample, Sample)
        entry: InputEntry = {
            "memories": sample.memories,
            "query": sample.prompt,
            "hash_id": sample.sample_id,
            "original_index": i,
            "failure_type": sample.metadata.get("failure_type", FAILURE_TYPE_CROSS_DOMAIN),
        }
        entries.append(entry)
    return entries


def ensure_entry_configuration(entry: dict[str, Any]) -> str:
    failure_type = entry.get("failure_type")

    if failure_type is None:
        failure_type = resolve_entry_configuration(entry)

    validate_failure_type(failure_type)
    entry["failure_type"] = failure_type

    return failure_type


def _hydrate_checkpoint_entry(
    checkpoint: Checkpoint,
    entry: InputEntry,
    ignore_config_mismatch: bool,
    output_file: Path,
) -> None:
    hash_id = entry["hash_id"]

    if hash_id not in checkpoint["entries"]:
        resolved_leak = ensure_entry_configuration(entry)
        entry_data: dict[str, Any] = {
            "memories": entry["memories"],
            "query": entry["query"],
            "results": {},
            "failure_type": resolved_leak,
        }
        if "full_memories" in entry:
            entry_data["full_memories"] = entry["full_memories"]
        checkpoint["entries"][hash_id] = entry_data
        return

    existing_entry = checkpoint["entries"][hash_id]
    existing_leak = ensure_entry_configuration(existing_entry)
    new_leak = ensure_entry_configuration(entry)

    if existing_leak != new_leak:
        if not ignore_config_mismatch:
            raise FatalBenchmarkError(
                f"Evaluation configuration changed for entry {hash_id}.\n"
                f"Checkpoint has: failure_type={existing_leak}\n"
                f"Input file has: failure_type={new_leak}\n"
                f"Cannot change evaluation config on resume. Either:\n"
                f"  1. Revert input file to original config, or\n"
                f"  2. Delete the checkpoint file at {output_file} to start fresh"
            )
        checkpoint["entries"][hash_id]["failure_type"] = new_leak


def _queue_generations_for_entry(
    checkpoint: Checkpoint,
    entry: InputEntry,
    models: list[ModelEntry],
    generations: int,
) -> tuple[list[WorkItem], int]:
    hash_id = entry["hash_id"]
    pending_work: list[WorkItem] = []
    completed_count = 0

    # model_affinity restricts which models process this entry (partitioned mode)
    model_affinity: set[str] | None = None
    stored_affinity = checkpoint["entries"][hash_id].get("model_affinity")
    if stored_affinity is not None:
        model_affinity = set(stored_affinity)
    elif "model_affinity" in entry:
        model_affinity = set(entry["model_affinity"])

    for model in models:
        model_name = model.name

        if model_affinity is not None and model_name not in model_affinity:
            continue  # This entry is not assigned to this model in partitioned mode

        if model_name not in checkpoint["entries"][hash_id]["results"]:
            checkpoint["entries"][hash_id]["results"][model_name] = {
                "generations": [],
            }

        for gen_idx in range(generations):
            status = get_generation_status(checkpoint, hash_id, model_name, gen_idx)
            if status == GenerationStatus.COMPLETED:
                completed_count += 1
            else:
                pending_work.append((entry, model, gen_idx))

    return pending_work, completed_count


def _build_work_queue(
    checkpoint: Checkpoint,
    input_entries: list[InputEntry],
    config: BenchmarkConfig,
    ignore_config_mismatch: bool,
) -> tuple[list[WorkItem], int]:
    pending_work: list[WorkItem] = []
    completed_count = 0

    for entry in input_entries:
        _hydrate_checkpoint_entry(
            checkpoint, entry, ignore_config_mismatch, config.output
        )
        entry_generations = get_generations_for_failure_type(
            entry["failure_type"], config.generations
        )
        entry_work, entry_completed = _queue_generations_for_entry(
            checkpoint, entry, config.models, entry_generations
        )
        pending_work.extend(entry_work)
        completed_count += entry_completed

    return pending_work, completed_count


def extract_entries_from_checkpoint(checkpoint: Checkpoint) -> list[InputEntry]:
    """Build InputEntry dicts from checkpoint entries for resume without original input file."""
    entries: list[InputEntry] = []
    for hash_id, entry_data in checkpoint.get("entries", {}).items():
        entry: InputEntry = {
            "memories": entry_data["memories"],
            "query": entry_data["query"],
            "hash_id": hash_id,
            "failure_type": entry_data.get("failure_type"),
        }
        if "full_memories" in entry_data:
            entry["full_memories"] = entry_data["full_memories"]
        entries.append(entry)
    return entries


def reconstruct_config(
    checkpoint: Checkpoint, checkpoint_path: Path
) -> BenchmarkConfig:
    """Rebuild BenchmarkConfig from checkpoint's stored config.

    Uses load_benchmark_config_data to ensure prompt templates are loaded
    and model names are validated, same as a fresh config load.
    Overrides output to point to the checkpoint file path (the file may have been moved).
    """
    stored_config = checkpoint.get("config")
    if stored_config is None:
        raise FatalBenchmarkError(
            f"Checkpoint {checkpoint_path} has no stored config. "
            f"This checkpoint was created before config-in-checkpoint support. "
            f"Please provide the original config file instead."
        )

    config = load_benchmark_config_data(
        dict(stored_config), config_path=checkpoint_path
    )
    config.output = checkpoint_path
    return config


def prepare_work_plan(
    input_entries: list[InputEntry],
    config: BenchmarkConfig,
    ignore_config_mismatch: bool = False,
    judge_provider: str | None = None,
    config_dict: dict[str, Any] | None = None,
    existing_checkpoint: Checkpoint | None = None,
) -> WorkPlan:
    """Initialize checkpoint, build work queue, and return planning summary."""
    checkpoint = initialize_checkpoint(
        input_entries,
        config,
        ignore_config_mismatch,
        judge_provider=judge_provider,
        config_dict=config_dict,
        existing_checkpoint=existing_checkpoint,
    )

    total_count = sum(
        get_generations_for_failure_type(e["failure_type"], config.generations)
        for e in input_entries
    ) * len(config.models)

    pending_work, completed_count = _build_work_queue(
        checkpoint, input_entries, config, ignore_config_mismatch
    )

    save_checkpoint(checkpoint, config.output)

    return WorkPlan(
        checkpoint=checkpoint,
        pending_work=pending_work,
        completed=completed_count,
        total=total_count,
    )
