#!/usr/bin/env python3
"""Backfill `full_memories` into existing RAG output checkpoints.

Why this exists
---------------
RAG runs (`outputs/persistbench/rag/persist_rag_tau*.json`) were produced with a
benchmark JSONL whose `memories` field had been overwritten with the
RAG-filtered subset. The judge then scored model outputs against that same
filtered subset, which is wrong: the judge needs the FULL memory pool to
correctly evaluate cross-domain leakage, beneficial usage, and sycophancy.

The code path now reads `entry.full_memories` for the judge when present and
falls back to `entry.memories` otherwise. This script populates that new
field on existing checkpoints by matching each entry's (query, failure_type)
against the baseline benchmark JSONL, which still has the complete memory
pool. Hash IDs are NOT touched, so existing model responses remain valid —
the only thing that needs to be redone is the judge call.

Usage
-----
    # Dry-run: show what would change, write nothing.
    uv run python analysis/persistbench/backfill_rag_full_memories.py --dry-run

    # Apply the backfill (adds full_memories field).
    uv run python analysis/persistbench/backfill_rag_full_memories.py

    # Apply AND clear judge fields so the next runner pass re-judges with the
    # full memory context. Generations themselves are preserved.
    uv run python analysis/persistbench/backfill_rag_full_memories.py --reset-judges

After running with --reset-judges, re-execute the existing RAG configs with
your normal runner. It will detect generations as NEEDS_JUDGE (response exists
but judge is None) and only pay for judge calls — no new generations.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
BASELINE_JSONL = REPO_ROOT / "benchmark_samples" / "persistbench" / "baseline" / "full_benchmark.jsonl"
RAG_OUTPUT_DIR = REPO_ROOT / "outputs" / "persistbench" / "rag"
RAG_BENCHMARK_DIR = REPO_ROOT / "benchmark_samples" / "persistbench" / "rag"


def load_baseline_index() -> dict[tuple[str, str], list[str]]:
    """Build (query, failure_type) -> full memories list from the baseline JSONL."""
    index: dict[tuple[str, str], list[str]] = {}
    with open(BASELINE_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ft = row.get("failure_type") or row.get("leakage_type")
            key = (row["query"], ft)
            if key in index:
                raise RuntimeError(f"Duplicate (query, failure_type) in baseline: {key!r}")
            index[key] = list(row["memories"])
    return index


def process_jsonl(
    path: pathlib.Path,
    baseline_index: dict[tuple[str, str], list[str]],
    *,
    dry_run: bool,
) -> dict[str, int]:
    """Add full_memories to each row in a benchmark JSONL file."""
    stats = {"rows_total": 0, "rows_already_have_full": 0, "rows_backfilled": 0,
             "rows_unmatched": 0, "rows_filter_not_subset": 0}
    new_lines: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                new_lines.append(line)
                continue
            row = json.loads(stripped)
            stats["rows_total"] += 1
            if "full_memories" in row:
                stats["rows_already_have_full"] += 1
            else:
                ft = row.get("failure_type") or row.get("leakage_type")
                key = (row["query"], ft)
                full = baseline_index.get(key)
                if full is None:
                    stats["rows_unmatched"] += 1
                    print(
                        f"  WARN: no baseline match for query={row['query'][:60]!r} "
                        f"failure_type={ft}", file=sys.stderr,
                    )
                else:
                    filtered_set = set(row.get("memories", []))
                    if not filtered_set.issubset(set(full)):
                        stats["rows_filter_not_subset"] += 1
                        print(
                            f"  WARN: filtered memories not subset of baseline for "
                            f"query={row['query'][:60]!r}", file=sys.stderr,
                        )
                    else:
                        row["full_memories"] = full
                        stats["rows_backfilled"] += 1
            new_lines.append(json.dumps(row, ensure_ascii=False) + "\n")

    if not dry_run and stats["rows_backfilled"]:
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

    return stats


def process_checkpoint(
    path: pathlib.Path,
    baseline_index: dict[tuple[str, str], list[str]],
    *,
    reset_judges: bool,
    dry_run: bool,
) -> dict[str, int]:
    stats = {
        "entries_total": 0,
        "entries_already_have_full": 0,
        "entries_backfilled": 0,
        "entries_unmatched": 0,
        "entries_filter_not_subset": 0,
        "judges_cleared": 0,
        "judges_already_empty": 0,
    }

    with open(path, encoding="utf-8") as f:
        checkpoint = json.load(f)

    for entry in checkpoint.get("entries", {}).values():
        stats["entries_total"] += 1
        if "full_memories" in entry:
            stats["entries_already_have_full"] += 1
        else:
            ft = entry.get("failure_type") or entry.get("leakage_type")
            key = (entry["query"], ft)
            full = baseline_index.get(key)
            if full is None:
                stats["entries_unmatched"] += 1
                print(
                    f"  WARN: no baseline match for query={entry['query'][:60]!r} "
                    f"failure_type={ft}",
                    file=sys.stderr,
                )
                continue

            filtered_set = set(entry.get("memories", []))
            full_set = set(full)
            if not filtered_set.issubset(full_set):
                stats["entries_filter_not_subset"] += 1
                missing = filtered_set - full_set
                print(
                    f"  WARN: filtered memories not a subset of baseline full set "
                    f"({len(missing)} extras) for query={entry['query'][:60]!r}",
                    file=sys.stderr,
                )
                continue

            entry["full_memories"] = full
            stats["entries_backfilled"] += 1

        if reset_judges:
            for model_data in entry.get("results", {}).values():
                for gen in model_data.get("generations", []):
                    if gen.get("memory_response") and gen.get("judge") is not None:
                        gen["judge"] = None
                        gen["error"] = None
                        stats["judges_cleared"] += 1
                    else:
                        stats["judges_already_empty"] += 1

    if not dry_run and (stats["entries_backfilled"] or stats["judges_cleared"]):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("Usage")[0])
    parser.add_argument(
        "--dry-run", action="store_true", help="Report changes without writing"
    )
    parser.add_argument(
        "--reset-judges",
        action="store_true",
        help="Also clear judge fields on each generation so the runner re-judges",
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=pathlib.Path,
        help="Specific checkpoint files (defaults to all RAG outputs in outputs/persistbench/rag/)",
    )
    args = parser.parse_args()

    if not BASELINE_JSONL.exists():
        print(f"ERROR: baseline JSONL not found at {BASELINE_JSONL}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading baseline index from {BASELINE_JSONL.relative_to(REPO_ROOT)}")
    baseline_index = load_baseline_index()
    print(f"  {len(baseline_index)} (query, failure_type) keys indexed\n")

    files = args.files or sorted(RAG_OUTPUT_DIR.glob("persist_rag_*.json"))
    if not files:
        print(f"No RAG output files found under {RAG_OUTPUT_DIR}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print("DRY RUN — no files will be modified\n")

    # ── Phase 1: patch the benchmark JSONL files. The runner reads these on
    # every fresh run and constructs InputEntries from them — the judge
    # ultimately reads `full_memories` off the InputEntry, so the JSONL is
    # the load-bearing place to put it. ─────────────────────────────────────
    jsonl_files = sorted(RAG_BENCHMARK_DIR.glob("persistbench_rag_tau*.jsonl"))
    if jsonl_files:
        print("Phase 1: patching benchmark JSONL files")
        for jpath in jsonl_files:
            rel = jpath.relative_to(REPO_ROOT)
            print(f"  {rel}")
            stats = process_jsonl(jpath, baseline_index, dry_run=args.dry_run)
            print(
                f"    rows={stats['rows_total']} backfilled={stats['rows_backfilled']} "
                f"already_had={stats['rows_already_have_full']} "
                f"unmatched={stats['rows_unmatched']} "
                f"non_subset={stats['rows_filter_not_subset']}"
            )
        print()
    else:
        print(f"No RAG benchmark JSONLs found under {RAG_BENCHMARK_DIR}\n")

    print("Phase 2: patching output checkpoints")
    grand_total = {k: 0 for k in (
        "entries_total", "entries_already_have_full", "entries_backfilled",
        "entries_unmatched", "entries_filter_not_subset",
        "judges_cleared", "judges_already_empty",
    )}

    for path in files:
        rel = path.relative_to(REPO_ROOT)
        print(f"Processing {rel}")
        stats = process_checkpoint(
            path, baseline_index, reset_judges=args.reset_judges, dry_run=args.dry_run
        )
        for k, v in stats.items():
            grand_total[k] += v
        print(
            f"  entries={stats['entries_total']} "
            f"backfilled={stats['entries_backfilled']} "
            f"already_had={stats['entries_already_have_full']} "
            f"unmatched={stats['entries_unmatched']} "
            f"non_subset={stats['entries_filter_not_subset']}"
        )
        if args.reset_judges:
            print(
                f"  judges_cleared={stats['judges_cleared']} "
                f"judges_already_empty={stats['judges_already_empty']}"
            )

    print()
    print("Totals across all files:")
    for k, v in grand_total.items():
        print(f"  {k}: {v}")
    if args.dry_run:
        print("\n(Dry run — re-run without --dry-run to apply.)")
    elif args.reset_judges:
        print(
            "\nNext step: re-run the existing RAG configs with the runner. It will "
            "detect cleared judges as NEEDS_JUDGE and pay only for judge calls."
        )


if __name__ == "__main__":
    main()
