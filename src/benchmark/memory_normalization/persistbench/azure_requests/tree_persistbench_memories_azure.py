#!/usr/bin/env python3
"""Build a two-level memory tree for each PersistBench sample — once per row.

Azure AI Foundry version — uses AsyncOpenAI with a Foundry endpoint instead of VertexAI.

Reads a baseline PersistBench JSONL file (memories as a flat list), then uses an
LLM to build a full two-level tree in two sequential calls per sample:

  Phase 1 — Tree skeleton: given the flat memory list and the 11 default
             top-level categories, generate up to 7 subcategories per category
             that reflect the actual memories.
  Phase 2 — Tree fill: assign every memory from the flat list to exactly one
             category → subcategory leaf, building the complete tree.

Each model in MODELS is processed sequentially and saved to its own subdirectory.

Output memories field shape:
  {
    "health": {
      "physical health": ["memory1", "memory2"],
      "mental health":   ["memory3"]
    },
    ...
  }

Usage:
  uv run python tree_persistbench_memories.py
  uv run python tree_persistbench_memories.py --input /path/to/input.jsonl

─── HOW TO EDIT ───────────────────────────────────────────────────────────────
  * Add / remove models                    ->  MODELS list below
  * Change the Azure endpoint / key env    ->  MODEL block below
  * Change temperature                     ->  MODEL block below
  * Change input/output paths              ->  RUN CONFIG below
  * Change concurrency or retry behaviour  ->  RUN CONFIG below
  * Change what the LLM is told to do      ->  PROMPTS block below
───────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

# ── MODEL ─────────────────────────────────────────────────────────────────────
MODELS = [
    "DeepSeek-V3.2", "gpt-oss-120b",                  # Azure deployment names
]
AZURE_ENDPOINT    = "https://algoverse-hakeem.services.ai.azure.com/openai/v1/"
AZURE_API_KEY_ENV = "AZURE_OPENAI_API_KEY"
TEMPERATURE       = 0.7
# ──────────────────────────────────────────────────────────────────────────────

# ── RUN CONFIG ────────────────────────────────────────────────────────────────
CONCURRENCY   = 10   # max simultaneous API requests
MAX_RETRIES   = 5    # retry attempts per API call on parse / API failure
MAX_SUBCATS   = 7    # maximum subcategories per top-level category
MAX_SAMPLES   = None # max number of samples to process (None = all)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent  # → repo root
INPUT_FILE    = _PROJECT_ROOT / "benchmark_samples/persistbench/baseline/full_benchmark.jsonl"
OUTPUT_DIR    = _PROJECT_ROOT / "benchmark_samples/persistbench/tree"
# Output per model: OUTPUT_DIR / <model_name> / full_benchmark.jsonl
# ──────────────────────────────────────────────────────────────────────────────

# ── CATEGORIES ────────────────────────────────────────────────────────────────
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
# ──────────────────────────────────────────────────────────────────────────────

# ── PROMPTS ───────────────────────────────────────────────────────────────────
SUBCATEGORY_SYSTEM_PROMPT = """\
You are a memory organizer. You will be given a flat list of a person's
memories. Your task is to propose subcategories for each of the 11 top-level
categories listed below so that any memory could later be placed into the
resulting two-level tree.

The 11 top-level categories are:
  health
  identity
  social
  romantic
  personal
  education
  employment
  finance
  housing
  legal
  schedule

Rules:
1. Each category must have between 1 and {max_subcats} subcategories.
2. Subcategory names must be short, descriptive, and lowercase (1–3 words).
3. Subcategory names must be unique within a category.
4. Base your subcategory choices on the actual memories provided — make them
   specific enough to be useful, not generic filler.
5. Do not include any memories in your response — only propose names.

Return ONLY a single-line JSON object where every key is one of the 11 category
names and every value is a list of subcategory name strings. Include all 11.

Example shape (values are illustrative only):
{{"health": ["physical health", "mental health", "medications"],
  "identity": ["core values", "religious beliefs"],
  "social": ["friendships", "family bonds", "community"],
  ...}}
""".format(max_subcats=MAX_SUBCATS)

FILL_SYSTEM_PROMPT = """\
You are a memory sorter. You will be given:
  1. A tree skeleton: an object mapping each of 11 categories to a list of
     subcategory names.
  2. A flat memory list: a JSON array of memory strings.

Your task is to assign every memory from the flat list to exactly one leaf in
the tree, choosing both the best top-level category and the best subcategory
within that category.

The 11 top-level categories are:
  health
  identity
  social
  romantic
  personal
  education
  employment
  finance
  housing
  legal
  schedule

Rules:
1. Every memory must appear in exactly one subcategory of exactly one category.
2. Do not drop or duplicate memories.
3. Do not modify the memory text.
4. If a memory could fit multiple categories, choose the most specific one.

Return ONLY a single-line JSON object that mirrors the tree skeleton but with
each subcategory mapped to a list of memory strings (may be empty []).

Example shape:
{{"health": {{"physical health": ["..."], "mental health": []}},
  "identity": {{"core values": ["...", "..."], "religious beliefs": []}},
  ...}}
"""
# ──────────────────────────────────────────────────────────────────────────────


# ── Internals (no need to edit below) ─────────────────────────────────────────

import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))  # → src/

from openai import AsyncOpenAI  # noqa: E402

from benchmark.utils import extract_json_from_response, generate_hash_id  # noqa: E402


def _output_file(model_name: str) -> Path:
    return OUTPUT_DIR / model_name / "full_benchmark.jsonl"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a two-level memory tree for PersistBench samples (Azure).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_FILE,
        help=f"Baseline PersistBench JSONL input file (default: {INPUT_FILE})",
    )
    return parser.parse_args()


def _load_checkpoint(output_file: Path) -> set[str]:
    """Return the set of queries already written to the output file."""
    done: set[str] = set()
    if output_file.exists():
        with open(output_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    done.add(row["query"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def _validate_skeleton(raw: dict) -> dict[str, list[str]]:
    """Ensure the LLM returned a valid skeleton with all 11 categories."""
    skeleton: dict[str, list[str]] = {}
    for cat in CATEGORIES:
        raw_subcats = raw.get(cat, [])
        if isinstance(raw_subcats, list):
            subcats = [s for s in raw_subcats if isinstance(s, str) and s.strip()][:MAX_SUBCATS]
        else:
            subcats = []
        if not subcats:
            subcats = [cat]
        seen: set[str] = set()
        unique: list[str] = []
        for s in subcats:
            if s not in seen:
                seen.add(s)
                unique.append(s)
        skeleton[cat] = unique
    return skeleton


def _validate_tree(
    memories: list[str],
    skeleton: dict[str, list[str]],
    raw: dict,
) -> dict[str, dict[str, list[str]]]:
    """Ensure every memory ends up in exactly one subcategory of some category."""
    memory_set = set(memories)
    placed: set[str] = set()

    tree: dict[str, dict[str, list[str]]] = {}
    for cat in CATEGORIES:
        subcats = skeleton[cat]
        raw_cat = raw.get(cat, {})
        cat_node: dict[str, list[str]] = {}
        for sub in subcats:
            mems = raw_cat.get(sub, []) if isinstance(raw_cat, dict) else []
            valid = [
                m for m in mems
                if isinstance(m, str) and m in memory_set and m not in placed
            ]
            for m in valid:
                placed.add(m)
            cat_node[sub] = valid
        tree[cat] = cat_node

    personal_first = skeleton["personal"][0]
    for mem in memories:
        if mem not in placed:
            tree["personal"][personal_first].append(mem)
            placed.add(mem)

    return tree


async def _generate_skeleton(
    client: AsyncOpenAI,
    model_name: str,
    memories: list[str],
    semaphore: asyncio.Semaphore,
) -> dict[str, list[str]]:
    """Phase 1: Ask the LLM to propose subcategories for each of the 11 categories."""
    user_message = json.dumps(memories, ensure_ascii=False)

    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": SUBCATEGORY_SYSTEM_PROMPT},
                        {"role": "user",   "content": user_message},
                    ],
                    temperature=TEMPERATURE,
                )
                content = response.choices[0].message.content or ""
                raw = extract_json_from_response(content)
                return _validate_skeleton(raw)

            except Exception as exc:
                if attempt == MAX_RETRIES - 1:
                    print(f"  [WARN] Skeleton: giving up after {MAX_RETRIES} attempts: {exc}")
                    return {cat: [cat] for cat in CATEGORIES}
                await asyncio.sleep(2 ** attempt)

    return {cat: [cat] for cat in CATEGORIES}


async def _fill_tree(
    client: AsyncOpenAI,
    model_name: str,
    memories: list[str],
    skeleton: dict[str, list[str]],
    semaphore: asyncio.Semaphore,
) -> dict[str, dict[str, list[str]]]:
    """Phase 2: Ask the LLM to assign flat memories into the skeleton tree."""
    user_message = json.dumps(
        {"tree_skeleton": skeleton, "memories": memories},
        ensure_ascii=False,
    )

    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": FILL_SYSTEM_PROMPT},
                        {"role": "user",   "content": user_message},
                    ],
                    temperature=TEMPERATURE,
                )
                content = response.choices[0].message.content or ""
                raw = extract_json_from_response(content)
                return _validate_tree(memories, skeleton, raw)

            except Exception as exc:
                if attempt == MAX_RETRIES - 1:
                    print(f"  [WARN] Fill: giving up after {MAX_RETRIES} attempts: {exc}")
                    personal_first = skeleton["personal"][0]
                    tree: dict[str, dict[str, list[str]]] = {
                        cat: {sub: [] for sub in skeleton[cat]}
                        for cat in CATEGORIES
                    }
                    tree["personal"][personal_first] = list(memories)
                    return tree
                await asyncio.sleep(2 ** attempt)

    personal_first = skeleton["personal"][0]
    tree = {cat: {sub: [] for sub in skeleton[cat]} for cat in CATEGORIES}
    tree["personal"][personal_first] = list(memories)
    return tree


async def _process_sample(
    raw_row: dict,
    client: AsyncOpenAI,
    model_name: str,
    semaphore: asyncio.Semaphore,
    out_file,
    lock: asyncio.Lock,
    counter: list[int],
    total: int,
) -> None:
    """Build a two-level memory tree for a single sample and write it out."""
    memories: list[str] = raw_row.get("memories", [])
    hash_id = raw_row.get("hash_id") or generate_hash_id(memories, raw_row["query"])

    if memories:
        skeleton = await _generate_skeleton(client, model_name, memories, semaphore)
        tree = await _fill_tree(client, model_name, memories, skeleton, semaphore)
    else:
        skeleton = {cat: [cat] for cat in CATEGORIES}
        tree = {cat: {cat: []} for cat in CATEGORIES}

    out_row = {
        "query": raw_row["query"],
        "memories": tree,
        "memory_domain": raw_row.get("memory_domain", ""),
        "query_domain": raw_row.get("query_domain", ""),
        "failure_type": raw_row.get("failure_type", ""),
        "hash_id": hash_id,
    }

    async with lock:
        out_file.write(json.dumps(out_row, ensure_ascii=False) + "\n")
        out_file.flush()
        counter[0] += 1
        print(f"  [{counter[0]}/{total}] {raw_row['query'][:70]}...")


async def _run_model(
    model_name: str,
    raw_rows: list[dict],
    client: AsyncOpenAI,
) -> None:
    """Process all samples for a single model."""
    output_file = _output_file(model_name)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    done_queries = _load_checkpoint(output_file)
    pending = [r for r in raw_rows if r["query"] not in done_queries]
    print(f"  Checkpoint: {len(done_queries)} rows written | Remaining: {len(pending)}")

    if not pending:
        print("  All samples already processed.")
        return

    total = len(raw_rows)
    semaphore = asyncio.Semaphore(CONCURRENCY)
    lock = asyncio.Lock()
    counter = [len(done_queries)]

    print(f"  Building memory trees for {len(pending)} sample(s) ...")
    with open(output_file, "a", encoding="utf-8") as out_file:
        tasks = [
            _process_sample(
                row, client, model_name, semaphore, out_file, lock, counter, total
            )
            for row in pending
        ]
        await asyncio.gather(*tasks)

    print(f"  {counter[0]} rows saved to {output_file}")


async def main() -> None:
    api_key = os.environ.get(AZURE_API_KEY_ENV)
    if not api_key:
        raise SystemExit(f"Error: {AZURE_API_KEY_ENV} environment variable not set")

    args = _parse_args()
    input_file: Path = args.input

    print(f"Loading PersistBench dataset from {input_file} ...")
    raw_rows: list[dict] = []
    with open(input_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_rows.append(json.loads(line))
    print(f"Loaded {len(raw_rows)} rows")

    if MAX_SAMPLES is not None:
        raw_rows = raw_rows[:MAX_SAMPLES]
        print(f"Limit: capped to {MAX_SAMPLES} sample(s)")

    async with AsyncOpenAI(base_url=AZURE_ENDPOINT, api_key=api_key) as client:
        for model_name in MODELS:
            print(f"\n{'─' * 60}")
            print(f"Model: {model_name}")
            print(f"{'─' * 60}")
            await _run_model(model_name, raw_rows, client)

    print("\nAll models complete.")


if __name__ == "__main__":
    asyncio.run(main())
