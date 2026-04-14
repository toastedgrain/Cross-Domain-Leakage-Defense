#!/usr/bin/env python3
"""Partition CIM memories into 11 categories — once per persona.

Reads samples from the facebook/CIMemories HuggingFace dataset (full_profile
mode), partitions each *persona's* flat memory list into 11 categories using an
LLM (one LLM call per persona, not per sample), and writes a JSONL file
compatible with the benchmark runner's partitioned mode.

Each output row preserves all CIM-specific fields (required_attributes,
forbidden_attributes, cim_metadata) so the judge can evaluate properly.

Usage:
  # Partition specific personas only
  uv run python partition_cim_memories.py --personas "Jeffery Day" "Shawn Franklin"

  # Partition all personas
  uv run python partition_cim_memories.py

─── HOW TO EDIT ───────────────────────────────────────────────────────────────
  * Change the model / location / temperature  ->  MODEL block below
  * Change input/output paths                  ->  RUN CONFIG below
  * Change concurrency or retry behaviour      ->  RUN CONFIG below
  * Change what the LLM is told to do          ->  SYSTEM_PROMPT below
───────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import os

os.environ.setdefault(
    "VERTEXAI_SERVICE_ACCOUNT_PATH",
    str(Path.home() / "Downloads" / "VERTEXAI_SERVICE_ACCOUNT.json"),
)

# ── MODEL ─────────────────────────────────────────────────────────────────────
MODEL_NAME     = "meta/llama-3.3-70b-instruct-maas"
MODEL_LOCATION = "us-central1"   # VertexAI region where the model is deployed
TEMPERATURE    = 0
# ──────────────────────────────────────────────────────────────────────────────

# ── RUN CONFIG ────────────────────────────────────────────────────────────────
CONCURRENCY     = 7    # max simultaneous API requests
MAX_RETRIES     = 5    # retry attempts per sample on parse / API failure
MAX_PERSONAS    = None  # max number of personas to process (None = all)
_PROJECT_ROOT   = Path(__file__).resolve().parent.parent.parent.parent.parent  # repo root
CIM_INPUT_FILE = _PROJECT_ROOT / "benchmark_samples" / "CIM" / "baseline" / "cim_normalized.jsonl"
OUTPUT_FILE    = _PROJECT_ROOT / "cim_partitioned_llama3p3.jsonl"
# ──────────────────────────────────────────────────────────────────────────────

# ── PROMPT ────────────────────────────────────────────────────────────────────
# Same 11 categories as partition_memories.py for consistency across datasets.
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

SYSTEM_PROMPT = """\
You are a memory classifier. Your task is to sort a list of personal memories
into exactly one of the following categories:

health       – physical or mental health, medical conditions, treatments, medications, fitness, therapy
identity     – core personal identity traits such as nationality, religion, gender identity, values, beliefs
social       – non-romantic relationships and interactions with friends, family, acquaintances, or colleagues
romantic     – intimate or romantic relationships including dating, partners, marriage, attraction, breakups
personal     – hobbies, preferences, lifestyle choices, personality traits, interests
education    – schooling, degrees, courses, academic history, tutoring, learning experiences
employment   – jobs, work history, workplace experiences, colleagues, professional skills
finance      – money, savings, income, expenses, debt, investments, banking, taxes
housing      – home, residence, living situation, roommates, neighbors, rent, mortgage
legal        – legal issues, contracts, court matters, rights, criminal record, official documents
schedule     – appointments, routines, recurring events, time-based plans, daily habits

Rules:
1. Each memory must appear in exactly one category.
2. Do not drop or duplicate memories.
3. If a memory could fit multiple categories, choose the most specific category.
4. Categories with no memories must contain an empty list [].
5. Do not modify the memory text.

Return ONLY a single-line JSON object with the following keys in this exact order:

{"health": [...], "identity": [...], "social": [...], "romantic": [...], "personal": [...], "education": [...], "employment": [...], "finance": [...], "housing": [...], "legal": [...], "schedule": [...]}
"""
# ──────────────────────────────────────────────────────────────────────────────


# ── Internals (no need to edit below) ─────────────────────────────────────────

import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))  # add src/ so 'benchmark' is importable

from benchmark.utils import extract_json_from_response, get_vertex_ai_client  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Partition CIM memories into 11 categories (once per persona).",
    )
    parser.add_argument(
        "--personas",
        nargs="+",
        default=None,
        help='Persona names to process (e.g. --personas "Jeffery Day" "Shawn Franklin"). '
             "Omit to process all personas.",
    )
    return parser.parse_args()


def _row_key(row: dict) -> str:
    """Unique key for a single output row: name + task + recipient."""
    return f"{row.get('name', '')}|{row.get('task', '')}|{row.get('recipient', '')}"


def _load_checkpoint() -> tuple[set[str], dict[str, dict[str, list[str]]]]:
    """Return (done_keys, persona_partitions) from the existing output file.

    done_keys are 'name|task|recipient' strings.
    persona_partitions maps persona name → recovered 11-category dict.
    """
    done: set[str] = set()
    persona_partitions: dict[str, dict[str, list[str]]] = {}

    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    done.add(_row_key(row))
                    name = row.get("name")
                    if name and name not in persona_partitions:
                        persona_partitions[name] = row["memories"]
                except (json.JSONDecodeError, KeyError):
                    pass

    return done, persona_partitions


def _validate_partition(
    memories: list[str], raw: dict
) -> dict[str, list[str]]:
    """Ensure every input memory appears exactly once in the result.

    Accepts whatever the model returned for each category, then appends any
    memories the model missed to 'personal' as a safe fallback.
    """
    result: dict[str, list[str]] = {cat: [] for cat in CATEGORIES}
    placed: set[str] = set()

    for cat in CATEGORIES:
        for mem in raw.get(cat, []):
            if mem in memories and mem not in placed:
                result[cat].append(mem)
                placed.add(mem)

    # Fallback: anything the model missed goes to 'personal'
    for mem in memories:
        if mem not in placed:
            result["personal"].append(mem)

    return result


async def _classify(
    client,
    memories: list[str],
    semaphore: asyncio.Semaphore,
) -> dict[str, list[str]]:
    """Call the LLM and return a validated 11-category partition."""
    user_message = json.dumps(memories, ensure_ascii=False)

    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_message},
                    ],
                    temperature=TEMPERATURE,
                )
                content = response.choices[0].message.content or ""
                raw = extract_json_from_response(content)
                return _validate_partition(memories, raw)

            except Exception as exc:
                if attempt == MAX_RETRIES - 1:
                    print(f"  [WARN] Giving up after {MAX_RETRIES} attempts: {exc}")
                    fallback = {cat: [] for cat in CATEGORIES}
                    fallback["personal"] = list(memories)
                    return fallback
                await asyncio.sleep(2**attempt)

    # unreachable, but satisfies type checker
    fallback = {cat: [] for cat in CATEGORIES}
    fallback["personal"] = list(memories)
    return fallback


def _make_hash_id(name: str, task: str, recipient: str) -> str:
    """Stable hash matching cim_normalize.py — keyed on name + task + recipient."""
    return hashlib.md5(
        json.dumps(
            {"name": name, "task": task, "recipient": recipient},
            ensure_ascii=False,
            sort_keys=True,
        ).encode()
    ).hexdigest()


def _build_sample_row(raw_row: dict, partition: dict[str, list[str]]) -> dict:
    """Build the output JSONL row for a single sample.

    Mirrors the cim_normalized.jsonl format exactly, replacing the flat
    'memories' list with the 11-category partition dict.
    hash_id uses the same formula as cim_normalize.py so baseline and
    partitioned files share identical hashes for matching rows.
    """
    name = raw_row["name"]
    task = raw_row["task"]
    recipient = raw_row["recipient"]
    return {
        "name": name,
        "query": raw_row["query"],
        "memories": partition,
        "task": task,
        "recipient": recipient,
        "attribute_memory_map": raw_row.get("attribute_memory_map", {}),
        "required_attributes": raw_row.get("required_attributes", []),
        "forbidden_attributes": raw_row.get("forbidden_attributes", []),
        "hash_id": raw_row.get("hash_id") or _make_hash_id(name, task, recipient),
    }


async def main() -> None:
    args = _parse_args()

    # ── Load raw rows directly from normalized JSONL ─────────────────────────
    print(f"Loading CIM dataset from {CIM_INPUT_FILE} ...")
    raw_rows: list[dict] = []
    with open(CIM_INPUT_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_rows.append(json.loads(line))
    print(f"Loaded {len(raw_rows)} CIM rows")

    # ── Group rows by persona ────────────────────────────────────────────────
    persona_rows: dict[str, list[dict]] = defaultdict(list)
    for row in raw_rows:
        persona_rows[row["name"]].append(row)

    # Apply persona filter from CLI
    if args.personas is not None:
        requested = set(args.personas)
        available = set(persona_rows.keys())
        unknown = requested - available
        if unknown:
            print(f"[WARN] Unknown persona(s): {unknown}")
            print(f"       Available: {sorted(available)}")
        persona_rows = {
            name: rows for name, rows in persona_rows.items()
            if name in requested
        }

    if not persona_rows:
        print("No personas to process.")
        return

    if MAX_PERSONAS is not None:
        persona_rows = dict(list(persona_rows.items())[:MAX_PERSONAS])
        print(f"Limit: capped to {MAX_PERSONAS} persona(s)")

    total_rows = sum(len(rows) for rows in persona_rows.values())
    print(
        f"Processing {len(persona_rows)} persona(s), "
        f"{total_rows} rows total"
    )

    # ── Resume support ───────────────────────────────────────────────────────
    done_keys, persona_partitions = _load_checkpoint()
    print(f"Checkpoint: {len(done_keys)} rows written, "
          f"{len(persona_partitions)} persona partition(s) recovered")

    # ── Phase 1: Classify once per persona ───────────────────────────────────
    personas_to_classify = [
        name for name in persona_rows
        if name not in persona_partitions
    ]

    if personas_to_classify:
        print(f"\nPhase 1: Classifying {len(personas_to_classify)} persona(s) ...")
        semaphore = asyncio.Semaphore(CONCURRENCY)

        async with get_vertex_ai_client(MODEL_LOCATION) as client:
            classify_tasks = []
            for name in personas_to_classify:
                # All rows for the same persona share identical memories,
                # so we use the first row's flat memory list.
                memories = persona_rows[name][0]["memories"]
                classify_tasks.append(_classify(client, memories, semaphore))

            results = await asyncio.gather(*classify_tasks)

            for name, partition in zip(personas_to_classify, results):
                persona_partitions[name] = partition
                mem_count = sum(len(v) for v in partition.values())
                print(f"  {name}: {mem_count} memories classified")
    else:
        print("\nPhase 1: All persona partitions recovered from checkpoint.")

    # ── Phase 2: Write sample rows ───────────────────────────────────────────
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    counter = len(done_keys)

    print(f"\nPhase 2: Writing sample rows ...")
    with open(OUTPUT_FILE, "a", encoding="utf-8") as out_file:
        for name, rows in persona_rows.items():
            partition = persona_partitions[name]
            for raw_row in rows:
                if _row_key(raw_row) in done_keys:
                    continue
                out_row = _build_sample_row(raw_row, partition)
                out_file.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                out_file.flush()
                counter += 1
                task_preview = raw_row.get("task", "")[:50]
                print(f"[{counter}/{total_rows}] {name}: {task_preview}...")

    print(f"\nDone! {counter} rows saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
