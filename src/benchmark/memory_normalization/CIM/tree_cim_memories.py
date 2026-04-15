#!/usr/bin/env python3
"""Build a two-level memory tree for each CIM persona — once per persona.

Reads a normalized CIM JSONL file (memories as a flat list), then uses an LLM
to build a full two-level tree in two sequential calls per persona:

  Phase 1 — Tree skeleton: given the flat memory list and the 11 default
             top-level categories, generate up to 7 subcategories per category
             that reflect the actual memories.
  Phase 2 — Tree fill: assign every memory from the flat list to exactly one
             category → subcategory leaf, building the complete tree.

Output memories field shape (easy to inject as a tree prompt):
  {
    "health": {
      "physical health": ["memory1", "memory2"],
      "mental health":   ["memory3"]
    },
    "identity": {
      "religious beliefs": ["memory4"]
    },
    ...
  }

Usage:
  # Process specific personas only
  uv run python tree_cim_memories.py --personas "Jeffery Day" "Shawn Franklin"

  # Process all personas
  uv run python tree_cim_memories.py

─── HOW TO EDIT ───────────────────────────────────────────────────────────────
  * Change the model / location / temperature  ->  MODEL block below
  * Change input/output paths                  ->  RUN CONFIG below
  * Change concurrency or retry behaviour      ->  RUN CONFIG below
  * Change what the LLM is told to do          ->  PROMPTS block below
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
TEMPERATURE    = 0.7
# ──────────────────────────────────────────────────────────────────────────────

# ── RUN CONFIG ────────────────────────────────────────────────────────────────
CONCURRENCY     = 7    # max simultaneous API requests
MAX_RETRIES     = 5    # retry attempts per API call on parse / API failure
MAX_SUBCATS     = 7    # maximum subcategories per top-level category
MAX_PERSONAS    = None  # max number of personas to process (None = all)
_PROJECT_ROOT   = Path(__file__).resolve().parent.parent.parent.parent.parent  # repo root
CIM_INPUT_FILE  = _PROJECT_ROOT / "benchmark_samples" / "CIM" / "baseline" / "cim_normalized.jsonl"
OUTPUT_FILE     = _PROJECT_ROOT / "benchmark_samples" / "CIM" / "tree" / "cim_tree_llama3p3.jsonl"
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
1. Each category must have between 3 and {max_subcats} subcategories.
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
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))  # add src/ so 'benchmark' is importable

from benchmark.utils import extract_json_from_response, get_vertex_ai_client  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a two-level memory tree for CIM personas (once per persona).",
    )
    parser.add_argument(
        "--personas",
        nargs="+",
        default=None,
        help='Persona names to process (e.g. --personas "Jeffery Day" "Shawn Franklin"). '
             "Omit to process all personas.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=CIM_INPUT_FILE,
        help=f"Partitioned CIM JSONL input file (default: {CIM_INPUT_FILE})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FILE,
        help=f"Output JSONL file (default: {OUTPUT_FILE})",
    )
    return parser.parse_args()


def _row_key(row: dict) -> str:
    """Unique key for a single output row: name + task + recipient."""
    return f"{row.get('name', '')}|{row.get('task', '')}|{row.get('recipient', '')}"


def _load_checkpoint(output_file: Path) -> tuple[set[str], dict[str, dict]]:
    """Return (done_keys, persona_trees) from the existing output file.

    done_keys are 'name|task|recipient' strings.
    persona_trees maps persona name → recovered two-level tree dict.
    """
    done: set[str] = set()
    persona_trees: dict[str, dict] = {}

    if output_file.exists():
        with open(output_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    done.add(_row_key(row))
                    name = row.get("name")
                    if name and name not in persona_trees:
                        persona_trees[name] = row["memories"]
                except (json.JSONDecodeError, KeyError):
                    pass

    return done, persona_trees


def _validate_skeleton(raw: dict) -> dict[str, list[str]]:
    """Ensure the LLM returned a valid skeleton with all 11 categories.

    Fills missing categories with a single generic subcategory. Trims each
    category to at most MAX_SUBCATS subcategories.
    """
    skeleton: dict[str, list[str]] = {}
    for cat in CATEGORIES:
        raw_subcats = raw.get(cat, [])
        if isinstance(raw_subcats, list):
            # Keep only string entries and cap at MAX_SUBCATS
            subcats = [s for s in raw_subcats if isinstance(s, str) and s.strip()][:MAX_SUBCATS]
        else:
            subcats = []
        if not subcats:
            subcats = [cat]   # fallback: one generic subcategory named after the category
        # Deduplicate while preserving order
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
    """Ensure every memory ends up in exactly one subcategory of some category.

    Iterates through the raw model output collecting valid memory strings from
    the flat list.  Memories the model missed or duplicated are appended to the
    'personal' category's first subcategory as a safe fallback.
    """
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

    # Fallback: memories the model missed go into 'personal' first subcategory
    personal_first = skeleton["personal"][0]
    for mem in memories:
        if mem not in placed:
            tree["personal"][personal_first].append(mem)
            placed.add(mem)

    return tree


async def _generate_skeleton(
    client,
    memories: list[str],
    semaphore: asyncio.Semaphore,
) -> dict[str, list[str]]:
    """Phase 1: Ask the LLM to propose subcategories for each of the 11 categories."""
    user_message = json.dumps(memories, ensure_ascii=False)

    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
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
                    # Fallback: one generic subcategory per category
                    return {cat: [cat] for cat in CATEGORIES}
                await asyncio.sleep(2 ** attempt)

    return {cat: [cat] for cat in CATEGORIES}


async def _fill_tree(
    client,
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
                    model=MODEL_NAME,
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
                    # Fallback: dump all memories into 'personal' first subcategory
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


def _make_hash_id(name: str, task: str, recipient: str) -> str:
    """Stable hash keyed on name + task + recipient — matches cim_normalize.py."""
    return hashlib.md5(
        json.dumps(
            {"name": name, "task": task, "recipient": recipient},
            ensure_ascii=False,
            sort_keys=True,
        ).encode()
    ).hexdigest()


def _build_sample_row(raw_row: dict, tree: dict[str, dict[str, list[str]]]) -> dict:
    """Build the output JSONL row for a single sample.

    Mirrors the cim_partitioned format exactly, replacing the flat category
    dict with the two-level tree dict in the 'memories' field.
    """
    name = raw_row["name"]
    task = raw_row["task"]
    recipient = raw_row["recipient"]
    return {
        "name": name,
        "query": raw_row["query"],
        "memories": tree,
        "task": task,
        "recipient": recipient,
        "attribute_memory_map": raw_row.get("attribute_memory_map", {}),
        "required_attributes": raw_row.get("required_attributes", []),
        "forbidden_attributes": raw_row.get("forbidden_attributes", []),
        "hash_id": raw_row.get("hash_id") or _make_hash_id(name, task, recipient),
    }


async def main() -> None:
    args = _parse_args()
    input_file: Path = args.input
    output_file: Path = args.output

    # ── Load normalized rows ─────────────────────────────────────────────────
    print(f"Loading normalized CIM dataset from {input_file} ...")
    raw_rows: list[dict] = []
    with open(input_file, encoding="utf-8") as f:
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
    done_keys, persona_trees = _load_checkpoint(output_file)
    print(
        f"Checkpoint: {len(done_keys)} rows written, "
        f"{len(persona_trees)} persona tree(s) recovered"
    )

    # ── Phase 1 & 2: Build and fill tree once per persona ───────────────────
    personas_to_build = [
        name for name in persona_rows
        if name not in persona_trees
    ]

    if personas_to_build:
        print(f"\nPhase 1+2: Building memory trees for {len(personas_to_build)} persona(s) ...")
        semaphore = asyncio.Semaphore(CONCURRENCY)

        async with get_vertex_ai_client(MODEL_LOCATION) as client:

            # Phase 1: generate skeletons for all personas concurrently
            print("  Phase 1: Generating subcategory skeletons ...")
            skeleton_tasks = []
            for name in personas_to_build:
                # All rows for the same persona share identical flat memory lists
                memories = persona_rows[name][0]["memories"]
                skeleton_tasks.append(
                    _generate_skeleton(client, memories, semaphore)
                )

            skeletons = await asyncio.gather(*skeleton_tasks)

            for name, skeleton in zip(personas_to_build, skeletons):
                total_subcats = sum(len(v) for v in skeleton.values())
                print(f"    {name}: {total_subcats} subcategories across 11 categories")

            # Phase 2: fill trees for all personas concurrently
            print("  Phase 2: Filling trees with memories ...")
            fill_tasks = []
            for name, skeleton in zip(personas_to_build, skeletons):
                memories = persona_rows[name][0]["memories"]
                fill_tasks.append(
                    _fill_tree(client, memories, skeleton, semaphore)
                )

            trees = await asyncio.gather(*fill_tasks)

            for name, tree in zip(personas_to_build, trees):
                persona_trees[name] = tree
                mem_count = sum(
                    len(mems)
                    for cat_node in tree.values()
                    for mems in cat_node.values()
                )
                print(f"    {name}: {mem_count} memories placed in tree")
    else:
        print("\nPhase 1+2: All persona trees recovered from checkpoint.")

    # ── Phase 3: Write sample rows ───────────────────────────────────────────
    output_file.parent.mkdir(parents=True, exist_ok=True)
    counter = len(done_keys)

    print(f"\nPhase 3: Writing sample rows ...")
    with open(output_file, "a", encoding="utf-8") as out_file:
        for name, rows in persona_rows.items():
            tree = persona_trees[name]
            for raw_row in rows:
                if _row_key(raw_row) in done_keys:
                    continue
                out_row = _build_sample_row(raw_row, tree)
                out_file.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                out_file.flush()
                counter += 1
                task_preview = raw_row.get("task", "")[:50]
                print(f"[{counter}/{total_rows}] {name}: {task_preview}...")

    print(f"\nDone! {counter} rows saved to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
