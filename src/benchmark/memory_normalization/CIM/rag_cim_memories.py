#!/usr/bin/env python3
"""Build RAG-filtered CIM memory files — retrieve top-k memories per row.

Reads the normalized CIM JSONL (flat memory list per persona), embeds all
memories and each row's task query, then keeps only the top-k most relevant
memories per row based on cosine similarity.

The output JSONL has the same schema as cim_normalized.jsonl (memories is a
flat list[str]) so the benchmark runner treats it as a baseline run — no
method flag needed.

Supports two embedding providers:
  - vertexai  : Vertex AI text-embedding-005 (default, uses service account)
  - openai    : OpenAI text-embedding-3-small (uses OPENAI_API_KEY)

Usage:
  # Default: k=20, Vertex AI embeddings
  uv run python rag_cim_memories.py

  # Custom k, OpenAI embeddings
  uv run python rag_cim_memories.py --k 10 --provider openai

  # Multiple k values at once
  uv run python rag_cim_memories.py --k 10 20 30

  # Specific personas only
  uv run python rag_cim_memories.py --personas "Jeffery Day" "Shawn Franklin"

─── HOW TO EDIT ───────────────────────────────────────────────────────────────
  * Change embedding model             ->  EMBEDDING CONFIG below
  * Change input/output paths          ->  RUN CONFIG below
  * Change concurrency or retry        ->  RUN CONFIG below
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
import sys

import numpy as np

os.environ.setdefault(
    "VERTEXAI_SERVICE_ACCOUNT_PATH",
    str(Path.home() / "Downloads" / "VERTEXAI_SERVICE_ACCOUNT.json"),
)

# ── EMBEDDING CONFIG ─────────────────────────────────────────────────────────
VERTEX_EMBEDDING_MODEL = "text-embedding-005"
VERTEX_EMBEDDING_LOCATION = "us-central1"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
# ──────────────────────────────────────────────────────────────────────────────

# ── RUN CONFIG ───────────────────────────────────────────────────────────────
DEFAULT_K = 20
EMBED_BATCH_SIZE = 100  # texts per embedding API call
CONCURRENCY = 5
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
CIM_INPUT_FILE = _PROJECT_ROOT / "benchmark_samples" / "cim" / "baseline" / "cim_normalized.jsonl"
OUTPUT_DIR = _PROJECT_ROOT / "benchmark_samples" / "cim" / "rag"
# ──────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from benchmark.utils import get_vertex_credentials, get_vertex_project_id  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build RAG-filtered CIM memory files (top-k by embedding similarity).",
    )
    parser.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=[DEFAULT_K],
        help=f"Number of memories to retrieve per row (default: {DEFAULT_K}). "
             "Pass multiple values to generate multiple output files.",
    )
    parser.add_argument(
        "--provider",
        choices=["vertexai", "openai"],
        default="vertexai",
        help="Embedding provider (default: vertexai).",
    )
    parser.add_argument(
        "--personas",
        nargs="+",
        default=None,
        help='Persona names to process (e.g. --personas "Jeffery Day"). Omit for all.',
    )
    return parser.parse_args()


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between vector a and matrix b (one score per row of b)."""
    a_norm = a / (np.linalg.norm(a) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return b_norm @ a_norm


class _EmbeddingClient:
    """Unified async embedding interface for Vertex AI and OpenAI."""

    def __init__(self, provider: str) -> None:
        self._provider = provider
        if provider == "vertexai":
            from google import genai
            credentials = get_vertex_credentials()
            self._genai = genai.Client(
                vertexai=True,
                credentials=credentials,
                project=get_vertex_project_id(),
                location=VERTEX_EMBEDDING_LOCATION,
            )
            self._model = VERTEX_EMBEDDING_MODEL
        else:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                print("ERROR: OPENAI_API_KEY not set")
                sys.exit(1)
            self._openai = AsyncOpenAI(api_key=api_key)
            self._model = OPENAI_EMBEDDING_MODEL

    async def embed_batch(
        self, texts: list[str], semaphore: asyncio.Semaphore
    ) -> list[list[float]]:
        async with semaphore:
            if self._provider == "vertexai":
                response = await asyncio.to_thread(
                    self._genai.models.embed_content,
                    model=self._model,
                    contents=texts,
                )
                return [list(e.values) for e in response.embeddings]
            else:
                response = await self._openai.embeddings.create(
                    model=self._model, input=texts, encoding_format="float"
                )
                return [item.embedding for item in response.data]


async def _embed_all(
    client: _EmbeddingClient,
    texts: list[str],
    semaphore: asyncio.Semaphore,
    desc: str = "",
) -> np.ndarray:
    """Embed all texts in batches, returns (N, dim) numpy array."""
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        batch_embeddings = await client.embed_batch(batch, semaphore)
        all_embeddings.extend(batch_embeddings)
        if desc:
            print(f"  {desc}: {min(i + EMBED_BATCH_SIZE, len(texts))}/{len(texts)}")
    return np.array(all_embeddings)


def _build_query_text(row: dict) -> str:
    """Build the query string used for retrieval similarity."""
    task = row.get("task", "")
    recipient = row.get("recipient", "")
    return f"Write a complete message to {recipient} to achieve the following purpose: {task}"


def _make_hash_id(name: str, task: str, recipient: str) -> str:
    """Stable hash matching cim_normalize.py."""
    return hashlib.md5(
        json.dumps(
            {"name": name, "task": task, "recipient": recipient},
            ensure_ascii=False,
            sort_keys=True,
        ).encode()
    ).hexdigest()


def _build_output_row(raw_row: dict, filtered_memories: list[str]) -> dict:
    """Build output JSONL row with filtered memories."""
    return {
        "name": raw_row["name"],
        "query": raw_row["query"],
        "memories": filtered_memories,
        "task": raw_row.get("task", ""),
        "recipient": raw_row.get("recipient", ""),
        "attribute_memory_map": raw_row.get("attribute_memory_map", {}),
        "required_attributes": raw_row.get("required_attributes", []),
        "forbidden_attributes": raw_row.get("forbidden_attributes", []),
        "hash_id": raw_row.get("hash_id") or _make_hash_id(
            raw_row["name"], raw_row.get("task", ""), raw_row.get("recipient", "")
        ),
    }


async def main() -> None:
    args = _parse_args()

    # ── Load rows ────────────────────────────────────────────────────────────
    print(f"Loading CIM dataset from {CIM_INPUT_FILE} ...")
    raw_rows: list[dict] = []
    with open(CIM_INPUT_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_rows.append(json.loads(line))
    print(f"Loaded {len(raw_rows)} rows")

    # ── Filter personas ──────────────────────────────────────────────────────
    if args.personas:
        requested = set(args.personas)
        raw_rows = [r for r in raw_rows if r["name"] in requested]
        print(f"Filtered to {len(raw_rows)} rows for personas: {args.personas}")

    if not raw_rows:
        print("No rows to process.")
        return

    # ── Group by persona (all rows for a persona share the same memories) ───
    persona_rows: dict[str, list[dict]] = defaultdict(list)
    for row in raw_rows:
        persona_rows[row["name"]].append(row)

    # ── Phase 1: Embed all unique persona memories ───────────────────────────
    print(f"\nPhase 1: Embedding memories for {len(persona_rows)} persona(s) ...")
    client = _EmbeddingClient(args.provider)
    semaphore = asyncio.Semaphore(CONCURRENCY)

    persona_memories: dict[str, list[str]] = {}
    persona_memory_embeddings: dict[str, np.ndarray] = {}

    for persona_name, rows in persona_rows.items():
        memories = rows[0]["memories"]
        persona_memories[persona_name] = memories
        print(f"  Embedding {len(memories)} memories for {persona_name} ...")
        embeddings = await _embed_all(
            client, memories, semaphore, desc=persona_name
        )
        persona_memory_embeddings[persona_name] = embeddings

    # ── Phase 2: Embed all queries ───────────────────────────────────────────
    print(f"\nPhase 2: Embedding {len(raw_rows)} queries ...")
    query_texts = [_build_query_text(row) for row in raw_rows]
    query_embeddings = await _embed_all(
        client, query_texts, semaphore, desc="queries"
    )

    # ── Phase 3: For each k, select top-k and write output ───────────────────
    for k in args.k:
        output_file = OUTPUT_DIR / f"cim_rag_k{k}.jsonl"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        print(f"\nPhase 3: Selecting top-{k} memories and writing {output_file} ...")
        counter = 0

        with open(output_file, "w", encoding="utf-8") as out:
            for i, raw_row in enumerate(raw_rows):
                persona_name = raw_row["name"]
                memories = persona_memories[persona_name]
                mem_embs = persona_memory_embeddings[persona_name]
                query_emb = query_embeddings[i]

                similarities = _cosine_similarity(query_emb, mem_embs)

                actual_k = min(k, len(memories))
                top_indices = np.argsort(similarities)[-actual_k:][::-1]
                filtered = [memories[idx] for idx in top_indices]

                out_row = _build_output_row(raw_row, filtered)
                out.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                counter += 1

        print(f"  Wrote {counter} rows to {output_file}")
        print(f"  Memories per row: {k} (from ~{len(next(iter(persona_memories.values())))} total)")

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
