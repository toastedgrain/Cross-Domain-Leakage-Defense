#!/usr/bin/env python3
"""Build RAG-filtered PersistBench memory files — keep memories above a similarity threshold.

Reads the baseline PersistBench JSONL (flat memory list per row), embeds each
row's memories and its query, then keeps only the memories whose cosine
similarity to the query is >= threshold.

The output JSONL keeps the same schema as full_benchmark.jsonl (memories is a
flat list[str]) so the benchmark runner treats it as a baseline run — no
``method`` flag needed in the config.

Supports two embedding providers:
  - vertexai  : Vertex AI text-embedding-005 (default, uses service account)
  - openai    : OpenAI text-embedding-3-small (uses OPENAI_API_KEY)

Usage:
  # Default: threshold=0.5, Vertex AI embeddings
  uv run python rag_persistbench_memories.py

  # Multiple thresholds at once (one output file per threshold)
  uv run python rag_persistbench_memories.py --threshold 0.25 0.5 0.75

  # OpenAI embeddings
  uv run python rag_persistbench_memories.py --threshold 0.5 --provider openai

  # Filter to specific failure types
  uv run python rag_persistbench_memories.py --failure-types cross_domain sycophancy

─── HOW TO EDIT ───────────────────────────────────────────────────────────────
  * Change embedding model             ->  EMBEDDING CONFIG below
  * Change input/output paths          ->  RUN CONFIG below
  * Change concurrency                 ->  RUN CONFIG below
───────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import asyncio
import json
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
DEFAULT_THRESHOLD = 0.5
EMBED_BATCH_SIZE = 100  # texts per embedding API call
CONCURRENCY = 5
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
INPUT_FILE = (
    _PROJECT_ROOT
    / "benchmark_samples"
    / "persistbench"
    / "baseline"
    / "full_benchmark.jsonl"
)
OUTPUT_DIR = _PROJECT_ROOT / "benchmark_samples" / "persistbench" / "rag"
# ──────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from benchmark.utils import get_vertex_credentials, get_vertex_project_id  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build RAG-filtered PersistBench memory files (cosine similarity threshold).",
    )
    parser.add_argument(
        "--threshold",
        nargs="+",
        type=float,
        default=[DEFAULT_THRESHOLD],
        help=f"Cosine similarity threshold(s) in [0, 1] (default: {DEFAULT_THRESHOLD}). "
             "Memories with similarity >= threshold are kept. "
             "Pass multiple values to generate multiple output files.",
    )
    parser.add_argument(
        "--provider",
        choices=["vertexai", "openai"],
        default="vertexai",
        help="Embedding provider (default: vertexai).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_FILE,
        help=f"Input JSONL (default: {INPUT_FILE}).",
    )
    parser.add_argument(
        "--failure-types",
        nargs="+",
        default=None,
        help="Filter to specific failure_type values (e.g. cross_domain beneficial_memory_usage).",
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


async def main() -> None:
    args = _parse_args()

    # ── Load rows ────────────────────────────────────────────────────────────
    print(f"Loading PersistBench dataset from {args.input} ...")
    rows: list[dict] = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"Loaded {len(rows)} rows")

    if args.failure_types:
        wanted = set(args.failure_types)
        rows = [r for r in rows if r.get("failure_type") in wanted]
        print(f"Filtered to {len(rows)} rows for failure types: {args.failure_types}")

    if not rows:
        print("No rows to process.")
        return

    # ── Flatten memories with per-row offsets ────────────────────────────────
    # Each PersistBench row carries its own memory pool, so we flatten every
    # row's memories into one list for batched embedding, then slice the
    # results back out per row.
    all_memories: list[str] = []
    offsets: list[tuple[int, int]] = []  # (start, end) into all_memories per row
    for row in rows:
        start = len(all_memories)
        all_memories.extend(row["memories"])
        offsets.append((start, len(all_memories)))

    queries: list[str] = [row["query"] for row in rows]

    client = _EmbeddingClient(args.provider)
    semaphore = asyncio.Semaphore(CONCURRENCY)

    # ── Phase 1: Embed all memories ──────────────────────────────────────────
    print(
        f"\nPhase 1: Embedding {len(all_memories)} memories across {len(rows)} rows ..."
    )
    memory_embeddings = await _embed_all(
        client, all_memories, semaphore, desc="memories"
    )

    # ── Phase 2: Embed all queries ───────────────────────────────────────────
    print(f"\nPhase 2: Embedding {len(queries)} queries ...")
    query_embeddings = await _embed_all(
        client, queries, semaphore, desc="queries"
    )

    # ── Phase 3: For each threshold, filter by similarity and write output ───
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for threshold in args.threshold:
        output_file = OUTPUT_DIR / f"persistbench_rag_tau{threshold}.jsonl"
        print(f"\nPhase 3: Keeping memories with similarity >= {threshold} and writing {output_file} ...")
        counter = 0

        with open(output_file, "w", encoding="utf-8") as out:
            for row, (start, end), query_emb in zip(rows, offsets, query_embeddings):
                memories = all_memories[start:end]
                mem_embs = memory_embeddings[start:end]

                similarities = _cosine_similarity(query_emb, mem_embs)
                kept_indices = np.argsort(similarities)[::-1]
                kept_indices = [i for i in kept_indices if similarities[i] >= threshold]
                filtered = [memories[idx] for idx in kept_indices]

                # Generator sees the filtered subset via "memories"; judge later
                # uses "full_memories" so it can score leakage/beneficial-usage
                # against the complete pool (judgment.py prefers full_memories
                # when present).
                out_row = {**row, "memories": filtered, "full_memories": memories}
                out.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                counter += 1

        print(f"  Wrote {counter} rows to {output_file}")

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
