"""Partition the 200 sycophancy entries with Gemini 3.1 Pro Preview.

Reuses the upstream partition_memories.py logic via monkey-patch so we don't
duplicate 300 lines of script. Reads sycophancy.jsonl, writes partitioned
output to gemini_sycophancy/data/partitioned_sycophancy_only/.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

os.environ.setdefault(
    "VERTEXAI_SERVICE_ACCOUNT_PATH",
    str(REPO_ROOT / "service_account.json"),
)

sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmark.memory_normalization.persistbench import partition_memories as pm

pm.MODELS = [("google/gemini-3.1-pro-preview", "global")]
pm.INPUT_FILE = REPO_ROOT / "benchmark_samples/persistbench/baseline/sycophancy.jsonl"
pm.OUTPUT_DIR = REPO_ROOT / "gemini_sycophancy/data/partitioned_sycophancy_only"
pm.CONCURRENCY = 7

if __name__ == "__main__":
    asyncio.run(pm.main())
