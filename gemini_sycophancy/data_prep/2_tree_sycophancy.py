"""Build memory trees for the 200 sycophancy entries with Gemini 3.1 Pro Preview.

Reuses the upstream tree_persistbench_memories.py logic via monkey-patch.
Output goes to gemini_sycophancy/data/tree_sycophancy_only/.
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

from benchmark.memory_normalization.persistbench import tree_persistbench_memories as tm

tm.MODELS = [("google/gemini-3.1-pro-preview", "global")]
tm.INPUT_FILE = REPO_ROOT / "benchmark_samples/persistbench/baseline/sycophancy.jsonl"
tm.OUTPUT_DIR = REPO_ROOT / "gemini_sycophancy/data/tree_sycophancy_only"
tm.CONCURRENCY = 7

# Patch the args parser so the script doesn't try to parse our argv when run directly.
import argparse
def _no_args() -> argparse.Namespace:
    return argparse.Namespace(input=tm.INPUT_FILE)
tm._parse_args = _no_args

if __name__ == "__main__":
    asyncio.run(tm.main())
