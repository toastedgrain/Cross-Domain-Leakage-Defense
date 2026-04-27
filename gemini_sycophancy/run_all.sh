#!/usr/bin/env bash
# Runs all 10 Gemini sycophancy add-on benchmarks.
# Each command resumes the existing 300-entry checkpoint at outputs/persistbench/...
# and only generates+judges the 200 new sycophancy entries.
#
# Prereq: phase 1 (data prep) must have been completed -- see README.md
set -e

cd "$(dirname "$0")/.."

# Defensive (4) -- no data prep required
uv run benchmark run gemini_sycophancy/configs/defence/permissive.json --ignore-config-mismatch
uv run benchmark run gemini_sycophancy/configs/defence/restrictive.json --ignore-config-mismatch
uv run benchmark run gemini_sycophancy/configs/defence/rubric_informed.json --ignore-config-mismatch
uv run benchmark run gemini_sycophancy/configs/defence/gepa_optimized.json --ignore-config-mismatch

# Partitioned + 4 partition_defence (5) -- requires phase-1 partitioned data
uv run benchmark run gemini_sycophancy/configs/partitioned/partitioned.json --ignore-config-mismatch
uv run benchmark run gemini_sycophancy/configs/partitioned_defence/permissive.json --ignore-config-mismatch
uv run benchmark run gemini_sycophancy/configs/partitioned_defence/restrictive.json --ignore-config-mismatch
uv run benchmark run gemini_sycophancy/configs/partitioned_defence/rubric_informed.json --ignore-config-mismatch
uv run benchmark run gemini_sycophancy/configs/partitioned_defence/gepa_optimized.json --ignore-config-mismatch

# Tree (1) -- requires phase-1 tree data
uv run benchmark run gemini_sycophancy/configs/tree/informed_tree.json --ignore-config-mismatch
