#!/usr/bin/env bash
# Runs the 5 Gemini sycophancy benchmarks that haven't yet processed sycophancy:
#   - 4 partitioned_defence configs (never executed in the recent batch)
#   - 1 tree-informed config (was previously run with the wrong 300-entry input)
#
# Each command resumes the existing checkpoint and only generates+judges the
# 200 sycophancy entries that are still missing.
set -e

cd "$(dirname "$0")/.."

uv run benchmark run gemini_sycophancy/configs/partitioned_defence/permissive.json --ignore-config-mismatch
uv run benchmark run gemini_sycophancy/configs/partitioned_defence/restrictive.json --ignore-config-mismatch
uv run benchmark run gemini_sycophancy/configs/partitioned_defence/rubric_informed.json --ignore-config-mismatch
uv run benchmark run gemini_sycophancy/configs/partitioned_defence/gepa_optimized.json --ignore-config-mismatch

uv run benchmark run gemini_sycophancy/configs/tree/informed_tree.json --ignore-config-mismatch
