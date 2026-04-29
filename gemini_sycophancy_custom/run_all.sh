#!/usr/bin/env bash
# Run the 5 partitioned-custom-categories Gemini benchmarks that need
# sycophancy entries added.
#
# Why this exists: the original 5 Gemini runs in this method group used a
# 300-entry input snapshot (CD + Bnf only). Sycophancy was appended to the
# input file later, so the output checkpoints are missing 200 entries each.
# The runner's hash-keyed resume logic adds the missing 200 to each existing
# checkpoint and skips the 300 already-completed entries.
#
# Auth note: the runner reads the Vertex AI service account from
# VERTEXAI_SERVICE_ACCOUNT_PATH, defaulting to ~/Downloads/VERTEXAI_SERVICE_ACCOUNT.json
# if unset. We override here to use the SA at the repo root.

# Note: intentionally NOT using `set -e`. The benchmark runner exits non-zero
# when individual generations fail (e.g., Gemini safety-filter blocks on a few
# specific entries) even with --no-auto-rerun. Those persistent failures are
# expected, not catastrophic, and we want the script to continue to the next
# config. The pre-flight checks below cover the genuinely fatal cases.

# Always run from the repo root, regardless of where this script is invoked from.
cd "$(dirname "$0")/.."

export VERTEXAI_SERVICE_ACCOUNT_PATH="$(pwd)/service_account.json"

if [ ! -f "$VERTEXAI_SERVICE_ACCOUNT_PATH" ]; then
  echo "ERROR: service account not found at $VERTEXAI_SERVICE_ACCOUNT_PATH" >&2
  exit 1
fi

echo "Using service account: $VERTEXAI_SERVICE_ACCOUNT_PATH"
echo

echo "=== [1/5] partitioned_custom_categories ==="
uv run benchmark run configs/persistbench/partitioned_custom_categories/gemini3_pro_partitioned_custom_categories.json --ignore-config-mismatch --no-auto-rerun || true

echo "=== [2/5] + permissive ==="
uv run benchmark run configs/persistbench/partitioned_custom_categories_defence/gemini3_pro_partitioned_custom_permissive.json --ignore-config-mismatch --no-auto-rerun || true

echo "=== [3/5] + restrictive ==="
uv run benchmark run configs/persistbench/partitioned_custom_categories_defence/gemini3_pro_partitioned_custom_restrictive.json --ignore-config-mismatch --no-auto-rerun || true

echo "=== [4/5] + rubric_informed ==="
uv run benchmark run configs/persistbench/partitioned_custom_categories_defence/gemini3_pro_partitioned_custom_rubric_informed.json --ignore-config-mismatch --no-auto-rerun || true

echo "=== [5/5] + gepa_optimized ==="
uv run benchmark run configs/persistbench/partitioned_custom_categories_defence/gemini3_pro_partitioned_custom_gepa_optimized.json --ignore-config-mismatch --no-auto-rerun || true

echo
echo "All 5 runs completed. Rendering the failure-rate table:"
echo
python analysis/persistbench/failure_rates.py
