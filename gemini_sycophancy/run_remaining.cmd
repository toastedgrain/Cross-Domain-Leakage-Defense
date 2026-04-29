@echo off
REM Runs the 5 Gemini sycophancy benchmarks that haven't yet processed sycophancy:
REM   - 4 partitioned_defence configs (never executed in the recent batch)
REM   - 1 tree-informed config (was previously run with the wrong 300-entry input)
REM
REM Stops on first failure (exit /b 1) so you can re-run cleanly.

cd /d "%~dp0\.."

call uv run benchmark run gemini_sycophancy/configs/partitioned_defence/permissive.json --ignore-config-mismatch || exit /b 1
call uv run benchmark run gemini_sycophancy/configs/partitioned_defence/restrictive.json --ignore-config-mismatch || exit /b 1
call uv run benchmark run gemini_sycophancy/configs/partitioned_defence/rubric_informed.json --ignore-config-mismatch || exit /b 1
call uv run benchmark run gemini_sycophancy/configs/partitioned_defence/gepa_optimized.json --ignore-config-mismatch || exit /b 1

call uv run benchmark run gemini_sycophancy/configs/tree/informed_tree.json --ignore-config-mismatch || exit /b 1

echo All 5 remaining runs completed.
