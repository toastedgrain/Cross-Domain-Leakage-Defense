@echo off
REM Run the 5 partitioned-custom-categories Gemini benchmarks that need
REM sycophancy entries added. See run_all.sh for the full explanation.
REM Stops on first failure (exit /b 1).

cd /d "%~dp0\.."

set "VERTEXAI_SERVICE_ACCOUNT_PATH=%CD%\service_account.json"

if not exist "%VERTEXAI_SERVICE_ACCOUNT_PATH%" (
  echo ERROR: service account not found at %VERTEXAI_SERVICE_ACCOUNT_PATH% 1>&2
  exit /b 1
)

echo Using service account: %VERTEXAI_SERVICE_ACCOUNT_PATH%
echo.

echo === [1/5] partitioned_custom_categories ===
call uv run benchmark run configs/persistbench/partitioned_custom_categories/gemini3_pro_partitioned_custom_categories.json --ignore-config-mismatch --no-auto-rerun

echo === [2/5] + permissive ===
call uv run benchmark run configs/persistbench/partitioned_custom_categories_defence/gemini3_pro_partitioned_custom_permissive.json --ignore-config-mismatch --no-auto-rerun

echo === [3/5] + restrictive ===
call uv run benchmark run configs/persistbench/partitioned_custom_categories_defence/gemini3_pro_partitioned_custom_restrictive.json --ignore-config-mismatch --no-auto-rerun

echo === [4/5] + rubric_informed ===
call uv run benchmark run configs/persistbench/partitioned_custom_categories_defence/gemini3_pro_partitioned_custom_rubric_informed.json --ignore-config-mismatch --no-auto-rerun

echo === [5/5] + gepa_optimized ===
call uv run benchmark run configs/persistbench/partitioned_custom_categories_defence/gemini3_pro_partitioned_custom_gepa_optimized.json --ignore-config-mismatch --no-auto-rerun

echo.
echo All 5 runs completed. Rendering the failure-rate table:
echo.
call python analysis/persistbench/failure_rates.py
