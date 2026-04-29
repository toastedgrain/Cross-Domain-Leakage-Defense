@echo off
REM Runs all 10 Gemini sycophancy add-on benchmarks from cmd.exe.
REM Stops on first failure (errorlevel != 0).

cd /d "%~dp0\.."

call uv run benchmark run gemini_sycophancy/configs/defence/permissive.json --ignore-config-mismatch || exit /b 1
call uv run benchmark run gemini_sycophancy/configs/defence/restrictive.json --ignore-config-mismatch || exit /b 1
call uv run benchmark run gemini_sycophancy/configs/defence/rubric_informed.json --ignore-config-mismatch || exit /b 1
call uv run benchmark run gemini_sycophancy/configs/defence/gepa_optimized.json --ignore-config-mismatch || exit /b 1

call uv run benchmark run gemini_sycophancy/configs/partitioned/partitioned.json --ignore-config-mismatch || exit /b 1
call uv run benchmark run gemini_sycophancy/configs/partitioned_defence/permissive.json --ignore-config-mismatch || exit /b 1
call uv run benchmark run gemini_sycophancy/configs/partitioned_defence/restrictive.json --ignore-config-mismatch || exit /b 1
call uv run benchmark run gemini_sycophancy/configs/partitioned_defence/rubric_informed.json --ignore-config-mismatch || exit /b 1
call uv run benchmark run gemini_sycophancy/configs/partitioned_defence/gepa_optimized.json --ignore-config-mismatch || exit /b 1

call uv run benchmark run gemini_sycophancy/configs/tree/informed_tree.json --ignore-config-mismatch || exit /b 1

echo All 10 runs completed.
