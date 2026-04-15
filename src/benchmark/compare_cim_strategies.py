#!/usr/bin/env python3
"""Compare CIM benchmark results across strategies and judges.

Loads checkpoint files for each (strategy, judge) pair, computes the official
CIMemories paper metrics, and prints side-by-side comparison tables.

Usage:
    uv run python src/benchmark/compare_cim_strategies.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from benchmark.metrics_cim import compute_cim_metrics  # noqa: E402

# ── CONFIGURATION ────────────────────────────────────────────────────────────
# Each strategy maps to a dict of judge_name → checkpoint path (relative to repo root).
STRATEGIES: dict[str, dict[str, str]] = {
    "Baseline": {
        "Gemini 2.5 Flash": "outputs/CIM/llama3p3_labeled/baseline/cim_paper_replication_gemini.json",
        "Kimi K2":          "outputs/CIM/llama3p3_labeled/baseline/cim_paper_replication_kimi.json",
    },
    "Defense Medium": {
        "Gemini 2.5 Flash": "outputs/CIM/llama3p3_labeled/defense/cim_defense_medium_gemini.json",
        "Kimi K2":          "outputs/CIM/llama3p3_labeled/defense/cim_defense_medium_kimi.json",
    },
    "Defense High": {
        "Gemini 2.5 Flash": "outputs/CIM/llama3p3_labeled/defense/cim_defense_high_gemini.json",
        "Kimi K2":          "outputs/CIM/llama3p3_labeled/defense/cim_defense_high_kimi.json",
    },
    "Partitioned": {
        "Gemini 2.5 Flash": "outputs/CIM/llama3p3_labeled/partitioned/cim_partitioned_gemini.json",
        "Kimi K2":          "outputs/CIM/llama3p3_labeled/partitioned/cim_partitioned_kimi.json",
    },
}

JUDGE_ORDER = ["Gemini 2.5 Flash", "Kimi K2"]
# ─────────────────────────────────────────────────────────────────────────────


def _load(path: Path) -> dict | None:
    if not path.exists():
        print(f"  [SKIP] File not found: {path}")
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _fmt(val: float, std: float) -> str:
    return f"{val:5.2f} +/- {std:4.2f}"


_STRATEGY_SUBDIRS = ("baseline", "defense", "partitioned")


def _derive_strategy_name(subdir: str, file_stem: str) -> str:
    """Turn (subdir, filename) into a display label, e.g. ('defense', 'output_cim_high') -> 'Defense High'."""
    tail = file_stem.replace("output_cim_", "").replace("cim_", "")
    # Strip generator/judge suffixes like _llama3p3 for cleaner labels.
    for noise in ("_llama3p3", "_gemini", "_kimi", "_paper_replication"):
        tail = tail.replace(noise, "")
    tail = tail.strip("_") or subdir
    if subdir == "defense":
        return f"Defense {tail.title()}"
    return tail.replace("_", " ").title() or subdir.title()


def _discover_strategies(labels_dir: Path) -> dict[str, Path]:
    """Walk labels_dir and return {strategy_name: checkpoint_path}.

    Accepts both flat (labels_dir/baseline/...) and nested
    (labels_dir/<generator>/baseline/...) layouts.
    """
    candidates: list[Path] = []
    for sub in _STRATEGY_SUBDIRS:
        direct = labels_dir / sub
        if direct.is_dir():
            candidates.append(direct)
        # One-level nesting (e.g. llama3p3/baseline)
        for child in labels_dir.iterdir() if labels_dir.is_dir() else []:
            nested = child / sub
            if nested.is_dir():
                candidates.append(nested)

    strategies: dict[str, Path] = {}
    for strat_dir in candidates:
        subdir = strat_dir.name
        for jf in sorted(strat_dir.glob("*.json")):
            name = _derive_strategy_name(subdir, jf.stem)
            # Avoid collisions — append path hint if needed.
            if name in strategies:
                name = f"{name} ({strat_dir.parent.name})"
            strategies[name] = jf
    return strategies


def run_compare_cli(labels_dir: str | Path, per_persona: bool = False) -> int:
    """Auto-discover strategy JSONs under labels_dir and print a comparison table.

    Layout expected: labels_dir/[<generator>/]{baseline,defense,partitioned}/*.json
    Each JSON becomes one row. Judge column is omitted (assumes one judge per tree).
    """
    root = Path(labels_dir).resolve()
    if not root.is_dir():
        print(f"[ERROR] Not a directory: {root}")
        return 1

    discovered = _discover_strategies(root)
    if not discovered:
        print(f"[ERROR] No strategy checkpoints found under {root}")
        print(f"        Expected subdirs: {_STRATEGY_SUBDIRS}")
        return 1

    rows: list[tuple[str, dict | None]] = []
    skipped: list[tuple[str, Path, str]] = []
    for name, path in discovered.items():
        data = _load(path)
        if data is None:
            skipped.append((name, path, "file not found"))
            continue
        m = compute_cim_metrics(data)
        if m["n_entries"] == 0:
            skipped.append((name, path, "0 entries (unparseable or wrong schema)"))
            continue
        rows.append((name, m))

    for name, path, reason in skipped:
        print(f"[SKIP] {name} -> {path.name}: {reason}")

    width = 96
    print()
    print("=" * width)
    print(f"{'CIM Benchmark: Strategy Comparison':^{width}}")
    print(f"{f'Root: {root}':^{width}}")
    print("=" * width)
    print(f"{'Strategy':<30} | {'Violation %':>16} | {'Coverage %':>16} | {'Entries':>8}")
    print("-" * width)
    for name, m in rows:
        if m is None:
            print(f"{name:<30} | {'--':>16} | {'--':>16} | {'--':>8}")
            continue
        viol = _fmt(m["violation_mean"], m["violation_std"])
        cov = _fmt(m["coverage_mean"], m["coverage_std"])
        print(f"{name:<30} | {viol:>16} | {cov:>16} | {m['n_entries']:>8}")

    # Defense-vs-baseline reduction block
    baseline_row = next(
        (m for n, m in rows if n.lower() == "baseline" and m is not None), None
    )
    if baseline_row and baseline_row["violation_mean"] > 0:
        print()
        print(f"{'Change vs Baseline (- is better for violation, + for coverage)':^{width}}")
        print("-" * width)
        for name, m in rows:
            if m is None or name.lower() == "baseline":
                continue
            # Relative change in violation: negative = defense worked.
            vd = (m["violation_mean"] / baseline_row["violation_mean"] - 1) * 100
            cd = m["coverage_mean"] - baseline_row["coverage_mean"]
            print(f"{name:<30} | violation {vd:+6.1f}%   | coverage {cd:+6.2f}pp")

    if per_persona:
        _print_per_persona(rows, metric_key="per_user_violation", label="Violation %", width=width)
        _print_per_persona(rows, metric_key="per_user_coverage", label="Coverage %", width=width)

    print()
    print("Violation = % private attrs leaked  (lower is better)")
    print("Coverage  = % required attrs included (higher is better)")
    print()
    return 0


def _print_per_persona(
    rows: list[tuple[str, dict | None]],
    metric_key: str,
    label: str,
    width: int,
) -> None:
    """Print a persona x strategy matrix for the given metric."""
    strat_names = [n for n, m in rows if m is not None]
    personas: set[str] = set()
    for _, m in rows:
        if m is not None:
            personas.update(m.get(metric_key, {}).keys())
    if not personas:
        return

    print()
    print("=" * width)
    print(f"{f'Per-Persona {label} by Strategy':^{width}}")
    print("=" * width)
    persona_w = max(len("Persona"), max(len(p) for p in personas))
    col_w = 14
    header = f"{'Persona':<{persona_w}}" + "".join(
        f" | {s[:col_w]:>{col_w}}" for s in strat_names
    )
    print(header)
    print("-" * len(header))
    for persona in sorted(personas):
        row = f"{persona:<{persona_w}}"
        for name, m in rows:
            if m is None:
                continue
            val = m.get(metric_key, {}).get(persona)
            row += f" | {val * 100:>{col_w - 1}.2f}%" if val is not None else f" | {'--':>{col_w}}"
        print(row)


def main() -> None:
    # Collect metrics: metrics[strategy][judge] = dict
    metrics: dict[str, dict[str, dict | None]] = {}

    for strategy, judges in STRATEGIES.items():
        metrics[strategy] = {}
        for judge, rel_path in judges.items():
            path = _PROJECT_ROOT / rel_path
            data = _load(path)
            if data is None:
                metrics[strategy][judge] = None
            else:
                metrics[strategy][judge] = compute_cim_metrics(data)

    judges = JUDGE_ORDER

    # ── Main Comparison Table ────────────────────────────────────────────
    print()
    print("=" * 110)
    print(f"{'CIM Benchmark: Strategy x Judge Comparison':^110}")
    print(f"{'Generator: Llama 3.3 70B | Metrics: Official CIMemories Paper Aggregation':^110}")
    print("=" * 110)
    print()

    # Header
    print(f"{'Strategy':<18} | {'Judge':<18} | {'Violation %':>14} | {'Coverage %':>14} | {'Entries':>8}")
    print("-" * 110)

    for strategy in STRATEGIES:
        first = True
        for judge in judges:
            m = metrics[strategy].get(judge)
            strat_col = strategy if first else ""
            first = False

            if m is None:
                print(f"{strat_col:<18} | {judge:<18} | {'--':>14} | {'--':>14} | {'--':>8}")
                continue

            viol = _fmt(m["violation_mean"], m["violation_std"])
            cov = _fmt(m["coverage_mean"], m["coverage_std"])
            print(f"{strat_col:<18} | {judge:<18} | {viol:>14} | {cov:>14} | {m['n_entries']:>8}")

        # Delta row if both judges present
        g = metrics[strategy].get(judges[0])
        k = metrics[strategy].get(judges[1])
        if g and k:
            dv = g["violation_mean"] - k["violation_mean"]
            dc = g["coverage_mean"] - k["coverage_mean"]
            print(f"{'':<18} | {'  delta (G - K)':<18} | {dv:>+10.2f}      | {dc:>+10.2f}      | {'':>8}")
        print("-" * 110)

    # ── Defense Effectiveness (per judge) ────────────────────────────────
    print()
    print("=" * 110)
    print(f"{'Defense Effectiveness vs Baseline':^110}")
    print("=" * 110)

    for judge in judges:
        print(f"\n  Judge: {judge}")
        print(f"  {'Strategy':<18} | {'Violation %':>14} | {'Coverage %':>14} | {'VR Reduction':>14} | {'Cov Change':>12}")
        print(f"  {'-' * 88}")

        baseline = metrics["Baseline"].get(judge)
        if baseline is None:
            print(f"  {'(baseline missing)':>60}")
            continue

        for strategy in STRATEGIES:
            m = metrics[strategy].get(judge)
            if m is None:
                print(f"  {strategy:<18} | {'--':>14} | {'--':>14} | {'--':>14} | {'--':>12}")
                continue

            viol = _fmt(m["violation_mean"], m["violation_std"])
            cov = _fmt(m["coverage_mean"], m["coverage_std"])

            if strategy == "Baseline":
                reduction = "--"
                cov_delta = "--"
            else:
                if baseline["violation_mean"] > 0:
                    r = (1 - m["violation_mean"] / baseline["violation_mean"]) * 100
                    reduction = f"{r:+.0f}%"
                else:
                    reduction = "N/A"
                cd = m["coverage_mean"] - baseline["coverage_mean"]
                cov_delta = f"{cd:+.1f}pp"

            print(f"  {strategy:<18} | {viol:>14} | {cov:>14} | {reduction:>14} | {cov_delta:>12}")

    # ── Per-User Violation Breakdown ─────────────────────────────────────
    print()
    print("=" * 110)
    print(f"{'Per-User Violation Rate (%) by Strategy':^110}")
    print("=" * 110)

    # Collect all user names
    all_users: set[str] = set()
    for strategy in STRATEGIES:
        for judge in judges:
            m = metrics[strategy].get(judge)
            if m and m.get("per_user_violation"):
                all_users.update(m["per_user_violation"].keys())
    users_sorted = sorted(all_users)

    for judge in judges:
        print(f"\n  Judge: {judge}")
        strat_names = list(STRATEGIES.keys())
        header = f"  {'User':<20}" + "".join(f" | {s:>16}" for s in strat_names)
        print(header)
        print(f"  {'-' * (22 + 19 * len(strat_names))}")

        for user in users_sorted:
            row = f"  {user:<20}"
            for strategy in strat_names:
                m = metrics[strategy].get(judge)
                if m and m.get("per_user_violation") and user in m["per_user_violation"]:
                    val = m["per_user_violation"][user] * 100
                    row += f" | {val:>15.2f}%"
                else:
                    row += f" | {'--':>16}"
            print(row)

    print()
    print("Violation = % of private attributes leaked  (lower is better)")
    print("Coverage  = % of required attributes included (higher is better)")
    print("Metrics use official CIMemories paper aggregation (worst-case violation, average-case coverage)")
    print()


if __name__ == "__main__":
    main()
