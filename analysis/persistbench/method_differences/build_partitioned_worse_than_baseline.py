#!/usr/bin/env python3
"""Build a LaTeX fragment where fixed partitions underperform baseline."""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import pathlib
from typing import Any


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
OUTPUTS = REPO_ROOT / "outputs" / "persistbench" / "all_configs"
SAMPLES = REPO_ROOT / "benchmark_samples" / "persistbench"
SPLIT_DIR = SCRIPT_DIR / "diff_samples_split"
TEX_OUT = SCRIPT_DIR / "partitioned_worse_than_baseline.tex"
TXT_OUT = SCRIPT_DIR / "partitioned_worse_than_baseline.txt"

MODELS = {
    "zai-org/glm-4.7-maas": {
        "label": "GLM-4.7",
        "sample_dir": "zai-org_glm-4.7-maas",
        "slug": "glm_4_7",
    },
    "xai/grok-4.1-fast-non-reasoning": {
        "label": "Grok 4.1 Fast",
        "sample_dir": "xai_grok-4.1-fast-non-reasoning",
        "slug": "grok_4_1_fast",
    },
    "gpt-oss-120b": {
        "label": "GPT-OSS-120B",
        "sample_dir": "gpt-oss-120b",
        "slug": "gpt_oss_120b",
    },
}


def _load_tex_helpers():
    helper_path = SCRIPT_DIR / "build_partition_method_comparisons.py"
    spec = importlib.util.spec_from_file_location("partition_method_helpers", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_ft(entry: dict[str, Any]) -> str:
    ft = entry.get("failure_type") or entry.get("leakage_type") or "cross_domain"
    return "beneficial_memory_usage" if ft == "positive_memory_usage" else ft


def extract_score(generation: dict[str, Any]) -> int | None:
    judge = generation.get("judge")
    if generation.get("error") or not isinstance(judge, dict):
        return None
    score = judge.get("score")
    return score if isinstance(score, int) else None


def best_generation(entry: dict[str, Any], model_id: str) -> dict[str, Any]:
    model_data = entry.get("results", {}).get(model_id) or {}
    generations = model_data.get("generations", [])
    valid: list[tuple[int, int, dict[str, Any]]] = []
    all_scores: list[int | None] = []

    for pos, generation in enumerate(generations):
        score = extract_score(generation)
        all_scores.append(score)
        if score is not None:
            generation_index = generation.get("generation_index", pos)
            valid.append((score, generation_index, generation))

    if not valid:
        return {
            "score": None,
            "generation_index": None,
            "all_scores": all_scores,
            "output": "",
            "judge": None,
        }

    score, generation_index, generation = sorted(valid, key=lambda item: (-item[0], item[1]))[0]
    return {
        "score": score,
        "generation_index": generation_index,
        "all_scores": all_scores,
        "output": generation.get("memory_response") or "",
        "judge": generation.get("judge"),
    }


def load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl_by_hash(path: pathlib.Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            hash_id = entry.get("hash_id")
            if hash_id:
                rows[hash_id] = entry
    return rows


def compact_memories(value: Any) -> Any:
    if isinstance(value, list):
        return [item for item in value if item not in (None, "", [], {})]
    if isinstance(value, dict):
        compacted = {}
        for key, child in value.items():
            child_value = compact_memories(child)
            if child_value not in (None, "", [], {}):
                compacted[key] = child_value
        return compacted
    return value


def judge_text(judge: Any) -> str | None:
    if not isinstance(judge, dict):
        return None
    return (
        judge.get("explanation")
        or judge.get("reasoning")
        or judge.get("rationale")
        or judge.get("summary")
    )


def method_payload(
    *,
    method: str,
    entry: dict[str, Any],
    model_id: str,
    memories: Any,
    memory_format: str,
    sample_source: str,
) -> dict[str, Any]:
    best = best_generation(entry, model_id)
    payload = {
        "method": method,
        "score": best["score"],
        "generation_index": best["generation_index"],
        "all_scores": best["all_scores"],
        "memory_format": memory_format,
        "memories": compact_memories(memories),
        "highest_scored_output": best["output"],
        "sample_source": sample_source,
    }
    jt = judge_text(best["judge"])
    if jt:
        payload["judge_explanation"] = jt
    elif best["judge"]:
        payload["judge"] = best["judge"]
    return payload


def build_payloads() -> list[pathlib.Path]:
    baseline = load_json(OUTPUTS / "baseline" / "output_all_models_baseline.json")
    partitioned = load_json(OUTPUTS / "partitioned" / "output_all_models_partitioned.json")
    written: list[pathlib.Path] = []

    for model_id, model_meta in MODELS.items():
        partitioned_samples_path = (
            SAMPLES / "partitioned" / model_meta["sample_dir"] / "full_benchmark.jsonl"
        )
        partitioned_samples = load_jsonl_by_hash(partitioned_samples_path)
        model_dir = SPLIT_DIR / model_meta["slug"]
        model_dir.mkdir(parents=True, exist_ok=True)

        candidates: list[tuple[int, int, int, str]] = []
        shared_ids = set(baseline.get("entries", {})) & set(partitioned.get("entries", {}))
        for hash_id in shared_ids:
            baseline_entry = baseline["entries"][hash_id]
            partitioned_entry = partitioned["entries"][hash_id]
            if normalize_ft(baseline_entry) != "cross_domain":
                continue
            if normalize_ft(partitioned_entry) != "cross_domain":
                continue

            partitioned_best = best_generation(partitioned_entry, model_id)
            baseline_best = best_generation(baseline_entry, model_id)
            if partitioned_best["score"] is None or baseline_best["score"] is None:
                continue

            delta = partitioned_best["score"] - baseline_best["score"]
            if delta > 0:
                candidates.append(
                    (delta, partitioned_best["score"], baseline_best["score"], hash_id)
                )

        candidates.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
        samples: list[dict[str, Any]] = []
        for rank, (delta, partitioned_score, baseline_score, hash_id) in enumerate(candidates[:5], start=1):
            baseline_entry = baseline["entries"][hash_id]
            partitioned_entry = partitioned["entries"][hash_id]
            partitioned_sample = partitioned_samples.get(hash_id, partitioned_entry)
            sample = {
                "rank": rank,
                "hash_id": hash_id,
                "score_change": {
                    "from_method": "Partitioned",
                    "to_method": "Baseline",
                    "from_score": partitioned_score,
                    "to_score": baseline_score,
                    "improvement": delta,
                    "note": "Higher scores are worse; improvement means Baseline has a lower worst-case score than Partitioned.",
                },
                "query": baseline_entry.get("query") or partitioned_entry.get("query"),
                "memory_domain": baseline_entry.get("memory_domain") or partitioned_entry.get("memory_domain"),
                "query_domain": baseline_entry.get("query_domain") or partitioned_entry.get("query_domain"),
                "methods": {
                    "Partitioned": method_payload(
                        method="Partitioned",
                        entry=partitioned_entry,
                        model_id=model_id,
                        memories=partitioned_sample.get("memories", partitioned_entry.get("memories")),
                        memory_format="partitioned",
                        sample_source=str(partitioned_samples_path.relative_to(REPO_ROOT)),
                    ),
                    "Baseline": method_payload(
                        method="Baseline",
                        entry=baseline_entry,
                        model_id=model_id,
                        memories=baseline_entry.get("memories"),
                        memory_format="flat_list",
                        sample_source=str(
                            (OUTPUTS / "baseline" / "output_all_models_baseline.json").relative_to(REPO_ROOT)
                        ),
                    ),
                },
            }
            samples.append(sample)

        payload = {
            "metadata": {
                "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "failure_type": "cross_domain",
                "top_n": 5,
                "score_rule": "Each method score is max(valid judge scores across 3 runs). Scores are 1-5 and higher is worse.",
                "tie_break": "delta desc, partitioned score desc, baseline score asc, hash_id asc",
            },
            "model": {
                "label": model_meta["label"],
                "model_id": model_id,
            },
            "comparison": {
                "name": "Partitioned -> Baseline",
                "first_method": "Partitioned",
                "second_method": "Baseline",
            },
            "samples": samples,
        }
        out_path = model_dir / "partitioned_to_baseline.json"
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        written.append(out_path)

    return written


def build_tex() -> None:
    helpers = _load_tex_helpers()
    description = (
        "Scores are the highest valid judge score across three runs. Scores range "
        "from 1 to 5, and higher is worse. Each sample below is a top cross-domain "
        "case where fixed Partitioned memories have a higher worst-case score than "
        "the flat Baseline."
    )
    lines = helpers.tex_header("Fixed Partitions Worse Than Baseline", description)
    txt_lines = [
        "Fixed partitions worse than baseline\n",
        "Score rule: max(valid judge scores across 3 runs); higher is worse.\n\n",
    ]

    for model_id, model_meta in MODELS.items():
        lines.append(f"\\section*{{{helpers.tex_escape(model_meta['label'])}}}\n")
        txt_lines.append(f"{model_meta['label']}\n")
        txt_lines.append("-" * len(model_meta["label"]) + "\n")
        path = SPLIT_DIR / model_meta["slug"] / "partitioned_to_baseline.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        for sample in data["samples"]:
            score = sample["score_change"]
            txt_lines.append(
                f"  {sample['rank']}. {sample['hash_id']}  "
                f"{score['from_score']}->{score['to_score']}  +{score['improvement']}\n"
            )
            lines.append(helpers.sample_box(model_meta["label"], "Partitioned -> Baseline", sample))
        txt_lines.append("\n")

    TEX_OUT.write_text("".join(lines), encoding="utf-8")
    TXT_OUT.write_text("".join(txt_lines), encoding="utf-8")


def main() -> None:
    written = build_payloads()
    build_tex()
    for path in written:
        print(path.relative_to(REPO_ROOT))
    print(TEX_OUT.relative_to(REPO_ROOT))
    print(TXT_OUT.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
