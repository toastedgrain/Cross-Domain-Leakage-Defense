#!/usr/bin/env python3
"""Build samples where custom partitioned memories beat the baseline."""

from __future__ import annotations

import importlib.util
import json
import pathlib
from typing import Any


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
OUTPUTS = REPO_ROOT / "outputs" / "persistbench" / "all_configs"
SAMPLES = REPO_ROOT / "benchmark_samples" / "persistbench"
SPLIT_DIR = SCRIPT_DIR / "diff_samples_split"
TEX_OUT = SCRIPT_DIR / "custom_partitioned_better_than_baseline.tex"
TXT_OUT = SCRIPT_DIR / "custom_partitioned_better_than_baseline.txt"

MODELS = [
    {
        "model_id": "zai-org/glm-4.7-maas",
        "label": "GLM-4.7",
        "sample_dir": "zai-org_glm-4.7-maas",
        "slug": "glm_4_7",
    },
    {
        "model_id": "DeepSeek-V3.2",
        "label": "DeepSeek V3.2",
        "sample_dir": "DeepSeek-V3.2",
        "slug": "deepseek_v3_2",
    },
    {
        "model_id": "gpt-oss-120b",
        "label": "GPT-OSS-120B",
        "sample_dir": "gpt-oss-120b",
        "slug": "gpt_oss_120b",
    },
]


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
        compacted: dict[str, Any] = {}
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
    }
    jt = judge_text(best["judge"])
    if jt:
        payload["judge_explanation"] = jt
    elif best["judge"]:
        payload["judge"] = best["judge"]
    return payload


def build_deepseek_json() -> pathlib.Path:
    baseline = load_json(OUTPUTS / "baseline" / "output_all_models_baseline.json")
    custom = load_json(OUTPUTS / "partitioned_custom" / "output_all_models_partitioned_custom.json")
    model = next(item for item in MODELS if item["model_id"] == "DeepSeek-V3.2")
    custom_samples_path = (
        SAMPLES
        / "partitioned_custom_categories"
        / model["sample_dir"]
        / "full_benchmark.jsonl"
    )
    custom_samples = load_jsonl_by_hash(custom_samples_path)

    candidates: list[tuple[int, int, int, str]] = []
    shared_ids = set(baseline.get("entries", {})) & set(custom.get("entries", {}))
    for hash_id in shared_ids:
        baseline_entry = baseline["entries"][hash_id]
        custom_entry = custom["entries"][hash_id]
        if normalize_ft(baseline_entry) != "cross_domain":
            continue
        if normalize_ft(custom_entry) != "cross_domain":
            continue

        baseline_best = best_generation(baseline_entry, model["model_id"])
        custom_best = best_generation(custom_entry, model["model_id"])
        if baseline_best["score"] is None or custom_best["score"] is None:
            continue
        delta = baseline_best["score"] - custom_best["score"]
        if delta > 0:
            candidates.append((delta, baseline_best["score"], custom_best["score"], hash_id))

    candidates.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
    samples: list[dict[str, Any]] = []
    for rank, (delta, baseline_score, custom_score, hash_id) in enumerate(candidates[:5], start=1):
        baseline_entry = baseline["entries"][hash_id]
        custom_entry = custom["entries"][hash_id]
        custom_sample = custom_samples.get(hash_id, custom_entry)
        samples.append(
            {
                "rank": rank,
                "hash_id": hash_id,
                "score_change": {
                    "from_method": "Baseline",
                    "to_method": "Partitioned_Custom",
                    "from_score": baseline_score,
                    "to_score": custom_score,
                    "improvement": delta,
                    "note": "Higher scores are worse; improvement means custom partitioned has a lower worst-case score than baseline.",
                },
                "query": baseline_entry.get("query") or custom_entry.get("query"),
                "memory_domain": baseline_entry.get("memory_domain") or custom_entry.get("memory_domain"),
                "query_domain": baseline_entry.get("query_domain") or custom_entry.get("query_domain"),
                "methods": {
                    "Baseline": method_payload(
                        method="Baseline",
                        entry=baseline_entry,
                        model_id=model["model_id"],
                        memories=baseline_entry.get("memories"),
                        memory_format="flat_list",
                    ),
                    "Partitioned_Custom": method_payload(
                        method="Partitioned_Custom",
                        entry=custom_entry,
                        model_id=model["model_id"],
                        memories=custom_sample.get("memories", custom_entry.get("memories")),
                        memory_format="partitioned_custom",
                    ),
                },
            }
        )

    out_dir = SPLIT_DIR / model["slug"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "baseline_to_partitioned_custom.json"
    payload = {
        "model": {
            "label": model["label"],
            "model_id": model["model_id"],
        },
        "comparison": {
            "name": "Baseline -> Partitioned_Custom",
            "first_method": "Baseline",
            "second_method": "Partitioned_Custom",
        },
        "score_rule": "Each method score is max(valid judge scores across 3 runs). Scores are 1-5 and higher is worse.",
        "samples": samples,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def build_tex() -> None:
    helpers = _load_tex_helpers()
    description = (
        "Scores are the highest valid judge score across three runs. Scores range "
        "from 1 to 5, and higher is worse. Each sample below is a top cross-domain "
        "case where custom partitioned memories improve over the baseline."
    )
    lines = helpers.tex_header("Custom Partitioned Better Than Baseline", description)
    txt_lines = [
        "Custom partitioned better than baseline\n",
        "Score rule: max(valid judge scores across 3 runs); higher is worse.\n\n",
    ]

    for model in MODELS:
        path = SPLIT_DIR / model["slug"] / "baseline_to_partitioned_custom.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        lines.append(f"\\section*{{{helpers.tex_escape(data['model']['label'])}}}\n")
        txt_lines.append(f"{data['model']['label']}\n")
        txt_lines.append("-" * len(data["model"]["label"]) + "\n")
        for sample in data["samples"]:
            score = sample["score_change"]
            txt_lines.append(
                f"  {sample['rank']}. {sample['hash_id']}  "
                f"{score['from_score']}->{score['to_score']}  +{score['improvement']}\n"
            )
            lines.append(
                helpers.sample_box(
                    data["model"]["label"],
                    data["comparison"]["name"],
                    sample,
                )
            )
        txt_lines.append("\n")

    TEX_OUT.write_text("".join(lines), encoding="utf-8")
    TXT_OUT.write_text("".join(txt_lines), encoding="utf-8")


def main() -> None:
    deepseek_json = build_deepseek_json()
    build_tex()
    print(deepseek_json.relative_to(REPO_ROOT))
    print(TEX_OUT.relative_to(REPO_ROOT))
    print(TXT_OUT.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
