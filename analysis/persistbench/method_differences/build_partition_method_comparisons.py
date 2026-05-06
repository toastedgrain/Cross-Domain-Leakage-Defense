#!/usr/bin/env python3
"""Build partition-method diff JSON files and a LaTeX comparison fragment."""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import re
from typing import Any, Callable


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
OUTPUTS = REPO_ROOT / "outputs" / "persistbench" / "all_configs"
SAMPLES = REPO_ROOT / "benchmark_samples" / "persistbench"
SPLIT_DIR = SCRIPT_DIR / "diff_samples_split"
TEX_OUT = SCRIPT_DIR / "partition_method_comparisons.tex"
PARTITIONED_WORSE_TEX_OUT = SCRIPT_DIR / "partitioned_memories_perform_worse.tex"
PARTITIONED_WORSE_TXT_OUT = SCRIPT_DIR / "partitioned_memories_perform_worse.txt"
FOCUSED_TEX_OUTPUTS = [
    (
        "Cos Similarity Partitioned -> Partitioned",
        SCRIPT_DIR / "cos_similarity_partitioned_better_than_partitioned.tex",
        "Cos Similarity Partitioned Better Than Partitioned",
    ),
    (
        "Partitioned -> Cos Similarity Partitioned",
        SCRIPT_DIR / "partitioned_better_than_cos_similarity_partitioned.tex",
        "Partitioned Better Than Cos Similarity Partitioned",
    ),
    (
        "Partitioned -> Dynamic Partitioned",
        SCRIPT_DIR / "dynamic_partitioned_better_than_partitioned.tex",
        "Dynamic Partitioned Better Than Partitioned",
    ),
]
PARTITIONED_WORSE_COMPARISONS = [
    "Partitioned -> Cos Similarity Partitioned",
    "Partitioned -> Dynamic Partitioned",
]

MODELS: dict[str, dict[str, str]] = {
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

METHODS: dict[str, dict[str, Any]] = {
    "Partitioned": {
        "checkpoint": OUTPUTS / "partitioned" / "output_all_models_partitioned.json",
        "sample_path": lambda sample_dir: SAMPLES / "partitioned" / sample_dir / "full_benchmark.jsonl",
        "memory_format": "partitioned",
    },
    "Cos Similarity Partitioned": {
        "checkpoint": OUTPUTS / "partitioned_cos" / "output_all_models_partitioned_cos.json",
        "sample_path": lambda sample_dir: SAMPLES / "partitioned" / "cos_similarity" / "full_benchmark.jsonl",
        "memory_format": "partitioned_cos",
    },
    "Dynamic Partitioned": {
        "checkpoint": OUTPUTS / "partitioned_custom" / "output_all_models_partitioned_custom.json",
        "sample_path": lambda sample_dir: SAMPLES / "partitioned_custom_categories" / sample_dir / "full_benchmark.jsonl",
        "memory_format": "dynamic_partitioned",
    },
}

COMPARISONS = [
    ("Cos Similarity Partitioned", "Partitioned"),
    ("Partitioned", "Cos Similarity Partitioned"),
    ("Partitioned", "Dynamic Partitioned"),
    ("Dynamic Partitioned", "Partitioned"),
]


def slug(value: str) -> str:
    value = value.lower().replace("->", "to")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


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


def load_checkpoint(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl_by_hash(path: pathlib.Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
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
    model_meta: dict[str, str],
    hash_id: str,
    checkpoint_entry: dict[str, Any],
    sample_cache: dict[tuple[str, str], tuple[pathlib.Path, dict[str, dict[str, Any]]]],
) -> dict[str, Any]:
    sample_path_fn: Callable[[str], pathlib.Path] = METHODS[method]["sample_path"]
    sample_dir = model_meta["sample_dir"]
    cache_key = (method, sample_dir)
    if cache_key not in sample_cache:
        sample_path = sample_path_fn(sample_dir)
        sample_cache[cache_key] = (sample_path, load_jsonl_by_hash(sample_path))

    sample_path, rows = sample_cache[cache_key]
    sample_entry = rows.get(hash_id, checkpoint_entry)
    best = best_generation(checkpoint_entry, model_meta["model_id"])
    payload = {
        "method": method,
        "score": best["score"],
        "generation_index": best["generation_index"],
        "all_scores": best["all_scores"],
        "memory_format": METHODS[method]["memory_format"],
        "memories": compact_memories(sample_entry.get("memories", checkpoint_entry.get("memories"))),
        "highest_scored_output": best["output"],
        "sample_source": str(sample_path.relative_to(REPO_ROOT)),
    }
    jt = judge_text(best["judge"])
    if jt:
        payload["judge_explanation"] = jt
    elif best["judge"]:
        payload["judge"] = best["judge"]
    return payload


def build_comparison_payloads() -> list[pathlib.Path]:
    checkpoints = {
        method: load_checkpoint(config["checkpoint"])
        for method, config in METHODS.items()
    }
    sample_cache: dict[tuple[str, str], tuple[pathlib.Path, dict[str, dict[str, Any]]]] = {}
    written: list[pathlib.Path] = []

    for model_id, raw_model_meta in MODELS.items():
        model_meta = {**raw_model_meta, "model_id": model_id}
        model_dir = SPLIT_DIR / model_meta["slug"]
        model_dir.mkdir(parents=True, exist_ok=True)

        for first, second in COMPARISONS:
            first_entries = checkpoints[first].get("entries", {})
            second_entries = checkpoints[second].get("entries", {})
            shared_ids = set(first_entries) & set(second_entries)
            candidates: list[tuple[int, int, int, str]] = []

            for hash_id in shared_ids:
                first_entry = first_entries[hash_id]
                second_entry = second_entries[hash_id]
                if normalize_ft(first_entry) != "cross_domain":
                    continue
                if normalize_ft(second_entry) != "cross_domain":
                    continue

                first_best = best_generation(first_entry, model_id)
                second_best = best_generation(second_entry, model_id)
                if first_best["score"] is None or second_best["score"] is None:
                    continue

                delta = first_best["score"] - second_best["score"]
                if delta > 0:
                    candidates.append((delta, first_best["score"], second_best["score"], hash_id))

            candidates.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
            samples: list[dict[str, Any]] = []
            for rank, (delta, first_score, second_score, hash_id) in enumerate(candidates[:5], start=1):
                first_entry = checkpoints[first]["entries"][hash_id]
                second_entry = checkpoints[second]["entries"][hash_id]
                query = first_entry.get("query") or second_entry.get("query")
                sample = {
                    "rank": rank,
                    "hash_id": hash_id,
                    "score_change": {
                        "from_method": first,
                        "to_method": second,
                        "from_score": first_score,
                        "to_score": second_score,
                        "improvement": delta,
                        "note": "Higher scores are worse; improvement means the second method has a lower worst-case score.",
                    },
                    "query": query,
                    "memory_domain": first_entry.get("memory_domain") or second_entry.get("memory_domain"),
                    "query_domain": first_entry.get("query_domain") or second_entry.get("query_domain"),
                    "methods": {
                        first: method_payload(
                            method=first,
                            model_meta=model_meta,
                            hash_id=hash_id,
                            checkpoint_entry=first_entry,
                            sample_cache=sample_cache,
                        ),
                        second: method_payload(
                            method=second,
                            model_meta=model_meta,
                            hash_id=hash_id,
                            checkpoint_entry=second_entry,
                            sample_cache=sample_cache,
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
                    "tie_break": "delta desc, first score desc, second score asc, hash_id asc",
                },
                "model": {
                    "label": model_meta["label"],
                    "model_id": model_id,
                },
                "comparison": {
                    "name": f"{first} -> {second}",
                    "first_method": first,
                    "second_method": second,
                },
                "samples": samples,
            }
            out_path = model_dir / f"{slug(first)}_to_{slug(second)}.json"
            out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            written.append(out_path)

    return written


def clean_text(value: Any) -> str:
    text = "" if value is None else str(value)
    replacements = {
        "\u00a0": " ",
        "\u200b": "",
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "--",
        "\u2014": "---",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": "``",
        "\u201d": "''",
        "\u2022": "-",
        "\u202f": " ",
        "\u2212": "-",
        "\u00d7": "x",
        "\u2192": "->",
        "\u2264": "<=",
        "\u2265": ">=",
        "\ufe0f": "",
        "\u200d": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.encode("ascii", "ignore").decode("ascii")


def tex_escape(value: Any) -> str:
    text = clean_text(value)
    chars = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    text = "".join(chars.get(char, char) for char in text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    paragraphs = []
    for block in re.split(r"\n\s*\n", text):
        block = re.sub(r"\s*\n\s*", " ", block.strip())
        if block:
            paragraphs.append(block)
    return (r"\par " + "\n").join(paragraphs)


def tex_inline(value: Any) -> str:
    text = clean_text(value)
    pattern = re.compile(r"`([^`]+)`|\*\*([^*]+)\*\*|(?<!\*)\*([^*\n]+)\*(?!\*)")
    parts: list[str] = []
    cursor = 0
    for match in pattern.finditer(text):
        parts.append(tex_escape(text[cursor : match.start()]))
        if match.group(1) is not None:
            parts.append(r"\texttt{" + tex_escape(match.group(1)) + "}")
        elif match.group(2) is not None:
            parts.append(r"\textbf{" + tex_inline(match.group(2)) + "}")
        else:
            parts.append(r"\emph{" + tex_escape(match.group(3)) + "}")
        cursor = match.end()
    parts.append(tex_escape(text[cursor:]))
    return "".join(parts)


def render_markdown(value: Any) -> str:
    text = clean_text(value)
    lines = text.splitlines()
    out: list[str] = []
    paragraph_lines: list[str] = []
    list_type: str | None = None

    def flush_paragraph() -> None:
        if paragraph_lines:
            paragraph = " ".join(line.strip() for line in paragraph_lines if line.strip())
            if paragraph:
                out.append(tex_inline(paragraph) + "\n\n")
            paragraph_lines.clear()

    def close_list() -> None:
        nonlocal list_type
        if list_type:
            out.append(f"\\end{{{list_type}}}\n\n")
            list_type = None

    def open_list(kind: str) -> None:
        nonlocal list_type
        if list_type != kind:
            close_list()
            flush_paragraph()
            out.append(f"\\begin{{{kind}}}\n")
            list_type = kind

    def render_table(start: int) -> int:
        close_list()
        flush_paragraph()
        rows: list[list[str]] = []
        i = start
        while i < len(lines) and "|" in lines[i]:
            raw = lines[i].strip()
            cells = [cell.strip() for cell in raw.strip("|").split("|")]
            if cells and not all(re.fullmatch(r":?-{2,}:?", cell or "") for cell in cells):
                rows.append(cells)
            i += 1
        if rows:
            header, *body = rows
            out.append("\\begin{itemize}\n")
            for row in body or rows:
                parts = []
                for index, cell in enumerate(row):
                    label = header[index] if body and index < len(header) else None
                    if label:
                        parts.append(r"\textbf{" + tex_inline(label) + ":} " + tex_inline(cell))
                    else:
                        parts.append(tex_inline(cell))
                out.append("  \\item " + "; ".join(parts) + "\n")
            out.append("\\end{itemize}\n\n")
        return i

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            close_list()
            flush_paragraph()
            i += 1
            continue
        if "|" in stripped and i + 1 < len(lines) and "|" in lines[i + 1] and re.search(r"\|\s*:?-{2,}:?\s*\|", lines[i + 1]):
            i = render_table(i)
            continue
        heading = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading:
            close_list()
            flush_paragraph()
            out.append(r"\paragraph{" + tex_inline(heading.group(2)) + "}" + "\n")
            i += 1
            continue
        if re.fullmatch(r"[-*_]{3,}", stripped):
            close_list()
            flush_paragraph()
            i += 1
            continue
        unordered = re.match(r"^[-*]\s+(.+)$", stripped)
        if unordered:
            open_list("itemize")
            out.append("  \\item " + tex_inline(unordered.group(1)) + "\n")
            i += 1
            continue
        ordered = re.match(r"^\d+[.)]\s+(.+)$", stripped)
        if ordered:
            open_list("enumerate")
            out.append("  \\item " + tex_inline(ordered.group(1)) + "\n")
            i += 1
            continue
        close_list()
        paragraph_lines.append(stripped)
        i += 1

    close_list()
    flush_paragraph()
    return "".join(out).strip()


def paragraph(label: str, value: Any) -> str:
    return f"\\textbf{{{tex_escape(label)}}} {tex_escape(value)}\n\n"


def render_memories(memories: Any, level: int = 0) -> str:
    indent = "  " * level
    if isinstance(memories, list):
        if not memories:
            return f"{indent}\\emph{{No memories.}}\n"
        lines = [f"{indent}\\begin{{itemize}}\n"]
        for item in memories:
            lines.append(f"{indent}  \\item {tex_escape(item)}\n")
        lines.append(f"{indent}\\end{{itemize}}\n")
        return "".join(lines)
    if isinstance(memories, dict):
        if not memories:
            return f"{indent}\\emph{{No memories.}}\n"
        lines: list[str] = []
        for key, child in memories.items():
            lines.append(f"{indent}\\textbf{{{tex_escape(key)}}}\n")
            lines.append(render_memories(child, level))
        return "".join(lines)
    return f"{indent}{tex_escape(memories)}\n"


def method_block(method_name: str, method: dict[str, Any]) -> str:
    return "".join(
        [
            f"\\subsubsection*{{{tex_escape(method_name)}}}\n",
            paragraph("Judge score:", method.get("score")),
            "\\textbf{Memories:}\n",
            render_memories(method.get("memories")),
            "\n\\textbf{Model response:}\n\n",
            render_markdown(method.get("highest_scored_output")),
            "\n\n\\textbf{Judge:}\n\n",
            render_markdown(method.get("judge_explanation", "")),
            "\n\n",
        ]
    )


def sample_box(model_label: str, comparison_name: str, sample: dict[str, Any]) -> str:
    score = sample["score_change"]
    first = score["from_method"]
    second = score["to_method"]
    title = (
        f"{model_label} -- {comparison_name} -- "
        f"Sample {sample['rank']} (+{score['improvement']} Improvement)"
    )
    return "".join(
        [
            "\\clearpage\n",
            f"\\begin{{persistbenchsample}}[title={{{tex_escape(title)}}}]\n",
            paragraph("query:", sample.get("query")),
            method_block(first, sample["methods"][first]),
            method_block(second, sample["methods"][second]),
            "\\end{persistbenchsample}\n",
        ]
    )

def tex_header(title: str, description: str) -> list[str]:
    return [
        "% Generated by analysis/persistbench/method_differences/build_partition_method_comparisons.py\n",
        "% Required in the main preamble: \\usepackage[most]{tcolorbox}\n",
        "% Optional but recommended: \\usepackage[T1]{fontenc}\n\n",
        "\\tcbset{\n",
        "  persistbenchsample/.style={\n",
        "    enhanced,\n",
        "    breakable,\n",
        "    colback=gray!8,\n",
        "    colframe=gray!55!black,\n",
        "    coltitle=black,\n",
        "    colbacktitle=gray!24,\n",
        "    fonttitle=\\bfseries,\n",
        "    boxrule=0.6pt,\n",
        "    arc=1.5mm,\n",
        "    left=2mm,\n",
        "    right=2mm,\n",
        "    top=2mm,\n",
        "    bottom=2mm,\n",
        "    before skip=8pt,\n",
        "    after skip=8pt\n",
        "  }\n",
        "}\n",
        "\\newtcolorbox{persistbenchsample}[1][]{persistbenchsample,#1}\n\n",
        f"\\section*{{{tex_escape(title)}}}\n",
        f"\\noindent {tex_escape(description)}\n\n",
    ]


def build_focused_tex_outputs() -> list[pathlib.Path]:
    outputs: list[pathlib.Path] = []
    description = (
        "Scores are the highest valid judge score across three runs. Scores range "
        "from 1 to 5, and higher is worse. Each sample below is a top cross-domain "
        "case where the second method in the comparison improves over the first."
    )
    for comparison_name, out_path, title in FOCUSED_TEX_OUTPUTS:
        lines = tex_header(title, description)
        for model_id, model_meta in MODELS.items():
            lines.append(f"\\section*{{{tex_escape(model_meta['label'])}}}\n")
            model_dir = SPLIT_DIR / model_meta["slug"]
            matching = None
            for first, second in COMPARISONS:
                if f"{first} -> {second}" == comparison_name:
                    matching = (first, second)
                    break
            if matching is None:
                raise RuntimeError(f"Unknown comparison: {comparison_name}")
            first, second = matching
            path = model_dir / f"{slug(first)}_to_{slug(second)}.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            for sample in data["samples"]:
                lines.append(sample_box(model_meta["label"], comparison_name, sample))

        out_path.write_text("".join(lines), encoding="utf-8")
        outputs.append(out_path)
    return outputs


def build_partitioned_worse_outputs() -> list[pathlib.Path]:
    description = (
        "Scores are the highest valid judge score across three runs. Scores range "
        "from 1 to 5, and higher is worse. This file collects top cross-domain "
        "samples where fixed Partitioned memories are worse than another "
        "partitioning method."
    )
    lines = tex_header("Partitioned Memories Perform Worse", description)
    txt_lines = [
        "Partitioned memories perform worse\n",
        "Score rule: max(valid judge scores across 3 runs); higher is worse.\n",
        "Included comparisons: "
        + ", ".join(PARTITIONED_WORSE_COMPARISONS)
        + "\n\n",
    ]

    for model_id, model_meta in MODELS.items():
        lines.append(f"\\section*{{{tex_escape(model_meta['label'])}}}\n")
        txt_lines.append(f"{model_meta['label']}\n")
        txt_lines.append("-" * len(model_meta["label"]) + "\n")
        model_dir = SPLIT_DIR / model_meta["slug"]

        for comparison_name in PARTITIONED_WORSE_COMPARISONS:
            matching = None
            for first, second in COMPARISONS:
                if f"{first} -> {second}" == comparison_name:
                    matching = (first, second)
                    break
            if matching is None:
                raise RuntimeError(f"Unknown comparison: {comparison_name}")
            first, second = matching
            path = model_dir / f"{slug(first)}_to_{slug(second)}.json"
            data = json.loads(path.read_text(encoding="utf-8"))

            lines.append(f"\\subsection*{{{tex_escape(comparison_name)}}}\n")
            txt_lines.append(f"\n{comparison_name}\n")
            for sample in data["samples"]:
                score = sample["score_change"]
                txt_lines.append(
                    f"  {sample['rank']}. {sample['hash_id']}  "
                    f"{score['from_score']}->{score['to_score']}  "
                    f"+{score['improvement']}\n"
                )
                lines.append(sample_box(model_meta["label"], comparison_name, sample))
        txt_lines.append("\n")

    PARTITIONED_WORSE_TEX_OUT.write_text("".join(lines), encoding="utf-8")
    PARTITIONED_WORSE_TXT_OUT.write_text("".join(txt_lines), encoding="utf-8")
    return [PARTITIONED_WORSE_TEX_OUT, PARTITIONED_WORSE_TXT_OUT]


def build_tex() -> None:
    lines = tex_header(
        "Partition Method Differences: Top Cross-Domain Improvements",
        "Scores are the highest valid judge score across three runs. Scores range from 1 to 5, and higher is worse. Each box compares the highest-scored responses for one method pair on the same query.",
    )

    for model_id, model_meta in MODELS.items():
        lines.append(f"\\section*{{{tex_escape(model_meta['label'])}}}\n")
        model_dir = SPLIT_DIR / model_meta["slug"]
        for first, second in COMPARISONS:
            path = model_dir / f"{slug(first)}_to_{slug(second)}.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            comparison_name = data["comparison"]["name"]
            lines.append(f"\\subsection*{{{tex_escape(comparison_name)}}}\n")
            for sample in data["samples"]:
                lines.append(sample_box(model_meta["label"], comparison_name, sample))

    TEX_OUT.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    written = build_comparison_payloads()
    build_tex()
    focused_tex = build_focused_tex_outputs()
    partitioned_worse_outputs = build_partitioned_worse_outputs()
    for path in written:
        print(path.relative_to(REPO_ROOT))
    print(f"Wrote {len(written)} JSON files")
    print(TEX_OUT.relative_to(REPO_ROOT))
    for path in focused_tex:
        print(path.relative_to(REPO_ROOT))
    for path in partitioned_worse_outputs:
        print(path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
