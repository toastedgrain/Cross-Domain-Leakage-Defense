#!/usr/bin/env python3
"""Generate a LaTeX fragment for Baseline vs Partitioned diff samples."""

from __future__ import annotations

import json
import pathlib
import re
from typing import Any


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT_PATH = REPO_ROOT / "baseline_vs_partitoned.tex"

INPUTS = [
    REPO_ROOT / "diff_samples_split" / "glm_4_7" / "baseline_to_partitioned.json",
    REPO_ROOT / "diff_samples_split" / "gpt_oss_120b" / "baseline_to_partitioned.json",
]


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
    text = text.encode("ascii", "ignore").decode("ascii")
    return text


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
    """Escape inline text and render a small subset of Markdown emphasis."""
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
    """Render common README/Markdown blocks into LaTeX without fragile line breaks."""
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
        line = lines[i].rstrip()
        stripped = line.strip()

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
        lines = []
        for key, value in memories.items():
            lines.append(f"{indent}\\textbf{{{tex_escape(key)}}}\n")
            lines.append(render_memories(value, level))
        return "".join(lines)

    return f"{indent}{tex_escape(memories)}\n"


def method_block(method_name: str, method: dict[str, Any]) -> str:
    lines = [
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
    return "".join(lines)


def sample_box(model_label: str, comparison_name: str, sample: dict[str, Any]) -> str:
    score = sample["score_change"]
    methods = sample["methods"]
    title = f"{model_label} -- Sample {sample['rank']} (+{score['improvement']} Improvement)"

    lines = [
        "\\clearpage\n",
        f"\\begin{{persistbenchsample}}[title={{{tex_escape(title)}}}]\n",
        paragraph("query:", sample.get("query")),
        method_block("Baseline", methods["Baseline"]),
        method_block("Partitioned", methods["Partitioned"]),
        "\\end{persistbenchsample}\n",
    ]
    return "".join(lines)


def main() -> None:
    lines = [
        "% Generated by analysis/persistbench/build_baseline_vs_partitioned_tex.py\n",
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
        "\\section*{Baseline vs. Partitioned: Top Cross-Domain Improvements}\n",
        "\\noindent Scores are the highest valid judge score across three runs. Scores range from 1 to 5, and higher is worse. Each sample below compares the highest-scored Baseline response against the highest-scored Partitioned response for the same query.\n\n",
    ]

    for input_path in INPUTS:
        data = json.loads(input_path.read_text(encoding="utf-8"))
        model_label = data["model"]["label"]
        comparison_name = data["comparison"]["name"]
        lines.append(f"\\section*{{{tex_escape(model_label)}}}\n")
        for sample in data["samples"]:
            lines.append(sample_box(model_label, comparison_name, sample))

    OUT_PATH.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
