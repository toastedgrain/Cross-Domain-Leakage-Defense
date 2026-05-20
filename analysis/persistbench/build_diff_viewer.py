#!/usr/bin/env python3
"""Build JSON and a self-contained HTML viewer for top score-diff samples."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys
from typing import Any


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.persistbench.output_manifest import (
    BASELINE_CHECKPOINTS,
    PARTITIONED_CHECKPOINTS,
    PARTITIONED_COS_CHECKPOINTS,
    PARTITIONED_CUSTOM_CHECKPOINTS,
    SAMPLES,
    TREE_CHECKPOINTS,
)

MODELS: dict[str, dict[str, str]] = {
    "gpt-oss-120b": {
        "label": "GPT-OSS-120B",
        "sample_dir": "gpt-oss-120b",
    },
    "zai-org/glm-4.7-maas": {
        "label": "GLM-4.7",
        "sample_dir": "zai-org_glm-4.7-maas",
    },
}

METHODS: dict[str, dict[str, Any]] = {
    "Baseline": {
        "checkpoint": BASELINE_CHECKPOINTS[0],
        "sample_path": None,
        "memory_format": "flat_list",
    },
    "Partitioned": {
        "checkpoint": PARTITIONED_CHECKPOINTS[0],
        "sample_path": lambda sample_dir: SAMPLES / "partitioned" / sample_dir / "full_benchmark.jsonl",
        "memory_format": "partitioned",
    },
    "Partitioned_COS": {
        "checkpoint": PARTITIONED_COS_CHECKPOINTS[0],
        "sample_path": lambda sample_dir: SAMPLES / "partitioned" / "cos_similarity" / "full_benchmark.jsonl",
        "memory_format": "partitioned_cos",
    },
    "Partitioned_Custom": {
        "checkpoint": PARTITIONED_CUSTOM_CHECKPOINTS[0],
        "sample_path": lambda sample_dir: SAMPLES / "partitioned_custom_categories" / sample_dir / "full_benchmark.jsonl",
        "memory_format": "partitioned_custom",
    },
    "Tree": {
        "checkpoint": TREE_CHECKPOINTS[0],
        "sample_path": lambda sample_dir: SAMPLES / "tree" / sample_dir / "full_benchmark.jsonl",
        "memory_format": "tree",
    },
}

COMPARISONS = [
    ("Baseline", "Partitioned"),
    ("Baseline", "Partitioned_COS"),
    ("Baseline", "Partitioned_Custom"),
    ("Baseline", "Tree"),
    ("Partitioned", "Partitioned_Custom"),
]


def load_checkpoint(path: pathlib.Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl_by_hash(path: pathlib.Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}

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


def normalize_failure_type(entry: dict[str, Any]) -> str:
    failure_type = entry.get("failure_type") or entry.get("leakage_type") or "cross_domain"
    if failure_type == "positive_memory_usage":
        return "beneficial_memory_usage"
    return failure_type


def score_generation(generation: dict[str, Any]) -> int | None:
    if generation.get("error") or not generation.get("judge"):
        return None
    score = generation["judge"].get("score")
    return score if isinstance(score, int) else None


def best_generation(entry: dict[str, Any], model_id: str) -> dict[str, Any]:
    model_data = entry.get("results", {}).get(model_id) or {}
    generations = model_data.get("generations", [])

    valid: list[tuple[int, int, dict[str, Any]]] = []
    all_scores: list[int | None] = []
    for index, generation in enumerate(generations):
        score = score_generation(generation)
        all_scores.append(score)
        if score is not None:
            generation_index = generation.get("generation_index", index)
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


def source_entry(
    method: str,
    model_meta: dict[str, str],
    hash_id: str,
    checkpoint_entry: dict[str, Any],
    sample_cache: dict[tuple[str, str], tuple[pathlib.Path | None, dict[str, dict[str, Any]]]],
) -> tuple[pathlib.Path | None, dict[str, Any]]:
    sample_path_factory = METHODS[method]["sample_path"]
    if sample_path_factory is None:
        return None, checkpoint_entry

    sample_dir = model_meta["sample_dir"]
    cache_key = (method, sample_dir)
    if cache_key not in sample_cache:
        sample_path = sample_path_factory(sample_dir)
        sample_cache[cache_key] = (sample_path, load_jsonl_by_hash(sample_path))

    sample_path, rows = sample_cache[cache_key]
    return sample_path, rows.get(hash_id, checkpoint_entry)


def build_data() -> dict[str, Any]:
    checkpoints = {
        method: load_checkpoint(config["checkpoint"])
        for method, config in METHODS.items()
    }
    sample_cache: dict[tuple[str, str], tuple[pathlib.Path | None, dict[str, dict[str, Any]]]] = {}

    data: dict[str, Any] = {
        "metadata": {
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "score_rule": "Per method/model/sample score is max(valid judge scores across the 3 runs). Higher score is worse, so improvement = first_method_score - second_method_score.",
            "failure_type": "cross_domain",
            "top_n": 5,
            "tie_break": "delta desc, first score desc, second score asc, hash_id asc",
            "methods": {
                method: {
                    "checkpoint": str(config["checkpoint"].relative_to(REPO_ROOT)),
                    "memory_format": config["memory_format"],
                }
                for method, config in METHODS.items()
            },
        },
        "models": [],
    }

    for model_id, model_meta in MODELS.items():
        model_block: dict[str, Any] = {
            "model_id": model_id,
            "label": model_meta["label"],
            "comparisons": [],
        }

        for first, second in COMPARISONS:
            first_entries = checkpoints[first].get("entries", {})
            second_entries = checkpoints[second].get("entries", {})
            shared_ids = set(first_entries) & set(second_entries)
            candidates: list[tuple[int, int, int, str]] = []

            for hash_id in shared_ids:
                first_entry = first_entries[hash_id]
                second_entry = second_entries[hash_id]
                if normalize_failure_type(first_entry) != "cross_domain":
                    continue
                if normalize_failure_type(second_entry) != "cross_domain":
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
                method_payloads: dict[str, Any] = {}
                query = None
                memory_domain = None
                query_domain = None

                for method in (first, second):
                    checkpoint_entry = checkpoints[method]["entries"][hash_id]
                    sample_path, sample_entry = source_entry(
                        method,
                        model_meta,
                        hash_id,
                        checkpoint_entry,
                        sample_cache,
                    )
                    best = best_generation(checkpoint_entry, model_id)

                    query = query or sample_entry.get("query") or checkpoint_entry.get("query")
                    memory_domain = memory_domain or sample_entry.get("memory_domain") or checkpoint_entry.get("memory_domain")
                    query_domain = query_domain or sample_entry.get("query_domain") or checkpoint_entry.get("query_domain")

                    method_payloads[method] = {
                        "method": method,
                        "score": best["score"],
                        "generation_index": best["generation_index"],
                        "all_scores": best["all_scores"],
                        "output": best["output"],
                        "judge": best["judge"],
                        "memories": sample_entry.get("memories", checkpoint_entry.get("memories")),
                        "memory_format": METHODS[method]["memory_format"],
                        "sample_source": str(sample_path.relative_to(REPO_ROOT)) if sample_path else str(METHODS[method]["checkpoint"].relative_to(REPO_ROOT)),
                    }

                samples.append(
                    {
                        "rank": rank,
                        "hash_id": hash_id,
                        "query": query,
                        "memory_domain": memory_domain,
                        "query_domain": query_domain,
                        "improvement": delta,
                        "first_score": first_score,
                        "second_score": second_score,
                        "methods": method_payloads,
                    }
                )

            model_block["comparisons"].append(
                {
                    "name": f"{first} -> {second}",
                    "first_method": first,
                    "second_method": second,
                    "samples": samples,
                }
            )

        data["models"].append(model_block)

    return data


def html_template(data_json: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PersistBench Score-Diff Viewer</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #060912;
      --panel: rgba(13, 20, 35, 0.92);
      --panel-2: rgba(16, 28, 45, 0.88);
      --line: rgba(131, 238, 255, 0.24);
      --text: #ecf7ff;
      --muted: #93a8ba;
      --cyan: #42f5ef;
      --green: #7dff9b;
      --pink: #ff5bd6;
      --amber: #ffcf66;
      --red: #ff6b7a;
      --shadow: rgba(0, 0, 0, 0.38);
      --radius: 8px;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}

    * {{ box-sizing: border-box; }}

    body {{
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      background:
        linear-gradient(rgba(66, 245, 239, 0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(66, 245, 239, 0.035) 1px, transparent 1px),
        radial-gradient(circle at 18% 12%, rgba(255, 91, 214, 0.16), transparent 28rem),
        radial-gradient(circle at 88% 18%, rgba(66, 245, 239, 0.14), transparent 30rem),
        linear-gradient(135deg, #050713 0%, #09111f 48%, #050812 100%);
      background-size: 34px 34px, 34px 34px, auto, auto, auto;
      overflow: hidden;
    }}

    button, input, select {{
      font: inherit;
    }}

    .app {{
      display: grid;
      grid-template-columns: 300px minmax(0, 1fr);
      height: 100vh;
    }}

    .sidebar {{
      border-right: 1px solid var(--line);
      background: rgba(6, 10, 20, 0.76);
      backdrop-filter: blur(18px);
      padding: 18px;
      overflow: auto;
      box-shadow: 12px 0 36px var(--shadow);
    }}

    .brand {{
      margin-bottom: 18px;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--line);
    }}

    .brand h1 {{
      margin: 0;
      font-size: 18px;
      letter-spacing: 0;
    }}

    .brand p {{
      margin: 7px 0 0;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }}

    .nav-section {{
      margin: 18px 0;
    }}

    .nav-title {{
      color: var(--cyan);
      font-size: 11px;
      font-weight: 800;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      margin: 0 0 10px;
    }}

    .tab-list {{
      display: grid;
      gap: 8px;
    }}

    .tab {{
      width: 100%;
      min-height: 38px;
      border: 1px solid rgba(131, 238, 255, 0.18);
      border-radius: var(--radius);
      color: var(--text);
      background: rgba(255, 255, 255, 0.035);
      text-align: left;
      padding: 9px 10px;
      cursor: pointer;
    }}

    .tab:hover {{
      border-color: rgba(66, 245, 239, 0.65);
    }}

    .tab.active {{
      border-color: var(--cyan);
      background: linear-gradient(90deg, rgba(66, 245, 239, 0.18), rgba(255, 91, 214, 0.10));
      box-shadow: 0 0 18px rgba(66, 245, 239, 0.16);
    }}

    .sample-list {{
      display: grid;
      gap: 8px;
      max-height: 34vh;
      overflow: auto;
      padding-right: 3px;
    }}

    .sample-button {{
      display: grid;
      gap: 5px;
      width: 100%;
      border: 1px solid rgba(131, 238, 255, 0.16);
      border-radius: var(--radius);
      background: rgba(255, 255, 255, 0.035);
      color: var(--text);
      padding: 10px;
      text-align: left;
      cursor: pointer;
    }}

    .sample-button.active {{
      border-color: var(--pink);
      background: rgba(255, 91, 214, 0.12);
    }}

    .hash {{
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
      font-size: 11px;
      color: #d6faff;
      word-break: break-all;
    }}

    .metric-row {{
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 12px;
    }}

    .badge {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 23px;
      border: 1px solid rgba(125, 255, 155, 0.38);
      border-radius: 999px;
      padding: 2px 8px;
      color: var(--green);
      background: rgba(125, 255, 155, 0.08);
      font-weight: 800;
      font-size: 12px;
    }}

    main {{
      min-width: 0;
      overflow: auto;
      padding: 22px;
    }}

    .topbar {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 16px;
    }}

    .title-block h2 {{
      margin: 0 0 8px;
      font-size: 24px;
      letter-spacing: 0;
    }}

    .title-block .subtitle {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }}

    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 8px;
    }}

    .control {{
      min-height: 36px;
      border: 1px solid rgba(131, 238, 255, 0.22);
      border-radius: var(--radius);
      background: rgba(13, 20, 35, 0.82);
      color: var(--text);
      padding: 7px 10px;
    }}

    button.control {{
      cursor: pointer;
    }}

    button.control:hover {{
      border-color: var(--cyan);
    }}

    .query-panel {{
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: var(--panel);
      padding: 16px;
      margin-bottom: 16px;
      box-shadow: 0 16px 40px var(--shadow);
    }}

    .query-panel h3, .method-panel h3 {{
      margin: 0 0 10px;
      font-size: 13px;
      color: var(--cyan);
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}

    .query-text {{
      margin: 0;
      line-height: 1.55;
      white-space: pre-wrap;
    }}

    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-top: 14px;
    }}

    .stat {{
      border: 1px solid rgba(131, 238, 255, 0.16);
      border-radius: var(--radius);
      background: rgba(255, 255, 255, 0.035);
      padding: 10px;
      min-width: 0;
    }}

    .stat .label {{
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.1em;
    }}

    .stat .value {{
      margin-top: 5px;
      color: var(--text);
      font-size: 14px;
      overflow-wrap: anywhere;
    }}

    .compare-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      align-items: start;
    }}

    .method-panel {{
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: var(--panel-2);
      box-shadow: 0 16px 40px var(--shadow);
      min-width: 0;
      overflow: hidden;
    }}

    .method-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(90deg, rgba(66, 245, 239, 0.10), rgba(255, 91, 214, 0.08));
    }}

    .method-name {{
      font-size: 16px;
      font-weight: 850;
    }}

    .score-pill {{
      min-width: 66px;
      border: 1px solid rgba(255, 207, 102, 0.48);
      border-radius: 999px;
      color: var(--amber);
      background: rgba(255, 207, 102, 0.08);
      padding: 5px 10px;
      text-align: center;
      font-weight: 850;
    }}

    .method-body {{
      padding: 16px;
      display: grid;
      gap: 14px;
    }}

    .section {{
      min-width: 0;
    }}

    .section-title {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      margin-bottom: 8px;
    }}

    .content-box {{
      border: 1px solid rgba(131, 238, 255, 0.14);
      border-radius: var(--radius);
      background: rgba(1, 5, 12, 0.42);
      padding: 12px;
      max-height: 330px;
      overflow: auto;
      line-height: 1.52;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }}

    .memory-tree {{
      display: grid;
      gap: 10px;
      white-space: normal;
    }}

    .memory-group {{
      border: 1px solid rgba(131, 238, 255, 0.12);
      border-radius: var(--radius);
      padding: 10px;
      background: rgba(255, 255, 255, 0.025);
    }}

    .memory-group-title {{
      color: var(--cyan);
      font-weight: 850;
      margin-bottom: 7px;
    }}

    .memory-subtitle {{
      color: var(--pink);
      font-weight: 800;
      margin: 8px 0 5px;
      font-size: 12px;
    }}

    .memory-item {{
      margin: 5px 0;
      color: #d9e9f3;
    }}

    .empty {{
      color: #5e7282;
      font-style: italic;
    }}

    .judge {{
      color: #cbdbe6;
      font-size: 13px;
    }}

    @media (max-width: 980px) {{
      body {{
        overflow: auto;
      }}
      .app {{
        grid-template-columns: 1fr;
        height: auto;
      }}
      .sidebar {{
        position: relative;
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }}
      .sample-list {{
        max-height: 220px;
      }}
      .topbar {{
        display: grid;
      }}
      .toolbar {{
        justify-content: flex-start;
      }}
      .compare-grid, .meta-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <div class="brand">
        <h1>PersistBench Diff Viewer</h1>
        <p>Highest-score run per method. Lower second-side score means the structured memory setup improved the sample.</p>
      </div>

      <section class="nav-section">
        <p class="nav-title">Model</p>
        <div id="modelTabs" class="tab-list"></div>
      </section>

      <section class="nav-section">
        <p class="nav-title">Comparison</p>
        <div id="comparisonTabs" class="tab-list"></div>
      </section>

      <section class="nav-section">
        <p class="nav-title">Samples</p>
        <div id="sampleList" class="sample-list"></div>
      </section>
    </aside>

    <main>
      <div class="topbar">
        <div class="title-block">
          <h2 id="pageTitle"></h2>
          <div id="pageSubtitle" class="subtitle"></div>
        </div>
        <div class="toolbar">
          <button id="prevSample" class="control" type="button">Prev Sample</button>
          <button id="nextSample" class="control" type="button">Next Sample</button>
          <select id="sampleJump" class="control" aria-label="Jump to sample"></select>
        </div>
      </div>

      <section class="query-panel">
        <h3>Query</h3>
        <p id="queryText" class="query-text"></p>
        <div class="meta-grid">
          <div class="stat"><div class="label">Hash</div><div id="hashValue" class="value hash"></div></div>
          <div class="stat"><div class="label">Improvement</div><div id="improvementValue" class="value"></div></div>
          <div class="stat"><div class="label">Memory Domain</div><div id="memoryDomain" class="value"></div></div>
          <div class="stat"><div class="label">Query Domain</div><div id="queryDomain" class="value"></div></div>
        </div>
      </section>

      <section id="methodGrid" class="compare-grid"></section>
    </main>
  </div>

  <script id="diff-data" type="application/json">{data_json}</script>
  <script>
    const DATA = JSON.parse(document.getElementById('diff-data').textContent);
    const state = {{ model: 0, comparison: 0, sample: 0 }};

    const el = (id) => document.getElementById(id);
    const escapeText = (value) => String(value ?? '');

    function scoreClass(score) {{
      if (score >= 4) return 'var(--red)';
      if (score === 3) return 'var(--amber)';
      return 'var(--green)';
    }}

    function renderTabs(container, items, activeIndex, onClick, labelFn) {{
      container.innerHTML = '';
      items.forEach((item, index) => {{
        const button = document.createElement('button');
        button.className = 'tab' + (index === activeIndex ? ' active' : '');
        button.type = 'button';
        button.textContent = labelFn(item, index);
        button.addEventListener('click', () => onClick(index));
        container.appendChild(button);
      }});
    }}

    function renderSampleList(samples) {{
      const list = el('sampleList');
      list.innerHTML = '';
      samples.forEach((sample, index) => {{
        const button = document.createElement('button');
        button.className = 'sample-button' + (index === state.sample ? ' active' : '');
        button.type = 'button';
        button.innerHTML = `
          <span class="hash">#${{sample.rank}} ${{sample.hash_id}}</span>
          <span class="metric-row"><span class="badge">+${{sample.improvement}}</span> ${{sample.first_score}} -> ${{sample.second_score}}</span>
        `;
        button.addEventListener('click', () => {{
          state.sample = index;
          render();
        }});
        list.appendChild(button);
      }});
    }}

    function renderJump(samples) {{
      const jump = el('sampleJump');
      jump.innerHTML = '';
      samples.forEach((sample, index) => {{
        const option = document.createElement('option');
        option.value = String(index);
        option.textContent = `#${{sample.rank}} ${{sample.hash_id.slice(0, 10)}} +${{sample.improvement}}`;
        jump.appendChild(option);
      }});
      jump.value = String(state.sample);
    }}

    function countMemories(memories) {{
      if (Array.isArray(memories)) return memories.filter(Boolean).length;
      if (!memories || typeof memories !== 'object') return 0;
      let count = 0;
      for (const value of Object.values(memories)) {{
        if (Array.isArray(value)) {{
          count += value.filter(Boolean).length;
        }} else if (value && typeof value === 'object') {{
          count += countMemories(value);
        }} else if (value) {{
          count += 1;
        }}
      }}
      return count;
    }}

    function memoryHtml(memories) {{
      if (Array.isArray(memories)) {{
        if (!memories.length) return '<div class="empty">No memories</div>';
        return '<div class="memory-tree">' + memories.map((item) => `<div class="memory-item">${{escapeHtml(item)}}</div>`).join('') + '</div>';
      }}
      if (!memories || typeof memories !== 'object') {{
        return `<div class="memory-item">${{escapeHtml(memories)}}</div>`;
      }}
      const groups = Object.entries(memories).map(([key, value]) => {{
        if (Array.isArray(value)) {{
          const rows = value.length
            ? value.map((item) => `<div class="memory-item">${{escapeHtml(item)}}</div>`).join('')
            : '<div class="empty">Empty</div>';
          return `<div class="memory-group"><div class="memory-group-title">${{escapeHtml(key)}}</div>${{rows}}</div>`;
        }}
        if (value && typeof value === 'object') {{
          const children = Object.entries(value).map(([childKey, childValue]) => {{
            const items = Array.isArray(childValue) && childValue.length
              ? childValue.map((item) => `<div class="memory-item">${{escapeHtml(item)}}</div>`).join('')
              : '<div class="empty">Empty</div>';
            return `<div class="memory-subtitle">${{escapeHtml(childKey)}}</div>${{items}}`;
          }}).join('');
          return `<div class="memory-group"><div class="memory-group-title">${{escapeHtml(key)}}</div>${{children}}</div>`;
        }}
        return `<div class="memory-group"><div class="memory-group-title">${{escapeHtml(key)}}</div><div class="memory-item">${{escapeHtml(value)}}</div></div>`;
      }}).join('');
      return `<div class="memory-tree">${{groups}}</div>`;
    }}

    function escapeHtml(value) {{
      return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
    }}

    function renderMethodPanel(method) {{
      const judge = method.judge || {{}};
      const explanation = judge.explanation || judge.reasoning || judge.rationale || '';
      const score = method.score ?? 'NA';
      const panel = document.createElement('article');
      panel.className = 'method-panel';
      panel.innerHTML = `
        <div class="method-head">
          <div>
            <div class="method-name">${{escapeHtml(method.method)}}</div>
            <div class="metric-row">${{escapeHtml(method.memory_format)}} / run ${{escapeHtml(method.generation_index)}} / scores [${{method.all_scores.map((s) => s ?? 'NA').join(', ')}}]</div>
          </div>
          <div class="score-pill" style="color:${{scoreClass(method.score)}}">Score ${{score}}</div>
        </div>
        <div class="method-body">
          <div class="section">
            <div class="section-title"><span>Memories</span><span>${{countMemories(method.memories)}} items</span></div>
            <div class="content-box">${{memoryHtml(method.memories)}}</div>
          </div>
          <div class="section">
            <div class="section-title"><span>Highest-Score Output</span><span>${{escapeHtml(method.sample_source)}}</span></div>
            <div class="content-box">${{escapeHtml(method.output)}}</div>
          </div>
          <div class="section">
            <div class="section-title"><span>Judge</span><span>score ${{score}}</span></div>
            <div class="content-box judge">${{escapeHtml(explanation || JSON.stringify(judge, null, 2))}}</div>
          </div>
        </div>
      `;
      return panel;
    }}

    function render() {{
      const model = DATA.models[state.model];
      const comparison = model.comparisons[state.comparison];
      if (state.sample >= comparison.samples.length) state.sample = comparison.samples.length - 1;
      if (state.sample < 0) state.sample = 0;
      const sample = comparison.samples[state.sample];

      renderTabs(el('modelTabs'), DATA.models, state.model, (index) => {{
        state.model = index;
        state.comparison = 0;
        state.sample = 0;
        render();
      }}, (item) => item.label);

      renderTabs(el('comparisonTabs'), model.comparisons, state.comparison, (index) => {{
        state.comparison = index;
        state.sample = 0;
        render();
      }}, (item) => item.name);

      renderSampleList(comparison.samples);
      renderJump(comparison.samples);

      el('pageTitle').textContent = `${{model.label}} / ${{comparison.name}}`;
      el('pageSubtitle').textContent = DATA.metadata.score_rule;
      el('queryText').textContent = sample.query || '';
      el('hashValue').textContent = sample.hash_id;
      el('improvementValue').textContent = `${{sample.first_score}} -> ${{sample.second_score}}  (+${{sample.improvement}})`;
      el('memoryDomain').textContent = sample.memory_domain || 'Unknown';
      el('queryDomain').textContent = sample.query_domain || 'Unknown';

      const grid = el('methodGrid');
      grid.innerHTML = '';
      grid.appendChild(renderMethodPanel(sample.methods[comparison.first_method]));
      grid.appendChild(renderMethodPanel(sample.methods[comparison.second_method]));
    }}

    el('prevSample').addEventListener('click', () => {{
      const samples = DATA.models[state.model].comparisons[state.comparison].samples;
      state.sample = (state.sample - 1 + samples.length) % samples.length;
      render();
    }});

    el('nextSample').addEventListener('click', () => {{
      const samples = DATA.models[state.model].comparisons[state.comparison].samples;
      state.sample = (state.sample + 1) % samples.length;
      render();
    }});

    el('sampleJump').addEventListener('change', (event) => {{
      state.sample = Number(event.target.value);
      render();
    }});

    window.addEventListener('keydown', (event) => {{
      if (event.key === 'ArrowLeft') el('prevSample').click();
      if (event.key === 'ArrowRight') el('nextSample').click();
    }});

    render();
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build diff_samples.json and diff_viewer.html")
    parser.add_argument("--json-out", type=pathlib.Path, default=REPO_ROOT / "diff_samples.json")
    parser.add_argument("--html-out", type=pathlib.Path, default=REPO_ROOT / "diff_viewer.html")
    args = parser.parse_args()

    data = build_data()
    args.json_out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    compact = json.dumps(data, ensure_ascii=False)
    args.html_out.write_text(html_template(compact), encoding="utf-8")

    print(f"Wrote {args.json_out}")
    print(f"Wrote {args.html_out}")


if __name__ == "__main__":
    main()
