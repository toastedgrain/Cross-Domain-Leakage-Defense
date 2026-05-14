"""PersistBench checkpoint locations used by analysis scripts.

Checkpoints are discovered dynamically by globbing ``outputs/persistbench``.
Adding a new method is a matter of dropping JSON files into the right
sub-directory (and, if the existing rules don't match it, adding one row to
``METHOD_PATTERNS``).
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = REPO_ROOT / "outputs" / "persistbench"
SAMPLES = REPO_ROOT / "benchmark_samples" / "persistbench"
DEFAULT_BENCHMARK = SAMPLES / "baseline" / "full_benchmark.jsonl"


def _collect(
    subdir: str,
    *includes: str,
    excludes: tuple[str, ...] = (),
    recursive: bool = False,
) -> tuple[Path, ...]:
    """Sorted ``*.json`` files in ``OUTPUTS/<subdir>``.

    Filename must contain every string in ``includes`` and none in
    ``excludes``. Non-recursive by default so helper subdirs like
    ``rag/fine_tuning`` don't bleed into the main view.
    """
    base = OUTPUTS / subdir
    if not base.exists():
        return ()
    paths = base.rglob("*.json") if recursive else base.glob("*.json")
    return tuple(
        sorted(
            p
            for p in paths
            if all(s in p.name for s in includes)
            and not any(s in p.name for s in excludes)
        )
    )


# ---------------------------------------------------------------------------
# Memory-structure groupings (one per method-of-storing-memory)
# ---------------------------------------------------------------------------

BASELINE_CHECKPOINTS = _collect("baseline")
PARTITIONED_CHECKPOINTS = _collect("partitioned")
PARTITIONED_COS_CHECKPOINTS = _collect("partitioned_cos_similarity")
PARTITIONED_CUSTOM_CHECKPOINTS = _collect("partitioned_custom_categories")
TREE_CHECKPOINTS = _collect("tree", "tree_informed")

MEMORY_STRUCTURE_CHECKPOINTS: dict[str, tuple[Path, ...]] = {
    "flat_memory_list": BASELINE_CHECKPOINTS,
    "partitions": PARTITIONED_CHECKPOINTS,
    "cosine_similarity_partitions": PARTITIONED_COS_CHECKPOINTS,
    "dynamic_partitions": PARTITIONED_CUSTOM_CHECKPOINTS,
    "2_level_tree": TREE_CHECKPOINTS,
}

_MEMORY_STRUCTURE_LABELS = {
    "flat_memory_list": "Flat List",
    "partitions": "Inference Fixed Partitions",
    "cosine_similarity_partitions": "Cosine Similarity Partitions",
    "dynamic_partitions": "Inference Dynamic Partitions",
    "2_level_tree": "2-Level Tree",
}

MEMORY_STRUCTURES = {
    key: {
        "label": _MEMORY_STRUCTURE_LABELS[key],
        "checkpoints": list(paths),
    }
    for key, paths in MEMORY_STRUCTURE_CHECKPOINTS.items()
}

_CHECKPOINT_LABEL_OVERRIDES = {
    "flat_memory_list": "Flat Memory List",
    "partitions": "Inference Fixed Partitions",
    "cosine_similarity_partitions": "Cosine Similarity Partitions",
    "dynamic_partitions": "Inference Dynamic Partitions",
    "2_level_tree": "2-Level Tree",
}

CHECKPOINT_LABELS = {
    path.name: _CHECKPOINT_LABEL_OVERRIDES[key]
    for key, paths in MEMORY_STRUCTURE_CHECKPOINTS.items()
    for path in paths
}


# ---------------------------------------------------------------------------
# Method × condition table
#
# Each row: (display name, sub-directory, required filename substrings,
# excluded filename substrings). Empty entries (missing dir / no matches)
# are filtered out at build time.
# ---------------------------------------------------------------------------

METHOD_PATTERNS: tuple[tuple[str, str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("baseline", "baseline", (), ()),
    ("defence: permissive", "defence", ("permissive",), ()),
    ("defence: restrictive", "defence", ("restrictive",), ()),
    ("defence: rubric_informed", "defence", ("rubric_informed",), ()),
    ("defence: gepa_optimized", "defence", ("gepa",), ()),
    ("partitioned", "partitioned", (), ()),
    ("partitioned + permissive", "partitioned_defence", ("permissive",), ()),
    ("partitioned + restrictive", "partitioned_defence", ("restrictive",), ()),
    ("partitioned + rubric_informed", "partitioned_defence", ("rubric_informed",), ()),
    ("partitioned + gepa_optimized", "partitioned_defence", ("gepa",), ()),
    ("partitioned (cosine similarity)", "partitioned_cos_similarity", (), ()),
    ("partitioned (labeled)", "partitioned_label", (), ()),
    ("partitioned (custom cats)", "partitioned_custom_categories", (), ()),
    ("partitioned (custom) + permissive", "partitioned_custom_categories_defence", ("permissive",), ()),
    ("partitioned (custom) + restrictive", "partitioned_custom_categories_defence", ("restrictive",), ()),
    ("partitioned (custom) + rubric_informed", "partitioned_custom_categories_defence", ("rubric_informed",), ()),
    ("partitioned (custom) + gepa_optimized", "partitioned_custom_categories_defence", ("gepa",), ()),
    ("rag tau=0.25", "rag", ("tau0.25_",), ()),
    ("rag tau=0.50", "rag", ("tau0.5_",), ()),
    ("rag tau=0.75", "rag", ("tau0.75_",), ()),
    ("rag tau=0.25 + permissive", "rag_defence", ("tau0.25_", "permissive"), ()),
    ("rag tau=0.25 + restrictive", "rag_defence", ("tau0.25_", "restrictive"), ()),
    ("rag tau=0.25 + rubric_informed", "rag_defence", ("tau0.25_", "rubric_informed"), ()),
    ("rag tau=0.25 + gepa_optimized", "rag_defence", ("tau0.25_", "gepa"), ()),
    ("rag tau=0.50 + permissive", "rag_defence", ("tau0.5_", "permissive"), ()),
    ("rag tau=0.50 + restrictive", "rag_defence", ("tau0.5_", "restrictive"), ()),
    ("rag tau=0.50 + rubric_informed", "rag_defence", ("tau0.5_", "rubric_informed"), ()),
    ("rag tau=0.50 + gepa_optimized", "rag_defence", ("tau0.5_", "gepa"), ()),
    ("informed_tree", "tree", ("tree_informed",), ()),
    ("tree + permissive", "tree_defence", ("permissive",), ()),
    ("tree + restrictive", "tree_defence", ("restrictive",), ()),
    ("tree + rubric_informed", "tree_defence", ("rubric_informed",), ()),
    ("tree + gepa_optimized", "tree_defence", ("gepa",), ()),
)


def _build_method_checkpoints() -> dict[str, tuple[Path, ...]]:
    out: dict[str, tuple[Path, ...]] = {}
    for name, subdir, includes, excludes in METHOD_PATTERNS:
        paths = _collect(subdir, *includes, excludes=excludes)
        if paths:
            out[name] = paths
    return out


METHOD_CHECKPOINTS: dict[str, tuple[Path, ...]] = _build_method_checkpoints()


# Every JSON checkpoint anywhere under ``outputs/persistbench`` (recursive),
# for callers that want the full dataset regardless of method grouping.
ALL_CHECKPOINTS: tuple[Path, ...] = tuple(sorted(OUTPUTS.rglob("*.json")))


# ---------------------------------------------------------------------------
# Model display labels (canonical → pretty)
# ---------------------------------------------------------------------------

MODEL_LABELS = {
    "DeepSeek-V3.2": "DeepSeek V3.2",
    "gpt-oss-120b": "GPT-OSS 120B",
    "google/gemini-3-pro-preview": "Gemini 3.1 Pro",
    "google/gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "Llama-3.3-70B-Instruct": "Llama 3.3-70B",
    "meta/llama-3.3-70b-instruct-maas": "Llama 3.3-70B",
    "qwen/qwen3-235b-a22b-instruct-2507-maas": "Qwen 3-235B",
    "xai/grok-4.1-fast-non-reasoning": "Grok 4.1 Fast",
    "zai-org/glm-4.7-maas": "GLM-4.7",
}

MODEL_ORDER = [
    "Llama 3.3-70B",
    "Qwen 3-235B",
    "DeepSeek V3.2",
    "GPT-OSS 120B",
    "GLM-4.7",
    "Grok 4.1 Fast",
    "Gemini 3.1 Pro",
]
