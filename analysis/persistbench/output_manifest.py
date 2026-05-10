"""Current PersistBench checkpoint locations used by analysis scripts."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = REPO_ROOT / "outputs" / "persistbench"
SAMPLES = REPO_ROOT / "benchmark_samples" / "persistbench"
DEFAULT_BENCHMARK = SAMPLES / "baseline" / "full_benchmark.jsonl"

BASELINE_CHECKPOINTS = (
    OUTPUTS / "baseline" / "output_persistbench_baseline_deepseek_gpt_llama_qwen_grok_glm.json",
    OUTPUTS / "baseline" / "output_persistbench_baseline_llama_qwen.json",
    OUTPUTS / "baseline" / "output_persistbench_baseline_gemini.json",
)

PARTITIONED_CHECKPOINTS = (
    OUTPUTS / "partitioned" / "output_persistbench_partitioned_deepseek_gpt_grok_glm.json",
    OUTPUTS / "partitioned" / "output_persistbench_partitioned_llama_qwen.json",
    OUTPUTS / "partitioned" / "output_persistbench_partitioned_gemini.json",
)

PARTITIONED_COS_CHECKPOINTS = (
    OUTPUTS
    / "partitioned_cos_similarity"
    / "output_persistbench_partitioned_cos_deepseek_gpt_gemini_llama_qwen_grok_glm.json",
)

PARTITIONED_CUSTOM_CHECKPOINTS = (
    OUTPUTS
    / "partitioned_custom_categories"
    / "output_persistbench_partitioned_custom_deepseek_gpt_llama_qwen_grok_glm.json",
    OUTPUTS / "partitioned_custom_categories" / "output_persistbench_partitioned_custom_llama_qwen.json",
    OUTPUTS / "partitioned_custom_categories" / "output_persistbench_partitioned_custom_gemini.json",
)

TREE_CHECKPOINTS = (
    OUTPUTS / "tree" / "output_persistbench_tree_informed_deepseek_gpt_llama_qwen_grok_glm.json",
    OUTPUTS / "tree" / "output_persistbench_tree_informed_llama_qwen.json",
    OUTPUTS / "tree" / "output_persistbench_tree_informed_gemini.json",
)

MEMORY_STRUCTURE_CHECKPOINTS: dict[str, tuple[Path, ...]] = {
    "flat_memory_list": BASELINE_CHECKPOINTS,
    "partitions": PARTITIONED_CHECKPOINTS,
    "cosine_similarity_partitions": PARTITIONED_COS_CHECKPOINTS,
    "dynamic_partitions": PARTITIONED_CUSTOM_CHECKPOINTS,
    "2_level_tree": TREE_CHECKPOINTS,
}

MEMORY_STRUCTURES = {
    "flat_memory_list": {
        "label": "Flat List",
        "checkpoints": list(BASELINE_CHECKPOINTS),
    },
    "partitions": {
        "label": "Inference Fixed Partitions",
        "checkpoints": list(PARTITIONED_CHECKPOINTS),
    },
    "cosine_similarity_partitions": {
        "label": "Cosine Similarity Partitions",
        "checkpoints": list(PARTITIONED_COS_CHECKPOINTS),
    },
    "dynamic_partitions": {
        "label": "Inference Dynamic Partitions",
        "checkpoints": list(PARTITIONED_CUSTOM_CHECKPOINTS),
    },
    "2_level_tree": {
        "label": "2-Level Tree",
        "checkpoints": list(TREE_CHECKPOINTS),
    },
}

METHOD_CHECKPOINTS: dict[str, tuple[Path, ...]] = {
    "baseline": BASELINE_CHECKPOINTS,
    "defence: permissive": (
        OUTPUTS / "defence" / "output_persistbench_permissive_deepseek_gpt_llama_qwen_grok_glm.json",
        OUTPUTS / "defence" / "output_persistbench_permissive_llama_qwen.json",
        OUTPUTS / "defence" / "output_persistbench_permissive_gemini.json",
    ),
    "defence: restrictive": (
        OUTPUTS / "defence" / "output_persistbench_restrictive_deepseek_gpt_llama_qwen_grok_glm.json",
        OUTPUTS / "defence" / "output_persistbench_restrictive_llama_qwen.json",
        OUTPUTS / "defence" / "output_persistbench_restrictive_gemini.json",
    ),
    "defence: rubric_informed": (
        OUTPUTS / "defence" / "output_persistbench_rubric_informed_deepseek_gpt_llama_qwen_grok_glm.json",
        OUTPUTS / "defence" / "output_persistbench_rubric_informed_llama_qwen.json",
        OUTPUTS / "defence" / "output_persistbench_rubric_informed_gemini.json",
    ),
    "defence: gepa_optimized": (
        OUTPUTS / "defence" / "output_persistbench_gepa_deepseek_gpt_llama_qwen_grok_glm.json",
        OUTPUTS / "defence" / "output_persistbench_gepa_llama_qwen.json",
        OUTPUTS / "defence" / "output_persistbench_gepa_gemini.json",
    ),
    "partitioned": PARTITIONED_CHECKPOINTS,
    "partitioned + permissive": (
        OUTPUTS
        / "partitioned_defence"
        / "output_persistbench_partitioned_permissive_deepseek_gpt_llama_qwen_grok_glm.json",
        OUTPUTS / "partitioned_defence" / "output_persistbench_partitioned_permissive_llama_qwen.json",
        OUTPUTS / "partitioned_defence" / "output_persistbench_partitioned_permissive_gemini.json",
    ),
    "partitioned + restrictive": (
        OUTPUTS
        / "partitioned_defence"
        / "output_persistbench_partitioned_restrictive_deepseek_gpt_llama_qwen_grok_glm.json",
        OUTPUTS / "partitioned_defence" / "output_persistbench_partitioned_restrictive_llama_qwen.json",
        OUTPUTS / "partitioned_defence" / "output_persistbench_partitioned_restrictive_gemini.json",
    ),
    "partitioned + rubric_informed": (
        OUTPUTS
        / "partitioned_defence"
        / "output_persistbench_partitioned_rubric_informed_deepseek_gpt_llama_qwen_grok_glm.json",
        OUTPUTS / "partitioned_defence" / "output_persistbench_partitioned_rubric_informed_llama_qwen.json",
        OUTPUTS / "partitioned_defence" / "output_persistbench_partitioned_rubric_informed_gemini.json",
    ),
    "partitioned + gepa_optimized": (
        OUTPUTS
        / "partitioned_defence"
        / "output_persistbench_partitioned_gepa_deepseek_gpt_llama_qwen_grok_glm.json",
        OUTPUTS / "partitioned_defence" / "output_persistbench_partitioned_gepa_llama_qwen.json",
        OUTPUTS / "partitioned_defence" / "output_persistbench_partitioned_gepa_gemini.json",
    ),
    "partitioned (cosine similarity)": PARTITIONED_COS_CHECKPOINTS,
    "partitioned (custom cats)": PARTITIONED_CUSTOM_CHECKPOINTS,
    "partitioned (custom) + permissive": (
        OUTPUTS
        / "partitioned_custom_categories_defence"
        / "output_persistbench_partitioned_custom_permissive_deepseek_gpt_llama_qwen_grok_glm.json",
        OUTPUTS
        / "partitioned_custom_categories_defence"
        / "output_persistbench_partitioned_custom_permissive_gemini.json",
    ),
    "partitioned (custom) + restrictive": (
        OUTPUTS
        / "partitioned_custom_categories_defence"
        / "output_persistbench_partitioned_custom_restrictive_deepseek_gpt_llama_qwen_grok_glm.json",
        OUTPUTS
        / "partitioned_custom_categories_defence"
        / "output_persistbench_partitioned_custom_restrictive_gemini.json",
    ),
    "partitioned (custom) + rubric_informed": (
        OUTPUTS
        / "partitioned_custom_categories_defence"
        / "output_persistbench_partitioned_custom_rubric_informed_deepseek_gpt_llama_qwen_grok_glm.json",
        OUTPUTS
        / "partitioned_custom_categories_defence"
        / "output_persistbench_partitioned_custom_rubric_informed_cross_domain_llama_qwen.json",
        OUTPUTS
        / "partitioned_custom_categories_defence"
        / "output_persistbench_partitioned_custom_rubric_informed_beneficial_cross_domain_llama_qwen.json",
        OUTPUTS
        / "partitioned_custom_categories_defence"
        / "output_persistbench_partitioned_custom_rubric_informed_gemini.json",
    ),
    "partitioned (custom) + gepa_optimized": (
        OUTPUTS
        / "partitioned_custom_categories_defence"
        / "output_persistbench_partitioned_custom_gepa_deepseek_gpt_llama_qwen_grok_glm.json",
        OUTPUTS / "partitioned_custom_categories_defence" / "output_persistbench_partitioned_custom_gepa_llama_qwen.json",
        OUTPUTS / "partitioned_custom_categories_defence" / "output_persistbench_partitioned_custom_gepa_gemini.json",
    ),
    "rag tau=0.25": (
        OUTPUTS / "rag" / "output_persistbench_rag_tau0.25_deepseek_gpt_grok_glm.json",
        OUTPUTS / "rag" / "output_persistbench_rag_tau0.25_llama_qwen.json",
        OUTPUTS / "rag" / "output_persistbench_rag_tau0.25_gemini.json",
    ),
    "rag tau=0.50": (
        OUTPUTS / "rag" / "output_persistbench_rag_tau0.5_deepseek_gpt_grok_glm.json",
        OUTPUTS / "rag" / "output_persistbench_rag_tau0.5_llama_qwen.json",
        OUTPUTS / "rag" / "output_persistbench_rag_tau0.5_gemini.json",
    ),
    "rag tau=0.75": (
        OUTPUTS / "rag" / "output_persistbench_rag_tau0.75_deepseek_gpt_grok_glm.json",
        OUTPUTS / "rag" / "output_persistbench_rag_tau0.75_llama_qwen.json",
        OUTPUTS / "rag" / "output_persistbench_rag_tau0.75_gemini.json",
    ),
    "informed_tree": TREE_CHECKPOINTS,
    "tree + permissive": (
        OUTPUTS / "tree_defence" / "output_persistbench_tree_permissive_deepseek_gpt_llama_qwen_grok_glm.json",
    ),
    "tree + restrictive": (
        OUTPUTS / "tree_defence" / "output_persistbench_tree_restrictive_deepseek_gpt_llama_qwen_grok_glm.json",
    ),
    "tree + rubric_informed": (
        OUTPUTS / "tree_defence" / "output_persistbench_tree_rubric_informed_deepseek_gpt_llama_qwen_grok_glm.json",
    ),
    "tree + gepa_optimized": (
        OUTPUTS / "tree_defence" / "output_persistbench_tree_gepa_deepseek_gpt_llama_qwen_grok_glm.json",
    ),
}

CHECKPOINT_LABELS = {
    path.name: label
    for key, label in {
        "flat_memory_list": "Flat Memory List",
        "partitions": "Inference Fixed Partitions",
        "cosine_similarity_partitions": "Cosine Similarity Partitions",
        "dynamic_partitions": "Inference Dynamic Partitions",
        "2_level_tree": "2-Level Tree",
    }.items()
    for path in MEMORY_STRUCTURE_CHECKPOINTS[key]
}

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
