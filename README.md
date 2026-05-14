![Overview of memory-based personalization and cross-domain leakage. External tools provide user information that the assistant retrieves as raw memories and later uses to personalize responses. In a flat memory list, unrelated personal, health, or scheduling memories are exposed together, allowing irrelevant details to leak into a workplace email. With partitioned memories, the model groups memories by domain and uses only the relevant domains, preserving useful personalization while reducing leakage from unrelated domains.](analysis/persistbench/figures/Hook%20Figure.png)

# Mitigating Over-Personalization

This repository evaluates privacy and trustworthiness risks that can appear when conversational assistants use persistent user memories for personalization. It extends PersistBench-style evaluations with structured memory representations and prompt defenses, so the same underlying memories can be tested as a flat list, partitions, retrieved subsets, or a two-level tree.

The main research question is practical: can reorganizing memories at inference time preserve useful personalization while reducing leakage from irrelevant domains?

## What This Measures

PersistBench is used to evaluate the effect of a set of user memory re-structuring methods. The benchmark checks for the following failure modes:

- Cross-domain leakage: agent leaks irrelevant memories across domains, such as using health or personal details in an unrelated work task;
- Sycophancy: agenet becomes memory-induced sycophantic, over-agreeing because of remembered preferences or beliefs;
- Use useful memory: agenet still uses relevant memories when personalization is actually helpful.

Lower failure rates are better.

## Methods Included

The repo contains configs and data for several memory conditions:

- **Flat baseline**: all memories are inserted together.
- **Fixed/domain partitioning**: memories are grouped into predefined domains.
- **Dynamic/custom partitioning**: categories can be created per example.
- **Cosine-similarity partitioning**: fixed categories are assigned by embedding similarity.
- **RAG variants**: memories are filtered by similarity threshold before prompting.
- **Tree-structured memories**: memories are organized as category -> subcategory -> memory.
- **Prompt defenses**: permissive, restrictive, rubric-informed, and GEPA-optimized system prompts where relevant.

## Repository Map

```text
benchmark_samples/persistbench/   Benchmark JSONL inputs for baseline, partitioned, RAG, and tree runs
configs/persistbench/             Run configs for each method and defense combination
outputs/persistbench/             Checkpoints and completed generation/judgment outputs
prompts/                          Evaluation prompts, judge prompts, and defensive system prompts
src/benchmark/                    CLI, providers, runners, dataset loading, prompting, and checkpoint logic
src/benchmark/memory_normalization/persistbench/
                                  Builders for partitioned, RAG, and tree memory inputs
analysis/persistbench/            Failure-rate summaries, comparison scripts, and figure generation
analysis/persistbench/figures/    Plotting scripts and rendered figures
analysis/fine_tuning/             RAG threshold sweep tooling
```

Start with `configs/persistbench/` to see what experiments are defined, `benchmark_samples/persistbench/` to inspect the inputs, and `outputs/persistbench/` to inspect prior runs. The benchmark runner writes outputs incrementally, so output JSON files also act as checkpoints.

## Basic Usage

Install from the repo root:

```bash
uv sync
uv pip install -e .
```

Preview a config without making model calls:

```bash
uv run benchmark generate configs/persistbench/config_METHOD.json --dry-run
```

Run generation plus judging:

```bash
uv run benchmark run configs/persistbench/config_METHOD.json
```

Generate only, or judge an existing checkpoint:

```bash
uv run benchmark generate configs/persistbench/config_METHOD.json
uv run benchmark judge outputs/persistbench/METHOD/output_METHOD_MODELS.json
```

Useful flags include `--limit`, `--concurrency`, `--dry-run`, `--cancel`, `--no-auto-rerun`, `--store-raw-api-responses`, and `--ignore-config-mismatch`.

Summarize failure rates:

```bash
uv run failure-rates
uv run failure-rates files outputs/persistbench/METHOD/output_METHOD_MODELS.json
uv run failure-rates dir outputs/persistbench/defence
```

## Configs And Outputs

Each config names an input file, output checkpoint, judge provider/model, concurrency, and model list. Most baseline, defense, and RAG configs share one top-level input. Partitioned and tree configs usually point each model at a method-specific input file because the memory representation can depend on the model that built it.

Common config families:

- `config_baseline.json`: flat memory list.
- `config_defence_*.json`: prompt defenses on the flat input.
- `config_partitioned*.json`: fixed, cosine, and custom partition variants, with defense combinations.
- `config_rag_tau*.json`: RAG threshold filtering.
- `config_tree_informed*.json`: two-level tree memory inputs, with optional defenses.
- `fine_tuning/`: RAG threshold sweep configs.

Provider credentials are expected through the provider-specific environment variables used by the model entries. The included configs target Azure OpenAI-compatible deployments and Vertex AI Model Garden models, but the runner also has providers for OpenAI, Anthropic, Gemini, OpenRouter, and generic OpenAI-compatible endpoints.

## Memory Input Builders

Memory normalization scripts live under:

```text
src/benchmark/memory_normalization/persistbench/
```

The active builders are organized under `vertexai_requests/` and `azure_requests/`. For example, the RAG builder supports threshold and provider flags:

```bash
uv run python src/benchmark/memory_normalization/persistbench/PROVIDER/rag_persistbench_memories.py --threshold 0.25 0.5 0.75 --provider vertexai
```

Many partition and tree builders keep model lists and output paths as constants near the top of the script, so check the file before launching a large run.

## Citation

This repo builds on PersistBench. If you use the benchmark samples or evaluation setup, cite the upstream PersistBench work and this repository or associated paper as appropriate.
