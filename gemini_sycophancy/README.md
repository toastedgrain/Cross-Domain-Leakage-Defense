# Gemini sycophancy add-on

Adds the missing **sycophancy** failure type to the 10 Gemini-3.1-Pro PersistBench
runs that originally only covered cross_domain + beneficial. Each new run
*resumes the existing 300-entry checkpoint* and only generates+judges the 200
new sycophancy entries -- no duplicated work.

## What this folder contains

```
gemini_sycophancy/
├── data_prep/                       # one-time data generation scripts
│   ├── 1_partition_sycophancy.py    # partition 200 sycophancy memories with Gemini 3.1 Pro
│   ├── 2_tree_sycophancy.py         # tree-build 200 sycophancy memories with Gemini 3.1 Pro
│   └── 3_build_merged_data.py       # combine new 200 + existing 300 -> 500-entry merged files
├── configs/                         # 10 ready-to-run configs
│   ├── defence/                     # 4 (no data prep needed)
│   ├── partitioned/                 # 1
│   ├── partitioned_defence/         # 4
│   └── tree/                        # 1
├── data/                            # populated by data_prep scripts (gitignore-able)
├── run_all.sh                       # convenience runner for all 10
└── README.md                        # you are here
```

## How resuming works

PersistBench keys checkpoint entries by `hash_id`. For each of the 10 configs:

- `output` points to the existing 300-entry checkpoint (e.g.
  `outputs/persistbench/defence/persist_permissive_gemini3_pro.json`).
- `input` points to a 500-entry file (existing 300 + new 200 sycophancy).
- On resume, the 300 entries that match by hash are loaded and skipped.
- Only the 200 sycophancy entries get generated + judged.

Required flag: `--ignore-config-mismatch` (because `input` path changed since
the 300-entry run was first executed).

---

## Phase 0 -- environment setup

```bash
# Set service account path (required if not already in your shell profile)
export VERTEXAI_SERVICE_ACCOUNT_PATH="$(pwd)/service_account.json"

# Sanity-check the SA can mint a token and reach Gemini 3.1 Pro
cd "C:/Users/andzh/OneDrive/UCI/personal project/cross domain"
uv run python -c "import os, asyncio; os.environ['VERTEXAI_SERVICE_ACCOUNT_PATH']='service_account.json'; from google.oauth2.service_account import Credentials; from google.auth.transport.requests import Request; from openai import AsyncOpenAI; c=Credentials.from_service_account_file('service_account.json',scopes=['https://www.googleapis.com/auth/cloud-platform']); c.refresh(Request()); cli=AsyncOpenAI(base_url=f'https://aiplatform.googleapis.com/v1/projects/{c.project_id}/locations/global/endpoints/openapi',api_key=c.token); r=asyncio.run(cli.chat.completions.create(model='google/gemini-3.1-pro-preview',messages=[{'role':'user','content':'hi'}],max_tokens=20)); print('OK:', r.choices[0].message.content)"
```

If that prints `OK: <something>`, you're good. If it 403s on Llama or Kimi,
make sure those models are subscribed in Vertex AI Model Garden under your
project (separate per-model subscriptions).

---

## Phase 1 -- data prep (one-time, ~30 min, ~400 Gemini calls)

Skip these if you only want the 4 defensive runs (defensive uses
`benchmark_samples/persistbench/baseline/full_benchmark.jsonl` directly).

```bash
# 1a. Partition the 200 sycophancy entries via Gemini 3.1 Pro (~5-10 min)
uv run python gemini_sycophancy/data_prep/1_partition_sycophancy.py

# 1b. Tree-build the 200 sycophancy entries via Gemini 3.1 Pro (~10-20 min)
uv run python gemini_sycophancy/data_prep/2_tree_sycophancy.py

# 1c. Combine new 200-entry sycophancy data with existing 300-entry data files
uv run python gemini_sycophancy/data_prep/3_build_merged_data.py
```

After phase 1 you should have:
- `gemini_sycophancy/data/partitioned_full_500/full_benchmark.jsonl` (500 lines)
- `gemini_sycophancy/data/tree_full_500/full_benchmark_tree.jsonl` (500 lines)

Verify with:
```bash
wc -l gemini_sycophancy/data/partitioned_full_500/full_benchmark.jsonl
wc -l gemini_sycophancy/data/tree_full_500/full_benchmark_tree.jsonl
```

---

## Phase 2 -- run the 10 benchmarks

### Option A: run everything sequentially

```bash
bash gemini_sycophancy/run_all.sh
```

### Option B: run one at a time (recommended for first attempt)

Test on a single config first to confirm resume works:

```bash
uv run benchmark run gemini_sycophancy/configs/defence/permissive.json --ignore-config-mismatch --limit 5
```

You should see something like `Loaded 500 rows... Already-completed: 300...
Pending: 200`. If "Already-completed" is 0, the resume isn't working --
check that the existing 300-entry checkpoint exists at the expected path.

Then run them all:

```bash
# Defensive (4)
uv run benchmark run gemini_sycophancy/configs/defence/permissive.json --ignore-config-mismatch
uv run benchmark run gemini_sycophancy/configs/defence/restrictive.json --ignore-config-mismatch
uv run benchmark run gemini_sycophancy/configs/defence/rubric_informed.json --ignore-config-mismatch
uv run benchmark run gemini_sycophancy/configs/defence/gepa_optimized.json --ignore-config-mismatch

# Partitioned + 4 partition_defence (5)
uv run benchmark run gemini_sycophancy/configs/partitioned/partitioned.json --ignore-config-mismatch
uv run benchmark run gemini_sycophancy/configs/partitioned_defence/permissive.json --ignore-config-mismatch
uv run benchmark run gemini_sycophancy/configs/partitioned_defence/restrictive.json --ignore-config-mismatch
uv run benchmark run gemini_sycophancy/configs/partitioned_defence/rubric_informed.json --ignore-config-mismatch
uv run benchmark run gemini_sycophancy/configs/partitioned_defence/gepa_optimized.json --ignore-config-mismatch

# Tree (1)
uv run benchmark run gemini_sycophancy/configs/tree/informed_tree.json --ignore-config-mismatch
```

---

## Cost & time estimate

| Phase | API calls | Wall time |
|-------|-----------|-----------|
| 0 setup     | ~1 (test call)                | <1 min |
| 1 data prep | ~400 Gemini-3.1-Pro calls     | ~30 min |
| 2 evals     | ~6,000 gens + ~6,000 judgments | 2-5 hrs |

Compared to running fresh 500-entry benchmarks (~16,000 generations), this
plan saves ~62% on API cost.

---

## Verifying the result after each run

Each run's output file should grow to 500 entries with all three failure types:

```bash
uv run python -c "import json; d=json.load(open('outputs/persistbench/defence/persist_permissive_gemini3_pro.json',encoding='utf-8')); from collections import Counter; print(Counter(e['failure_type'] for e in d['entries'].values()))"
```

Expected: `Counter({'cross_domain': 200, 'sycophancy': 200, 'beneficial_memory_usage': 100})`

---

## Troubleshooting

- **`Invalid grant: account not found`**: SA key was rotated or deleted in
  GCP. Mint a new key in IAM and update `service_account.json`.
- **`Vertex AI API has not been used in project`**: enable the API at
  `https://console.developers.google.com/apis/api/aiplatform.googleapis.com/overview?project=<project-id>`.
- **`Publisher Model ... not found`**: subscribe the model in Vertex AI Model
  Garden (separate per-model + per-project acknowledgement).
- **`Invalid dataset 'PersistBench'`**: a regression in the validator. All
  configs in this folder include `"dataset": "persistbench"` to avoid it.
- **Generation 429s on Gemini 3.1 Pro**: lower `concurrency` to 10 or 5 in
  the affected config. Preview-tier quotas are tight.
- **Judge auth fails ~1 hr into a long run**: SA token expired and the
  refresh broke. Stop, re-export `VERTEXAI_SERVICE_ACCOUNT_PATH`, resume.
