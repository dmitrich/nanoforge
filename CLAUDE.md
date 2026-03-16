# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Knowledge base: /Documents/know/llm-training/projects/nanoforge.md

## Project Overview

**nanoforge** is a modular PyTorch system for training and evaluating small GPT language models on the TinyStories dataset using Byte-Pair Encoding (BPE) tokenization. The project emphasizes reproducibility (config hashing, deterministic seeds, manifest tracking) and clean separation of concerns.

## Commands

### Environment
Uses `uv` for environment management with Python 3.12.12 and a `.venv` directory.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

### One-time Setup

```bash
python src/dataprep.py config/dataprep.json
```

`dataprep.py` runs three phases in sequence (all idempotent — skipped if already done):
1. **Download** — streams from HuggingFace, probes first 100 examples for special tokens, writes `data/raw/MMDD_SEQ_SIZEtok.txt`
2. **Tokenize** — trains a BPE tokenizer, writes artifact to `artifacts/tokenizers/<tok_id>/`
3. **Encode** — tokenizes the `.txt` into binary shards (`shard_000.bin` train, `shard_001.bin` val) written to the dataset output dir configured in `config/dataprep.json`

### Training
```bash
python src/train.py config/train.json   # batch=32, cosine LR, 1000 steps, TensorBoard on
```

### Inference
```bash
python src/infer.py config/infer.json   # Batch mode — 3 prompts, inherits latest run
python src/infer.py                     # Auto-detects latest training run
```

### Evaluation (deepeval + LLM-as-judge)
```bash
python src/evals.py config/evals.json   # Coherence, Fluency, Creativity via configured judges
python src/evals.py                     # latest training run + built-in defaults
```
Outputs land in `runs/evals/<eval_id>/`: `generations.jsonl`, per-judge `<provider>/results.jsonl` + `summary.json`, top-level `summary.json`, `manifest.json`.

Providers (Nebius, Together AI, AWS Bedrock, Azure) are configured in `"providers"` section of `config/evals.json`. Credentials resolved from env var or macOS Keychain (service name in provider config).

### Eval Provider Credentials (macOS)

Credentials are resolved in order: **environment variable first**, then **macOS Keychain**.

**Option 1 — Environment variable** (good for CI or one-off runs):
```bash
export NEBIUS_API_KEY=your_nebius_key
export TOGETHER_API_KEY=your_together_key
export AZURE_OPENAI_API_KEY=your_azure_key   # only if using Azure judge
```

**Option 2 — macOS Keychain** (recommended for local development — stored encrypted, never in shell history):
```bash
# Store once; the app reads it silently on every eval run
security add-generic-password -s nebius-api-key    -a nanoforge -w YOUR_NEBIUS_KEY
security add-generic-password -s together-api-key  -a nanoforge -w YOUR_TOGETHER_KEY
security add-generic-password -s bedrock-api-key   -a nanoforge -w YOUR_BEDROCK_GATEWAY_KEY
security add-generic-password -s azure-openai-api-key -a nanoforge -w YOUR_AZURE_KEY
```

Note: `bedrock-api-key` is the API Gateway key (from `src/bedrock/bedrock.py`), not IAM credentials.

To verify a stored key:
```bash
security find-generic-password -s nebius-api-key -w
```

To update a stored key (delete then re-add):
```bash
security delete-generic-password -s nebius-api-key
security add-generic-password -s nebius-api-key -a nanoforge -w NEW_KEY
```

The service names (`nebius-api-key`, `together-api-key`, `bedrock-api-key`, `azure-openai-api-key`) are set in the `"keychain_service"` field of each provider in `config/evals.json`. The env var names are in the `"env_var"` field.

### Monitoring & Manifest
```bash
tensorboard --logdir runs/train
python src/manifest.py list
python src/manifest.py rebuild    # only re-indexes train/ and infer/, not evals/
python src/manifest.py verify runs/train/<run_dir>
```

There is no test suite, linter, or Makefile.

## Architecture

### Configuration System (`src/config.py`)
Config-driven via JSON files in `config/`. Key dataclasses:
- `RunConfig` — training config with named sections: `meta`, `environment`, `tokenizer`, `dataset`, `model`, `training`, `observability`. Validates cross-section constraints (`model.block_size == dataset.max_seq_len`, `model.vocab_size == tokenizer.vocab_size`, `dataset.tok_id == tokenizer.tok_id`) at startup.
- `InferConfig` — supports `$from_run` inheritance: setting `model` or `tokenizer` to `{"$from_run": true}` copies those sections from the source training run's `resolved_run.json`.
- `TokenizerConfig` / `DatasetConfig` — artifact configs loaded from their respective manifest JSONs.

Configs are resolved at runtime (adds torch version, timestamp, platform) and saved as `resolved_run.json` in the run directory.

### GPT Model (`src/model.py`)
Standard decoder-only transformer: token + position embeddings → N transformer blocks (pre-norm, MHA + FFN) → LM head. No bias terms. Default: 4 layers, 4 heads, 1024 embedding (~98M params).

### Training Pipeline (`src/train.py`)
- AdamW optimizer (β1=0.9, β2=0.95, weight_decay=0.1)
- LR schedule: linear warmup (0→200 steps) then cosine decay to 10% of max_lr
- Gradient clipping at 1.0
- Saves `best.safetensors` (best val loss) and `latest.safetensors` (periodic/final) under `runs/train/<run_id>/checkpoints/`

### Data Pipeline
- `src/dataprep.py` — 3-phase pipeline: download `.txt` from HuggingFace → train BPE tokenizer → encode to binary shards. All artifacts written to `artifacts/tokenizers/<tok_id>/` and the dataset dir configured in `config/dataprep.json`
- `src/tokenizer.py` — `Tokenizer` loading class (not a standalone script); used by training, inference, and evals to encode/decode text
- `src/dataloader.py` — `ShardedDataset` + `build_dataloaders()`; memory-maps binary uint16 shards, produces sliding window (input, target) pairs; called by `train.py` at runtime, not run standalone

### Artifact & Manifest System
- **Artifacts** (`artifacts/`): Versioned tokenizers and datasets, each with a `*_manifest.json`
- **Run Registry** (`runs/registry.jsonl`): Global log of all runs
- **Per-run manifests** (`runs/train/<run_id>/manifest.json`): Status, config hash, lineage (tokenizer/dataset IDs, parent run), output paths, summary stats

### Experiment Tracking (`src/tracker.py`)
Dual tracking: TensorBoard (optional, graceful fallback) + JSONL metrics file. Controlled via `observability.tensorboard` and `observability.jsonlog` boolean flags in `config/train.json`.

### Observability (`src/observability.py`)
Langfuse tracing with graceful no-op fallback. Activated when `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`, and `LANGFUSE_HOST` env vars are all set. All downstream code calls the same API regardless of whether Langfuse is active.

### Inference Modes (`src/infer.py`)
Three modes controlled by `input` section of `InferConfig`: `batch` (encode prompts from config → save JSONL), `interactive` (REPL loop), `evals` (read from evals file — actual deepeval integration in `src/evals.py`).

### Evaluation Pipeline (`src/evals.py` + `src/judge.py`)
Entry point uses `os.chdir` to project root at startup. Flow: load JSON eval config → resolve `$from_run` references → run GPT inference on inline `test_cases` → score with deepeval `GEval` (LLM-as-judge) for each judge in `config/evals.json`'s `"judges"` list → write per-judge results.

`src/judge.py` provides `JudgeLLM` (DeepEvalBaseLLM subclass) and `build_judge` factory. Supported providers: `nebius`, `together`, `aws` (Bedrock Access Gateway), `azure`. Credentials resolved from env var then macOS Keychain.

`manifest.py`'s `rebuild_registry` does not index `runs/evals/`; eval runs are appended to `runs/registry.jsonl` at creation time.

### Shared Utilities (`src/utils.py`)
`write_json`, `append_jsonl`, `read_jsonl`, `hash_dict`, `hash_file`, `set_seed`, `get_device`.

## Key Design Patterns

- **Artifact-based**: Tokenizers and datasets are versioned independently of runs; referenced by ID in configs
- **Config inheritance**: `InferConfig` uses `$from_run` to inherit tokenizer/model settings from a training run
- **Device abstraction**: `get_device()` auto-detects MPS → CUDA → CPU
- **Reproducibility**: Seeds, config hashing, manifest verification
