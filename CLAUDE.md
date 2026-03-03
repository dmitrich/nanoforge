# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Knowledge base: /Documents/know/llm-training/projects/alg3.md

## Project Overview

**alg3** is a modular PyTorch system for training and evaluating small GPT language models on the TinyStories dataset using Byte-Pair Encoding (BPE) tokenization. The project emphasizes reproducibility (config hashing, deterministic seeds, manifest tracking) and clean separation of concerns.

## Commands

### Environment
Uses `uv` for environment management with Python 3.12.12 and a `.venv` directory.

### One-time Setup
```bash
python setup_tokenizer.py          # Extract BPE vocab/merges from data/raw/
python src/dataset_prep.py configs/ds_tinystories_pretrain.json  # Prepare dataset shards
```

### Training
```bash
python src/train.py configs/train.json   # batch=32, cosine LR, 1000 steps, TensorBoard on
```

### Inference
```bash
python src/infer.py configs/infer.json   # Batch mode — 3 prompts, inherits latest run
python src/infer.py                      # Auto-detects latest training run
```

### Evaluation (deepeval + Nebius Llama judge)
```bash
python evals.py configs/eval.json   # Coherence, Fluency, Creativity via Nebius judge
python evals.py                     # latest training run + built-in defaults
```
Outputs land in `runs/evals/<eval_id>/`: `generations.jsonl`, `results.jsonl`, `summary.json`, `manifest.json`.

Nebius credentials are resolved automatically via `~/Documents/dev/azure/providers.py` (macOS Keychain service `nebius-api-key`, or `NEBIUS_API_KEY` env var).

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
The project is config-driven via JSON files in `configs/`. Key dataclasses:
- `RunConfig` — training configuration (model dims, optimizer, data paths, env metadata)
- `InferConfig` — supports `$from_run` inheritance to reference a training run's config

Configs are hashed for reproducibility and resolved at runtime (adds torch version, timestamp, platform).

### GPT Model (`src/model.py`)
Standard decoder-only transformer: token + position embeddings → N transformer blocks (pre-norm, MHA + FFN) → LM head. No bias terms. Default: 4 layers, 4 heads, 1024 embedding (~98M params).

### Training Pipeline (`src/train.py`)
- AdamW optimizer (β1=0.9, β2=0.95, weight_decay=0.1)
- LR schedule: linear warmup (0→200 steps) then cosine decay to 10% of max_lr
- Gradient clipping at 1.0
- Saves `best.pt` (best val loss) and `latest.pt` (periodic/final) under `runs/train/<run_id>/checkpoints/`

### Data Pipeline (`src/dataloader.py`, `src/dataset_prep.py`)
Training data stored as binary uint16 shards in `artifacts/datasets/`. `ShardedDataset` memory-maps shards and produces sliding window (input, target) pairs.

### Artifact & Manifest System
- **Artifacts** (`artifacts/`): Versioned tokenizers and datasets, each with a `*_manifest.json`
- **Run Registry** (`runs/registry.jsonl`): Global log of all runs
- **Per-run manifests** (`runs/train/<run_id>/manifest.json`): Status, config hash, lineage (tokenizer/dataset IDs, parent run), output paths, summary stats

### Experiment Tracking (`src/tracker.py`)
Dual tracking: TensorBoard (optional, graceful fallback) + JSONL metrics file. Both can be disabled via config flags.

### Inference Modes (`src/infer.py`)
Three modes controlled by `InferConfig.mode`: `batch` (encode prompts from config → save JSONL), `interactive` (REPL loop), `evals` (stub — actual deepeval integration lives in `evals.py`).

### Evaluation Pipeline (`evals.py`)
Standalone entry point (project root). Flow: load JSON eval config → resolve `$from_run` references inline (same pattern as `InferConfig`) → run GPT inference on test cases in `evals.json` → score outputs with deepeval `GEval` (LLM-as-judge) → write results. The canonical eval config is `configs/eval.json`; it carries the same `source`/`model`/`tokenizer`/`generation` sections as infer configs plus `judge`, `metrics`, and `test_cases_file` sections. The judge class wraps a Nebius API client; credentials are fetched through `~/Documents/dev/azure/providers.py`. `manifest.py`'s `rebuild_registry` does not index `runs/evals/`; eval runs are appended to `runs/registry.jsonl` at creation time.

## Key Design Patterns

- **Artifact-based**: Tokenizers and datasets are versioned independently of runs; referenced by ID in configs
- **Config inheritance**: `InferConfig` uses `$from_run` to inherit tokenizer/model settings from a training run
- **Device abstraction**: Auto-detects MPS → CUDA → CPU
- **Reproducibility**: Seeds, config hashing, manifest verification
