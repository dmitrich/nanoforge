# Understanding manifest.json vs resolved_run.json

## Quick Answer

**manifest.json**: Lightweight metadata about the run (what, when, status, results)  
**resolved_run.json**: Complete configuration snapshot (how it was trained)

Think of it like:
- **manifest.json** = Table of contents + summary
- **resolved_run.json** = Full book with all details

## Detailed Comparison

### manifest.json

**Purpose**: Track run metadata, status, and results

**Size**: Small (~500 bytes)

**Contains**:
- Run identification (ID, name, type)
- Status tracking (running, completed, failed)
- Lineage (what artifacts were used)
- Output locations (where to find checkpoints, metrics)
- Summary results (final losses, training metrics)
- Timestamps (created, completed)

**Use cases**:
- Quick status check: "Did this run complete?"
- Find outputs: "Where are the checkpoints?"
- Compare results: "Which run had the best loss?"
- Track lineage: "What dataset/tokenizer was used?"
- Registry: All runs are indexed in `runs/registry.jsonl`

**Example structure**:
```json
{
  "run_id": "run_test_metrics_20260301_122732",
  "run_type": "train",
  "run_name": "run_test_metrics",
  "created_at": "2026-03-01T12:27:32.636982",
  "status": "completed",
  "config_hash": "e759e1301f9f",
  "lineage": {
    "tok_id": "tok_bpe_8k",
    "ds_id": "ds_tinystories_tok_bpe8k_T128",
    "parent_run_id": null
  },
  "outputs": {
    "best_checkpoint": "checkpoints/best.pt",
    "latest_checkpoint": "checkpoints/latest.pt",
    "resolved_config": "resolved_run.json",
    "metrics": "metrics.jsonl"
  },
  "summary": {
    "best_val_loss": 4.925,
    "final_loss": 4.53125,
    "steps": 50,
    "total_tokens_trained": 204800,
    "total_training_time_seconds": 20.36,
    "tokens_per_second": 10059.94
  },
  "completed_at": "2026-03-01T12:27:53.597799"
}
```

### resolved_run.json

**Purpose**: Complete, reproducible configuration snapshot

**Size**: Larger (~2-5 KB)

**Contains**:
- Full training configuration (all parameters)
- Model architecture (layers, dimensions, etc.)
- Dataset configuration (splits, sequence length)
- Tokenizer configuration (vocab size, type)
- Training hyperparameters (learning rate, batch size, etc.)
- Environment details (device, dtype, seed)
- Runtime metadata (torch version, platform, timestamps)
- Training metrics (tokens, time, speed)

**Use cases**:
- Reproduce run: "Train with exact same settings"
- Understand setup: "What hyperparameters were used?"
- Debug issues: "What was the learning rate?"
- Compare configs: "How do these two runs differ?"
- Resume training: "Load exact configuration"

**Example structure**:
```json
{
  "meta": {
    "run_name": "run_test_metrics",
    "description": "Test run to verify training metrics output",
    "author": "",
    "tags": ["test", "metrics"]
  },
  "environment": {
    "device": "mps",
    "dtype": "bfloat16",
    "seed": 42,
    "num_workers": 0
  },
  "tokenizer": {
    "tok_id": "tok_bpe_8k",
    "type": "bpe",
    "vocab_size": 8000,
    "artifacts_path": "artifacts/tokenizers/tok_bpe_8k"
  },
  "dataset": {
    "ds_id": "ds_tinystories_tok_bpe8k_T128",
    "dataset_type": "pretrain",
    "tok_id": "tok_bpe_8k",
    "artifacts_path": "artifacts/datasets/ds_tinystories_tok_bpe8k_T128",
    "max_seq_len": 128,
    "train_split": 0.9,
    "val_split": 0.1
  },
  "model": {
    "architecture": "gpt",
    "n_layer": 4,
    "n_head": 4,
    "n_embd": 1024,
    "dropout": 0.02,
    "block_size": 128,
    "bias": false,
    "vocab_size": 8000
  },
  "training": {
    "batch_size": 32,
    "max_steps": 50,
    "learning_rate": 0.0005,
    "scheduler": "cosine",
    "warmup_steps": 10,
    "grad_clip": 1.0,
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "eval_interval": 25,
    "eval_steps": 5,
    "checkpoint_interval": 1000,
    "checkpoint_mode": "best_and_latest"
  },
  "observe": {
    "log_interval": 10
  },
  "_resolved": {
    "run_id": "run_test_metrics_20260301_122732",
    "timestamp": "2026-03-01T12:27:32.623624",
    "torch_version": "2.10.0",
    "python": "3.12.12",
    "platform": "macOS-26.3-arm64-arm-64bit",
    "total_tokens_trained": 204800,
    "total_training_time_seconds": 20.36,
    "tokens_per_second": 10059.94,
    "training_completed_at": "2026-03-01 12:27:53"
  }
}
```

## Side-by-Side Comparison

| Aspect | manifest.json | resolved_run.json |
|--------|---------------|-------------------|
| **Purpose** | Metadata & results | Complete configuration |
| **Size** | Small (~500 bytes) | Larger (~2-5 KB) |
| **Created** | At run start | At run start |
| **Updated** | During & after run | At run start & end |
| **Contains config** | No (just hash) | Yes (full config) |
| **Contains results** | Yes (summary) | Yes (_resolved section) |
| **Contains status** | Yes | No |
| **Contains lineage** | Yes | No |
| **Indexed globally** | Yes (registry.jsonl) | No |
| **Used for inference** | No | Yes |
| **Human readable** | Very | Moderately |

## When to Use Each

### Use manifest.json when you want to:

✅ **Check run status**
```bash
jq '.status' runs/train/run_*/manifest.json
```

✅ **Find best run**
```bash
jq -r '"\(.run_id): \(.summary.best_val_loss)"' runs/train/run_*/manifest.json | sort -t: -k2 -n
```

✅ **List all runs**
```bash
jq -r '"\(.run_id) [\(.status)] - \(.summary.best_val_loss // "N/A")"' runs/train/run_*/manifest.json
```

✅ **Check what artifacts were used**
```bash
jq '.lineage' runs/train/run_*/manifest.json
```

✅ **Find output files**
```bash
jq '.outputs' runs/train/run_*/manifest.json
```

✅ **Get training metrics summary**
```bash
jq '.summary' runs/train/run_*/manifest.json
```

### Use resolved_run.json when you want to:

✅ **Reproduce a run**
```python
# Load exact configuration
with open('runs/train/run_exp001_*/resolved_run.json') as f:
    config = json.load(f)
# Use config to train again
```

✅ **Compare hyperparameters**
```bash
jq '.training.learning_rate' runs/train/run_*/resolved_run.json
```

✅ **Check model architecture**
```bash
jq '.model' runs/train/run_*/resolved_run.json
```

✅ **See environment details**
```bash
jq '._resolved | {torch: .torch_version, platform: .platform}' runs/train/run_*/resolved_run.json
```

✅ **Load config for inference**
```python
# Inference automatically loads from resolved_run.json
cfg = InferConfig.load('configs/infer_exp001.json')
# This reads model/tokenizer from resolved_run.json
```

✅ **Debug configuration issues**
```bash
# See all parameters that were actually used
jq '.' runs/train/run_*/resolved_run.json
```

## Lifecycle

### manifest.json Lifecycle

```
1. Run starts
   ↓
   create_manifest() creates manifest.json
   - status: "running"
   - created_at: timestamp
   - lineage: artifact IDs
   - outputs: file paths
   
2. During training
   ↓
   (manifest.json unchanged)
   
3. Run completes successfully
   ↓
   complete_manifest() updates manifest.json
   - status: "completed"
   - completed_at: timestamp
   - summary: results
   
4. Run fails
   ↓
   fail_manifest() updates manifest.json
   - status: "failed"
   - failed_at: timestamp
   - error: error message
```

### resolved_run.json Lifecycle

```
1. Run starts
   ↓
   config.resolve() creates resolved_run.json
   - Full config from JSON file
   - _resolved section with:
     - run_id
     - timestamp
     - torch_version
     - python version
     - platform
   
2. During training
   ↓
   (resolved_run.json unchanged)
   
3. Run completes
   ↓
   Updates _resolved section with:
   - total_tokens_trained
   - total_training_time_seconds
   - tokens_per_second
   - training_completed_at
```

## Relationship Between Files

```
Run Directory Structure:
runs/train/run_exp001_20260301_101611/
├── manifest.json              ← Metadata & results
├── resolved_run.json          ← Full configuration
├── metrics.jsonl              ← Detailed metrics (referenced by manifest)
├── checkpoints/
│   ├── best.pt               ← Referenced by manifest
│   └── latest.pt             ← Referenced by manifest
└── tb/                        ← TensorBoard logs
    └── events.out.tfevents.*

Global Registry:
runs/registry.jsonl            ← Index of all manifests
```

**Relationships**:
- `manifest.json` → points to `resolved_run.json` via `outputs.resolved_config`
- `manifest.json` → points to checkpoints via `outputs.best_checkpoint`, `outputs.latest_checkpoint`
- `manifest.json` → points to metrics via `outputs.metrics`
- `manifest.json` → indexed in `runs/registry.jsonl`
- `resolved_run.json` → used by inference to load model/tokenizer config
- `manifest.json.config_hash` → hash of `resolved_run.json` for verification

## Config Hash

The `config_hash` in manifest.json is a SHA256 hash of the resolved_run.json content.

**Purpose**: Verify configuration hasn't been tampered with

**Usage**:
```bash
# Verify a run's config integrity
python src/manifest.py verify runs/train/run_exp001_*
```

**Output**:
```
Run:    run_exp001_20260301_101611
Status: completed
Hash:   OK
  best_checkpoint: OK (checkpoints/best.pt)
  latest_checkpoint: OK (checkpoints/latest.pt)
  resolved_config: OK (resolved_run.json)
  metrics: OK (metrics.jsonl)
```

## Practical Examples

### Example 1: Find Fastest Training Run

```bash
# Use manifest.json for quick summary access
jq -r '"\(.summary.tokens_per_second // 0) \(.run_id)"' runs/train/run_*/manifest.json | sort -rn | head -1
```

### Example 2: Reproduce Best Run

```bash
# 1. Find best run using manifest.json
best_run=$(jq -r 'select(.status=="completed") | "\(.summary.best_val_loss) \(.run_id)"' runs/train/run_*/manifest.json | sort -n | head -1 | cut -d' ' -f2)

# 2. Copy resolved_run.json to create new config
cp "runs/train/$best_run/resolved_run.json" configs/reproduce_best.json

# 3. Remove _resolved section (not needed for input config)
jq 'del(._resolved)' configs/reproduce_best.json > configs/reproduce_best_clean.json

# 4. Train with same config
python src/train.py configs/reproduce_best_clean.json
```

### Example 3: Compare Two Runs

```python
import json

def compare_runs(run1_dir, run2_dir):
    # Load manifests for quick comparison
    with open(f'{run1_dir}/manifest.json') as f:
        m1 = json.load(f)
    with open(f'{run2_dir}/manifest.json') as f:
        m2 = json.load(f)
    
    print("Results Comparison:")
    print(f"Run 1: {m1['summary']['best_val_loss']:.4f} val loss")
    print(f"Run 2: {m2['summary']['best_val_loss']:.4f} val loss")
    print(f"Run 1: {m1['summary']['tokens_per_second']:.0f} tok/s")
    print(f"Run 2: {m2['summary']['tokens_per_second']:.0f} tok/s")
    
    # Load resolved configs for detailed comparison
    with open(f'{run1_dir}/resolved_run.json') as f:
        r1 = json.load(f)
    with open(f'{run2_dir}/resolved_run.json') as f:
        r2 = json.load(f)
    
    print("\nConfig Differences:")
    if r1['training']['learning_rate'] != r2['training']['learning_rate']:
        print(f"Learning rate: {r1['training']['learning_rate']} vs {r2['training']['learning_rate']}")
    if r1['training']['batch_size'] != r2['training']['batch_size']:
        print(f"Batch size: {r1['training']['batch_size']} vs {r2['training']['batch_size']}")
    # ... compare other parameters

compare_runs('runs/train/run_exp001_20260301_101611', 
             'runs/train/run_exp002_20260301_103022')
```

### Example 4: Check Run Status

```python
import json
from pathlib import Path

def check_run_status(run_dir):
    manifest_path = Path(run_dir) / 'manifest.json'
    
    if not manifest_path.exists():
        return "No manifest found"
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    status = manifest['status']
    
    if status == 'running':
        return f"Still running (started {manifest['created_at']})"
    elif status == 'completed':
        summary = manifest['summary']
        return f"Completed: {summary['best_val_loss']:.4f} val loss, {summary['tokens_per_second']:.0f} tok/s"
    elif status == 'failed':
        return f"Failed: {manifest.get('error', 'Unknown error')}"
    
    return f"Unknown status: {status}"

# Check all runs
for run_dir in Path('runs/train').iterdir():
    if run_dir.is_dir():
        print(f"{run_dir.name}: {check_run_status(run_dir)}")
```

## Summary

### manifest.json
- **What**: Lightweight metadata and results
- **Why**: Quick status checks, finding outputs, comparing results
- **When**: First place to look for run information
- **Size**: Small (~500 bytes)
- **Updates**: Created at start, updated at end

### resolved_run.json
- **What**: Complete configuration snapshot
- **Why**: Reproducibility, understanding setup, debugging
- **When**: Need to know exact training parameters
- **Size**: Larger (~2-5 KB)
- **Updates**: Created at start, metrics added at end

### Key Insight

**manifest.json** is optimized for **discovery and comparison**  
**resolved_run.json** is optimized for **reproducibility and understanding**

Both are essential:
- Use **manifest.json** to find what you're looking for
- Use **resolved_run.json** to understand how it was done

### Quick Reference

```bash
# Quick status check
jq '.status' runs/train/run_*/manifest.json

# Quick results
jq '.summary' runs/train/run_*/manifest.json

# Full configuration
jq '.' runs/train/run_*/resolved_run.json

# Compare configs
diff <(jq '.training' runs/train/run1/resolved_run.json) \
     <(jq '.training' runs/train/run2/resolved_run.json)
```
