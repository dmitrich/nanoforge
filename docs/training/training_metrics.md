# Training Metrics Documentation

## Overview

Each training run now tracks and outputs comprehensive performance metrics including token throughput, training time, and model performance.

## Metrics Tracked

### 1. Total Tokens Trained

**What it measures**: The total number of tokens processed during training.

**Calculation**:
```python
tokens_per_batch = batch_size × max_seq_len
total_tokens_trained = tokens_per_batch × max_steps
```

**Example**:
- Batch size: 32
- Sequence length: 128
- Training steps: 1000
- Total tokens: 32 × 128 × 1000 = 4,096,000 tokens

**Why it matters**:
- Standard metric for comparing training runs
- Independent of batch size or sequence length
- Used in research papers (e.g., "trained on 100B tokens")
- Helps estimate compute requirements

### 2. Total Training Time (seconds)

**What it measures**: Wall-clock time from training start to completion.

**Includes**:
- Forward passes
- Backward passes
- Optimizer steps
- Evaluation runs
- Checkpointing
- Logging overhead

**Excludes**:
- Initial setup (model initialization, data loading setup)
- Final cleanup

**Why it matters**:
- Real-world training duration
- Cost estimation (cloud compute)
- Iteration speed for experiments

### 3. Training Speed (tokens/second)

**What it measures**: Throughput of the training process.

**Calculation**:
```python
tokens_per_second = total_tokens_trained / total_training_time_seconds
```

**Example**:
- Total tokens: 4,096,000
- Training time: 300 seconds
- Speed: 13,653 tokens/second

**Why it matters**:
- Hardware efficiency comparison
- Optimization effectiveness
- Scaling predictions
- Bottleneck identification

### 4. Best Validation Loss

**What it measures**: Lowest validation loss achieved during training.

**Why it matters**:
- Model quality indicator
- Checkpoint selection
- Overfitting detection

### 5. Final Training Loss

**What it measures**: Training loss at the last step.

**Why it matters**:
- Training convergence indicator
- Comparison with validation loss (overfitting check)

## Output Locations

### 1. Console Output (End of Training)

At the end of each training run, you'll see:

```
======================================================================
Training Complete: run_exp001_20260301_101611
======================================================================
Total tokens trained:  4,096,000
Total training time:   300.45 seconds (5.01 minutes)
Training speed:        13,634.21 tokens/second
Best validation loss:  4.3210
Final training loss:   4.2890
======================================================================
```

### 2. resolved_run.json (Permanent Record)

The `_resolved` section contains:

```json
{
  "_resolved": {
    "run_id": "run_exp001_20260301_101611",
    "timestamp": "2026-03-01T10:16:11.123456",
    "torch_version": "2.10.0",
    "python": "3.12.12",
    "platform": "macOS-26.3-arm64-arm-64bit",
    "total_tokens_trained": 4096000,
    "total_training_time_seconds": 300.45,
    "tokens_per_second": 13634.21,
    "training_completed_at": "2026-03-01 10:21:11"
  }
}
```

**Location**: `runs/train/<run_id>/resolved_run.json`

### 3. manifest.json (Summary)

The `summary` section contains:

```json
{
  "summary": {
    "best_val_loss": 4.3210,
    "final_loss": 4.2890,
    "steps": 1000,
    "total_tokens_trained": 4096000,
    "total_training_time_seconds": 300.45,
    "tokens_per_second": 13634.21
  }
}
```

**Location**: `runs/train/<run_id>/manifest.json`

## Accessing Metrics Programmatically

### Python

```python
import json

# Load resolved config
with open('runs/train/run_exp001_20260301_101611/resolved_run.json') as f:
    resolved = json.load(f)

# Access metrics
metrics = resolved['_resolved']
print(f"Tokens trained: {metrics['total_tokens_trained']:,}")
print(f"Training time: {metrics['total_training_time_seconds']:.2f}s")
print(f"Speed: {metrics['tokens_per_second']:,.2f} tokens/s")

# Or load from manifest
with open('runs/train/run_exp001_20260301_101611/manifest.json') as f:
    manifest = json.load(f)

summary = manifest['summary']
print(f"Best val loss: {summary['best_val_loss']:.4f}")
```

### Command Line (jq)

```bash
# Get training speed
jq '._resolved.tokens_per_second' runs/train/run_*/resolved_run.json

# Get all metrics
jq '._resolved | {tokens: .total_tokens_trained, time: .total_training_time_seconds, speed: .tokens_per_second}' runs/train/run_*/resolved_run.json

# Compare multiple runs
for run in runs/train/run_exp001_*/resolved_run.json; do
  echo "$(basename $(dirname $run)):"
  jq '._resolved | "  \(.total_tokens_trained) tokens in \(.total_training_time_seconds)s = \(.tokens_per_second) tok/s"' -r $run
done
```

### Bash Script

```bash
#!/bin/bash
# compare_runs.sh - Compare training speeds across runs

echo "Run ID                                    | Tokens      | Time (s) | Speed (tok/s)"
echo "------------------------------------------|-------------|----------|-------------"

for run_dir in runs/train/run_*; do
  if [ -f "$run_dir/resolved_run.json" ]; then
    run_id=$(basename "$run_dir")
    tokens=$(jq '._resolved.total_tokens_trained' "$run_dir/resolved_run.json")
    time=$(jq '._resolved.total_training_time_seconds' "$run_dir/resolved_run.json")
    speed=$(jq '._resolved.tokens_per_second' "$run_dir/resolved_run.json")
    printf "%-42s | %11s | %8s | %12s\n" "$run_id" "$tokens" "$time" "$speed"
  fi
done
```

## Interpreting Metrics

### Training Speed Benchmarks

Typical speeds on different hardware (4-layer, 1024-dim model, batch_size=32, seq_len=128):

| Hardware | Expected Speed | Notes |
|----------|---------------|-------|
| **Apple M1/M2 (MPS)** | 8,000-12,000 tok/s | Good for development |
| **Apple M3/M4 (MPS)** | 10,000-15,000 tok/s | Better memory bandwidth |
| **NVIDIA RTX 3090** | 15,000-25,000 tok/s | Excellent for training |
| **NVIDIA A100** | 30,000-50,000 tok/s | Professional GPU |
| **CPU (16 cores)** | 1,000-3,000 tok/s | Very slow, not recommended |

**Factors affecting speed**:
- Model size (larger = slower per token)
- Batch size (larger = faster, up to memory limit)
- Sequence length (longer = slower, quadratic attention)
- Mixed precision (bfloat16 = ~2x faster than float32)
- Evaluation frequency (more eval = slower overall)
- Checkpointing frequency (more checkpoints = slower)

### Token Count Interpretation

**Small experiments**: 1M-10M tokens
- Quick iteration
- Hyperparameter search
- Architecture testing

**Medium experiments**: 10M-100M tokens
- Baseline models
- Proof of concept
- Small datasets

**Large experiments**: 100M-1B tokens
- Production models
- Research papers
- Full dataset training

**Very large experiments**: 1B+ tokens
- State-of-the-art models
- Multi-GPU training
- Significant compute investment

### Training Time Interpretation

**Quick runs**: < 5 minutes
- Debugging
- Sanity checks
- Fast iteration

**Short runs**: 5-30 minutes
- Hyperparameter search
- Small experiments
- Development

**Medium runs**: 30 minutes - 2 hours
- Baseline models
- Typical experiments
- Most research

**Long runs**: 2-24 hours
- Production models
- Large datasets
- Final training

**Very long runs**: > 24 hours
- State-of-the-art models
- Requires checkpointing
- Multi-day training

## Using Metrics for Optimization

### 1. Identify Bottlenecks

Compare your speed to benchmarks:

```python
# Your speed
your_speed = 8000  # tokens/s

# Expected speed for your hardware
expected_speed = 12000  # tokens/s

efficiency = (your_speed / expected_speed) * 100
print(f"Running at {efficiency:.1f}% of expected speed")

if efficiency < 80:
    print("Potential bottlenecks:")
    print("- Too frequent evaluation")
    print("- Too frequent checkpointing")
    print("- Too frequent logging")
    print("- Slow disk I/O")
    print("- CPU bottleneck in data loading")
```

### 2. Estimate Training Time

```python
# Target tokens
target_tokens = 100_000_000  # 100M tokens

# Your measured speed
tokens_per_second = 10000

# Estimate time
estimated_seconds = target_tokens / tokens_per_second
estimated_hours = estimated_seconds / 3600

print(f"Estimated training time: {estimated_hours:.2f} hours")
```

### 3. Compare Configurations

```python
import json
from pathlib import Path

runs = []
for run_dir in Path('runs/train').iterdir():
    if run_dir.is_dir():
        resolved_path = run_dir / 'resolved_run.json'
        if resolved_path.exists():
            with open(resolved_path) as f:
                data = json.load(f)
                runs.append({
                    'run_id': data['_resolved']['run_id'],
                    'speed': data['_resolved']['tokens_per_second'],
                    'batch_size': data['training']['batch_size'],
                    'eval_interval': data['training']['eval_interval'],
                })

# Sort by speed
runs.sort(key=lambda x: x['speed'], reverse=True)

print("Fastest runs:")
for run in runs[:5]:
    print(f"{run['run_id']}: {run['speed']:,.0f} tok/s "
          f"(batch={run['batch_size']}, eval_interval={run['eval_interval']})")
```

### 4. Cost Estimation

```python
# Cloud GPU pricing (example: AWS p3.2xlarge with V100)
cost_per_hour = 3.06  # USD

# Your training metrics
training_time_seconds = 1800  # 30 minutes
training_time_hours = training_time_seconds / 3600

# Calculate cost
training_cost = training_time_hours * cost_per_hour
print(f"Training cost: ${training_cost:.2f}")

# Cost per million tokens
tokens_trained = 10_000_000
cost_per_million_tokens = (training_cost / tokens_trained) * 1_000_000
print(f"Cost per million tokens: ${cost_per_million_tokens:.4f}")
```

## Troubleshooting

### Speed is much slower than expected

**Check**:
1. Evaluation frequency: `eval_interval` too low?
2. Checkpointing frequency: `checkpoint_interval` too low?
3. Logging frequency: `log_interval` too low?
4. Batch size: Too small?
5. Data loading: `num_workers` set correctly?
6. Mixed precision: Using `bfloat16`?

**Solution**: See `docs/training_speed_optimization.md`

### Metrics not appearing in output

**Check**:
1. Using latest version of `src/train.py`?
2. Training completed successfully?
3. No errors during training?

**Solution**: Re-run training with updated code

### Token count seems wrong

**Verify calculation**:
```python
batch_size = 32
max_seq_len = 128
max_steps = 1000

expected_tokens = batch_size * max_seq_len * max_steps
print(f"Expected: {expected_tokens:,} tokens")
```

**Note**: Token count is based on training steps, not unique tokens in dataset

## Example Analysis

### Comparing Two Runs

```bash
# Run 1: Default config
python src/train.py configs/run_exp001.json
# Output: 4,096,000 tokens in 320.5s = 12,780 tok/s

# Run 2: Speed-optimized config
python src/train.py configs/run_fast.json
# Output: 4,096,000 tokens in 265.2s = 15,445 tok/s

# Speedup calculation
speedup = 15445 / 12780 = 1.21x (21% faster)
time_saved = 320.5 - 265.2 = 55.3 seconds
```

### Scaling Prediction

```python
# Measured on small run
small_run_tokens = 4_096_000
small_run_time = 300  # seconds
speed = small_run_tokens / small_run_time  # 13,653 tok/s

# Predict large run
large_run_tokens = 100_000_000
predicted_time = large_run_tokens / speed
predicted_hours = predicted_time / 3600

print(f"Predicted time for 100M tokens: {predicted_hours:.2f} hours")
```

## Summary

**Key Metrics**:
1. **Total tokens trained**: Standard unit for training scale
2. **Training time**: Real-world duration
3. **Tokens/second**: Hardware efficiency

**Where to find them**:
- Console output (end of training)
- `resolved_run.json` (`_resolved` section)
- `manifest.json` (`summary` section)

**Use cases**:
- Performance optimization
- Hardware comparison
- Cost estimation
- Experiment planning
- Bottleneck identification

**Next steps**:
- See `docs/training_speed_optimization.md` for optimization tips
- Use metrics to compare configurations
- Track improvements over time
