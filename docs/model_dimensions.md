# Model Dimensions Summary

## Overview

Before training starts, a summary table displays the key model dimensions (B, T, C, V) and their relationships. This helps you understand the model's memory and compute requirements.

## The Summary Table

```
======================================================================
Model Dimensions Summary
======================================================================
  B (Batch size):              32
  T (Block size):             128
  C (Embedding dim):        1,024
  V (Vocab size):           8,000
  ──────────────────────────────────────────────────────────────────
  B × T × C:             4,194,304  (activations per batch)
======================================================================
```

## Dimension Definitions

### B - Batch Size

**What it is**: Number of sequences processed in parallel per training step

**Configured in**: `training.batch_size`

**Example**: `32` means 32 sequences per batch

**Impact**:
- Larger B = Faster training (more parallelism)
- Larger B = More memory required
- Larger B = More stable gradients
- Typical values: 8-64 for small models, 1-8 for large models

**Memory scaling**: Linear (2× batch size = 2× memory)

### T - Block Size (Sequence Length)

**What it is**: Maximum number of tokens in each sequence

**Configured in**: `model.block_size` (must match `dataset.max_seq_len`)

**Example**: `128` means each sequence has 128 tokens

**Impact**:
- Larger T = More context for the model
- Larger T = Quadratic memory increase (attention is O(T²))
- Larger T = Slower training
- Typical values: 128-512 for small models, 1024-4096 for large models

**Memory scaling**: Quadratic (2× sequence length = 4× memory for attention)

### C - Embedding Dimension

**What it is**: Size of the internal representation vectors

**Configured in**: `model.n_embd`

**Example**: `1024` means each token is represented by a 1024-dimensional vector

**Impact**:
- Larger C = More model capacity
- Larger C = More parameters
- Larger C = More memory and compute
- Typical values: 384-768 for small models, 1024-4096 for medium, 8192+ for large

**Memory scaling**: Linear (2× embedding dim = 2× memory for activations)

**Relationship**: Must be divisible by `n_head` (number of attention heads)

### V - Vocabulary Size

**What it is**: Number of unique tokens the model can process

**Configured in**: `model.vocab_size` (must match `tokenizer.vocab_size`)

**Example**: `8000` means the model knows 8000 different tokens

**Impact**:
- Larger V = Can represent more words/subwords
- Larger V = Larger embedding and output layers
- Larger V = More memory for embeddings
- Typical values: 8k-32k for BPE, 50k-100k for word-level

**Memory scaling**: Linear (2× vocab size = 2× memory for embeddings)

## B × T × C - Activations per Batch

**What it is**: Total number of activation values in the main tensor

**Calculation**: `B × T × C`

**Example**: `32 × 128 × 1024 = 4,194,304` (4.2M values)

**Why it matters**:
- Represents the size of the main activation tensor
- Directly impacts memory usage
- Affects gradient computation memory
- Key metric for understanding compute requirements

**Memory usage** (approximate):
- Float32: `B × T × C × 4 bytes`
- BFloat16: `B × T × C × 2 bytes`
- Example: 4.2M × 2 bytes = 8.4 MB per layer

**Total activation memory** (rough estimate):
```
Total ≈ B × T × C × n_layers × 2 bytes (bfloat16)
Example: 4.2M × 4 layers × 2 bytes = 33.6 MB
```

## Understanding the Relationships

### Memory Hierarchy

From most to least memory impact:

1. **T (Block size)**: Quadratic impact due to attention
2. **B (Batch size)**: Linear impact, but multiplies everything
3. **C (Embedding dim)**: Linear impact on activations
4. **V (Vocab size)**: Only affects embedding layers

### Compute Hierarchy

From most to least compute impact:

1. **T (Block size)**: Quadratic for attention (O(T²))
2. **C (Embedding dim)**: Affects all matrix multiplications
3. **B (Batch size)**: More batches = more compute (but better parallelism)
4. **V (Vocab size)**: Only affects embedding lookups and final projection

### Typical Configurations

#### Small Model (Fast Training)
```
B = 64
T = 128
C = 384
V = 8000
B × T × C = 3,145,728 (3.1M)
```

#### Medium Model (Balanced)
```
B = 32
T = 256
C = 768
V = 32000
B × T × C = 6,291,456 (6.3M)
```

#### Large Model (High Quality)
```
B = 16
T = 512
C = 1024
V = 50000
B × T × C = 8,388,608 (8.4M)
```

## Practical Examples

### Example 1: Default Configuration

```json
{
  "model": {
    "n_embd": 1024,
    "block_size": 128,
    "vocab_size": 8000
  },
  "training": {
    "batch_size": 32
  }
}
```

**Output**:
```
B (Batch size):              32
T (Block size):             128
C (Embedding dim):        1,024
V (Vocab size):           8,000
B × T × C:             4,194,304  (activations per batch)
```

**Analysis**:
- Moderate batch size (32) for good parallelism
- Short sequences (128) for fast training
- Large embeddings (1024) for good capacity
- Small vocab (8k) for BPE tokenization
- 4.2M activations = manageable memory

### Example 2: Memory-Constrained Setup

```json
{
  "model": {
    "n_embd": 512,
    "block_size": 64,
    "vocab_size": 8000
  },
  "training": {
    "batch_size": 16
  }
}
```

**Output**:
```
B (Batch size):              16
T (Block size):              64
C (Embedding dim):          512
V (Vocab size):           8,000
B × T × C:               524,288  (activations per batch)
```

**Analysis**:
- Small batch (16) to fit in memory
- Very short sequences (64) to reduce memory
- Smaller embeddings (512) for less capacity
- 524K activations = 8× less memory than default

### Example 3: High-Quality Model

```json
{
  "model": {
    "n_embd": 2048,
    "block_size": 256,
    "vocab_size": 32000
  },
  "training": {
    "batch_size": 16
  }
}
```

**Output**:
```
B (Batch size):              16
T (Block size):             256
C (Embedding dim):        2,048
V (Vocab size):          32,000
B × T × C:             8,388,608  (activations per batch)
```

**Analysis**:
- Smaller batch (16) due to larger model
- Longer sequences (256) for more context
- Large embeddings (2048) for high capacity
- Large vocab (32k) for better tokenization
- 8.4M activations = 2× more memory than default

## Memory Estimation

### Activation Memory (per layer)

```
Memory = B × T × C × dtype_size
```

**Example** (bfloat16):
```
32 × 128 × 1024 × 2 bytes = 8,388,608 bytes = 8.4 MB
```

### Total Activation Memory (all layers)

```
Total = B × T × C × n_layers × dtype_size
```

**Example** (4 layers, bfloat16):
```
32 × 128 × 1024 × 4 × 2 bytes = 33,554,432 bytes = 33.6 MB
```

### Attention Memory (per layer)

```
Memory = B × n_heads × T × T × dtype_size
```

**Example** (4 heads, bfloat16):
```
32 × 4 × 128 × 128 × 2 bytes = 4,194,304 bytes = 4.2 MB
```

### Total Memory (rough estimate)

```
Total ≈ (Activations + Attention + Gradients + Optimizer) × safety_factor
Total ≈ (33.6 MB + 4.2 MB) × 3 × 1.5 ≈ 170 MB
```

**Note**: This is a simplified estimate. Actual memory usage includes:
- Model parameters
- Optimizer states (2× parameters for AdamW)
- Gradient buffers
- Temporary buffers
- Framework overhead

## Optimization Strategies

### To Reduce Memory

1. **Reduce B**: Smaller batch size
   - `batch_size: 16` instead of `32`
   - Impact: 2× less memory, slower training

2. **Reduce T**: Shorter sequences
   - `block_size: 64` instead of `128`
   - Impact: 4× less memory (quadratic), less context

3. **Reduce C**: Smaller embeddings
   - `n_embd: 512` instead of `1024`
   - Impact: 2× less memory, less capacity

4. **Use mixed precision**: Already enabled with `dtype: "bfloat16"`
   - Impact: 2× less memory vs float32

### To Increase Speed

1. **Increase B**: Larger batch size (if memory allows)
   - `batch_size: 64` instead of `32`
   - Impact: Faster training, more memory

2. **Reduce T**: Shorter sequences
   - `block_size: 64` instead of `128`
   - Impact: 4× faster attention, less context

3. **Reduce C**: Smaller embeddings
   - `n_embd: 768` instead of `1024`
   - Impact: Faster matrix ops, less capacity

### To Increase Quality

1. **Increase C**: Larger embeddings
   - `n_embd: 1536` instead of `1024`
   - Impact: More capacity, more memory

2. **Increase T**: Longer sequences
   - `block_size: 256` instead of `128`
   - Impact: More context, much more memory

3. **Increase V**: Larger vocabulary
   - `vocab_size: 16000` instead of `8000`
   - Impact: Better tokenization, more memory

## Troubleshooting

### Out of Memory Error

**Symptoms**: Training crashes with OOM error

**Solutions** (in order of preference):
1. Reduce `batch_size` (e.g., 32 → 16 → 8)
2. Reduce `block_size` (e.g., 128 → 64)
3. Reduce `n_embd` (e.g., 1024 → 768 → 512)
4. Reduce `n_layer` (e.g., 4 → 3 → 2)

**Check your dimensions**:
```
B × T × C should be < 10M for most GPUs
B × T × C should be < 5M for Apple Silicon (MPS)
```

### Training Too Slow

**Symptoms**: Very low tokens/second

**Solutions**:
1. Increase `batch_size` (if memory allows)
2. Reduce `block_size` (quadratic speedup)
3. Reduce `n_embd` (linear speedup)
4. Reduce `eval_interval` (less evaluation overhead)

**Check your dimensions**:
```
B × T × C > 1M for good GPU utilization
B × T × C > 500K for Apple Silicon (MPS)
```

### Model Quality Too Low

**Symptoms**: High validation loss, poor generation

**Solutions**:
1. Increase `n_embd` (more capacity)
2. Increase `n_layer` (deeper model)
3. Increase `block_size` (more context)
4. Train longer (`max_steps`)

## Summary

### Key Dimensions

- **B**: Batch size (parallelism)
- **T**: Block size (context length)
- **C**: Embedding dimension (capacity)
- **V**: Vocabulary size (tokenization)

### Key Metric

- **B × T × C**: Activations per batch (memory/compute indicator)

### Quick Reference

```
Small:   B=64,  T=128,  C=384,   V=8k    → 3.1M activations
Medium:  B=32,  T=256,  C=768,   V=32k   → 6.3M activations
Large:   B=16,  T=512,  C=1024,  V=50k   → 8.4M activations
```

### When You See This Table

The dimensions summary appears at the start of every training run, helping you:
- Verify your configuration
- Estimate memory requirements
- Understand compute complexity
- Debug OOM errors
- Compare different configurations
