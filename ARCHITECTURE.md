# Project Architecture

## Overview

This is a modular PyTorch-based system for training small GPT language models on the TinyStories dataset using BPE tokenization. The project emphasizes reproducibility, experiment tracking, and clean separation of concerns across data preparation, training, and inference phases.

## Project Structure

```
nanoforge/
├── artifacts/              # Versioned tokenizers and datasets
│   ├── tokenizers/
│   │   └── tok_bpe_8k/    # BPE tokenizer with 8K vocab
│   └── datasets/
│       └── ds_tinystories_tok_bpe8k_T128/  # Tokenized shards
├── configs/               # JSON configuration files
│   ├── tok_bpe_8k.json
│   ├── ds_tinystories_pretrain.json
│   ├── run_exp001.json    # Training config
│   └── infer_exp001.json  # Inference config
├── data/
│   └── raw/              # Raw pre-tokenized data
├── docs/                 # Documentation
├── notebooks/            # Jupyter notebooks for exploration
├── runs/                 # Experiment outputs
│   ├── train/           # Training runs with checkpoints
│   ├── infer/           # Inference outputs
│   └── registry.jsonl   # Global run registry
└── src/                 # Source code
    ├── config.py        # Configuration management
    ├── model.py         # GPT model architecture
    ├── train.py         # Training pipeline
    ├── infer.py         # Inference pipeline
    ├── tokenizer.py     # Tokenizer wrapper
    ├── dataset_prep.py  # Dataset preparation
    ├── dataloader.py    # PyTorch data loading
    ├── tracker.py       # Experiment tracking
    ├── manifest.py      # Run registry and lineage
    └── utils.py         # Utility functions
```

## Tokenization

This project uses **Byte Pair Encoding (BPE)** via HuggingFace's `tokenizers` library.

**Tokenizer**: `ByteLevelBPETokenizer` from `tokenizers` package
- **Type**: Byte-level BPE (handles any Unicode without unknown tokens)
- **Vocabulary**: 8000 tokens
- **Source**: Pre-trained on TinyStories dataset
- **Files**: 
  - `vocab.json`: Token-to-ID mapping
  - `merges.txt`: BPE merge operations
- **Library**: HuggingFace `tokenizers` (not `transformers`)

**Why ByteLevelBPE?**
- Operates on bytes rather than Unicode characters
- No unknown tokens (can encode any text)
- Efficient for English text
- Compatible with GPT-style models

## Training Metrics

Each training run now tracks comprehensive performance metrics:

**Metrics tracked**:
1. **Total tokens trained**: batch_size × max_seq_len × max_steps
2. **Total training time**: Wall-clock seconds from start to finish
3. **Training speed**: tokens/second throughput
4. **Best validation loss**: Lowest val loss achieved
5. **Final training loss**: Loss at last step

**Model dimensions summary**: Before training starts, a table displays:
- B (Batch size)
- T (Block size / sequence length)
- C (Embedding dimension)
- V (Vocabulary size)
- B × T × C (activations per batch)

**Example output**:
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

For detailed explanation of dimensions and memory estimation, see: [`docs/model_dimensions.md`](docs/model_dimensions.md)

**Output locations**:
- **Console**: Printed at end of training with formatted summary
- **resolved_run.json**: In `_resolved` section with timestamp
- **manifest.json**: In `summary` section

**Example console output**:
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

**Accessing metrics**:
```bash
# View training speed
jq '._resolved.tokens_per_second' runs/train/run_*/resolved_run.json

# Compare runs
jq '._resolved | {tokens: .total_tokens_trained, speed: .tokens_per_second}' runs/train/run_*/resolved_run.json
```

For detailed documentation, benchmarks, and optimization tips, see: [`docs/training_metrics.md`](docs/training_metrics.md)

---

## Core Components

### 1. Configuration System (`config.py`)

**Purpose**: Centralized configuration management with validation and resolution.

**Key Classes**:
- `TokenizerConfig`: Tokenizer specifications (type, vocab size, special tokens)
- `DatasetConfig`: Dataset metadata (source, splits, sequence length)
- `RunConfig`: Complete training configuration (model, training, environment)
- `InferConfig`: Inference configuration with checkpoint references

**Features**:
- Dataclass-based configs loaded from JSON
- Config validation (vocab size matching, dimension checks)
- Runtime resolution with environment metadata (torch version, platform, timestamp)
- `$from_run` references to inherit settings from training runs
- Config hashing for reproducibility verification

**Example**:
```python
run_cfg = RunConfig.load('configs/run_exp001.json')
run_cfg.validate()
resolved = run_cfg.resolve(run_id)  # Adds runtime metadata
```

### 2. Model Architecture (`model.py`)

**Architecture**: Standard GPT (Generative Pre-trained Transformer)

**Components**:
- `GPT`: Main model class
  - Token embeddings (vocab_size → n_embd)
  - Position embeddings (block_size → n_embd)
  - Transformer blocks (n_layer × Block)
  - Layer normalization
  - Language modeling head (n_embd → vocab_size)

- `Block`: Transformer block
  - Multi-head self-attention with residual connection
  - Feed-forward network with residual connection
  - Pre-normalization (LayerNorm before attention/FFN)

- `MultiHeadAttention`: Parallel attention heads
  - n_head independent attention mechanisms
  - Projection and dropout

- `Head`: Single attention head
  - Causal masking (lower triangular)
  - Scaled dot-product attention
  - Key, query, value projections

- `FeedFoward`: Position-wise FFN
  - Linear → ReLU → Linear
  - 4× expansion factor (n_embd → 4×n_embd → n_embd)

**Default Configuration** (exp001):
- 4 layers, 4 heads, 1024 embedding dimension
- ~98M parameters
- Block size: 128 tokens
- Dropout: 0.02
- No bias terms

**Key Methods**:
- `forward()`: Compute logits and optional loss
- `generate()`: Autoregressive text generation with temperature/top-k sampling
- `save_checkpoint()` / `from_checkpoint()`: Checkpoint management

### 3. Training Pipeline (`train.py`)

**Purpose**: End-to-end training orchestration with experiment tracking.

**Training Loop**:
1. Load and validate configuration
2. Initialize model, optimizer, dataloaders
3. Training loop with:
   - Cosine learning rate schedule with warmup
   - Gradient clipping
   - Periodic evaluation on train/val splits
   - Checkpoint saving (best by val loss + latest)
   - TensorBoard logging

**Optimizer**: AdamW
- Learning rate: 5e-4 (default)
- Warmup: 200 steps
- Weight decay: 0.1
- Betas: (0.9, 0.95)

**Learning Rate Schedule** (Warmup + Cosine Decay):

The learning rate is **not constant** - it changes during training using a schedule with three phases:

1. **Warmup Phase** (steps 0 → warmup_steps):
   - Starts at 0 and linearly increases to max_lr
   - Formula: `lr = max_lr × (step / warmup_steps)`
   - Example: If max_lr=5e-4 and warmup_steps=200:
     - Step 0: lr = 0
     - Step 100: lr = 2.5e-4
     - Step 200: lr = 5e-4

2. **Cosine Decay Phase** (warmup_steps → max_steps):
   - Smoothly decreases from max_lr to min_lr (10% of max_lr)
   - Formula: `lr = min_lr + 0.5 × (1 + cos(π × progress)) × (max_lr - min_lr)`
   - Where: `progress = (step - warmup_steps) / (max_steps - warmup_steps)`
   - Creates a smooth cosine curve from max to min

3. **Final Phase** (step >= max_steps):
   - Stays at min_lr = 0.1 × max_lr

**Why Use a Learning Rate Schedule?**

- **Warmup prevents early instability**: Starting with a small LR prevents the model from making large, destructive updates when weights are randomly initialized
- **Cosine decay improves convergence**: Gradually reducing LR allows the model to fine-tune and settle into better minima
- **Better final performance**: Models trained with LR schedules typically achieve lower loss than constant LR

**Visual Example** (max_lr=5e-4, warmup=200, max_steps=1000):
```
LR
5e-4 |     ╱‾‾‾╲___
     |    ╱        ╲___
     |   ╱             ╲___
     |  ╱                  ╲___
5e-5 | ╱                       ‾‾‾
     |________________________________
     0   200   500   800   1000  step
     └warmup┘└─── cosine decay ───┘
```

**Checkpointing**:
- `best.pt`: Best validation loss checkpoint
- `latest.pt`: Most recent checkpoint
- Includes model state, optimizer state, step, val_loss

**Evaluation**:
- Periodic evaluation every N steps
- Computes average loss over eval_steps batches
- Tracks both train and val loss

**Usage**:
```bash
python src/train.py configs/run_exp001.json
```

### 4. Inference Pipeline (`infer.py`)

**Purpose**: Text generation from trained checkpoints.

**Features**:
- Load checkpoints from training runs
- Auto-detect latest training run if no config provided
- Multiple generation strategies (temperature, top-k sampling)
- Batch prompt processing
- Output to JSONL format

**Generation Parameters**:
- `max_new_tokens`: Maximum tokens to generate
- `temperature`: Sampling temperature (1.0 = no scaling)
- `top_k`: Top-k sampling (0 = disabled)
- `strategy`: Sampling strategy

**Modes**:
1. **Config-based**: Load explicit inference config
2. **Auto-detect**: Use latest training run with default settings

**Output**:
- `generations.jsonl`: Prompt + generated text pairs
- `summary.json`: Inference metadata
- `manifest.json`: Run tracking

**Usage**:
```bash
python src/infer.py configs/infer_exp001.json
# or
python src/infer.py  # auto-detect latest run
```

### 5. Data Pipeline

#### Dataset Preparation (`dataset_prep.py`)

**Purpose**: Convert raw data into tokenized train/val shards.

**Process**:
1. Load raw data (binary or text)
2. Tokenize if needed
3. Split into train/val by ratio
4. Save as binary shards (uint16 format)
5. Generate dataset manifest

**Output**:
- `shard_000.bin`: Training data
- `shard_001.bin`: Validation data
- `dataset_manifest.json`: Metadata (token counts, splits, etc.)

#### DataLoader (`dataloader.py`)

**Purpose**: Efficient PyTorch data loading with sliding windows.

**ShardedDataset**:
- Memory-maps binary shard files
- Returns (input, target) pairs with sliding window
- Input: tokens[i:i+block_size]
- Target: tokens[i+1:i+block_size+1]

**DataLoader Configuration**:
- Shuffling for training, sequential for validation
- Configurable batch size and num_workers
- Drop last batch for consistent shapes

### 6. Tokenizer (`tokenizer.py`)

**Purpose**: Wrapper around HuggingFace's `ByteLevelBPETokenizer` from the `tokenizers` library.

**Implementation**: Uses HuggingFace `tokenizers.ByteLevelBPETokenizer`
```python
from tokenizers import ByteLevelBPETokenizer
tok = ByteLevelBPETokenizer(
    vocab=str(vocab_path),
    merges=str(merges_path),
)
```

**Features**:
- **BPE (Byte Pair Encoding)** with 8K vocabulary
- **Byte-level encoding**: Handles any Unicode text without unknown tokens
- Special tokens: `<pad>`, `<unk>`, `<s>`, `</s>`
- Loads from pre-trained vocab.json + merges.txt
- Batch encoding support

**Methods**:
- `encode(text)`: Text → token IDs (uses `_tok.encode(text).ids`)
- `decode(ids)`: Token IDs → text (uses `_tok.decode(ids)`)
- `encode_batch(texts)`: Batch encoding

**Artifacts** (in `artifacts/tokenizers/tok_bpe_8k/`):
- `vocab.json`: Token → ID mapping (8000 tokens)
- `merges.txt`: BPE merge rules extracted from source model
- `tokenizer_manifest.json`: Metadata (vocab size, paths, load timestamp)

**Setup**: The tokenizer is pre-built from source files using `setup_tokenizer.py`:
```bash
python setup_tokenizer.py
```
This extracts vocab and merges from `data/raw/98m_8k_bpe.{model,vocab}` into the artifacts directory.

### 7. Experiment Tracking

#### Tracker (`tracker.py`)

**Purpose**: Unified logging to TensorBoard and JSONL.

**Features**:
- TensorBoard integration for visualization
- JSONL metrics file for programmatic access
- Config logging
- Graceful fallback if TensorBoard unavailable

**Logged Metrics**:
- `Loss/train_step`: Per-step training loss
- `Loss/train_eval`: Periodic training evaluation loss
- `Loss/val`: Validation loss
- `LR`: Learning rate

#### Manifest System (`manifest.py`)

**Purpose**: Run registry, lineage tracking, and reproducibility.

**Features**:
- Global registry (`runs/registry.jsonl`) of all runs
- Per-run manifests with:
  - Run ID, type (train/infer), status
  - Config hash for verification
  - Lineage (parent runs, dataset/tokenizer IDs)
  - Output paths (checkpoints, metrics, etc.)
  - Summary statistics
- Status tracking: running → completed/failed
- Config hash verification

**Key Functions**:
- `generate_run_id()`: Timestamped unique IDs
- `create_manifest()`: Initialize run tracking
- `complete_manifest()`: Mark run as completed
- `fail_manifest()`: Mark run as failed
- `rebuild_registry()`: Reconstruct registry from disk
- `verify_run()`: Check config hash and output files

**Lineage Tracking**:
- Tokenizer ID (`tok_id`)
- Dataset ID (`ds_id`)
- Parent run ID (for fine-tuning)

### 8. Utilities (`utils.py`)

**Purpose**: Common helper functions.

**Functions**:
- `write_json()` / `append_jsonl()` / `read_jsonl()`: JSON I/O
- `hash_dict()` / `hash_file()`: Content hashing for reproducibility
- `set_seed()`: Reproducible random seeds (Python, NumPy, PyTorch)
- `get_device()`: Device selection (MPS/CUDA/CPU with fallback)

## Data Flow

### Training Flow

```
1. Configuration
   configs/run_exp001.json
   ↓
2. Load & Validate
   RunConfig.load() → validate()
   ↓
3. Initialize
   - Generate run_id
   - Create run directory
   - Resolve config with metadata
   - Create manifest
   ↓
4. Build Components
   - Load tokenizer from artifacts
   - Build dataloaders from dataset shards
   - Initialize GPT model
   - Create optimizer
   - Initialize tracker
   ↓
5. Training Loop
   - Iterate over batches
   - Compute loss, backprop, optimize
   - Log metrics
   - Periodic evaluation
   - Save checkpoints
   ↓
6. Completion
   - Save final checkpoint
   - Complete manifest
   - Close tracker
```

### Inference Flow

```
1. Configuration
   configs/infer_exp001.json (or auto-detect)
   ↓
2. Resolve References
   - Find training run
   - Load resolved_run.json
   - Inject model/tokenizer config
   ↓
3. Load Components
   - Load checkpoint (best/latest)
   - Load tokenizer
   - Move to device
   ↓
4. Generate
   - Encode prompts
   - Autoregressive generation
   - Decode outputs
   ↓
5. Save Results
   - Write generations.jsonl
   - Create manifest
   - Save summary
```

### Dataset Preparation Flow

```
1. Configuration
   configs/ds_tinystories_pretrain.json
   ↓
2. Load Raw Data
   - Binary file (uint16) or text
   - Tokenize if text
   ↓
3. Split
   - Train: 90%
   - Val: 10%
   ↓
4. Save Shards
   - shard_000.bin (train)
   - shard_001.bin (val)
   - dataset_manifest.json
```

## Key Design Patterns

### 1. Artifact-Based Architecture

All reusable components (tokenizers, datasets) are versioned artifacts with unique IDs:
- Stored in `artifacts/` directory
- Referenced by ID in configs
- Include manifests with metadata
- Enable reproducibility and sharing

### 2. Config Inheritance

Inference configs can inherit from training runs:
```json
{
  "model": { "$from_run": true },
  "tokenizer": { "$from_run": true }
}
```

This ensures inference uses exact same settings as training.

### 3. Reproducibility

Multiple mechanisms ensure reproducibility:
- Config hashing (SHA256 of resolved config)
- Seed setting (Python, NumPy, PyTorch)
- Environment metadata (torch version, platform, timestamp)
- Manifest verification

### 4. Separation of Concerns

Clear boundaries between:
- Configuration (JSON files)
- Data preparation (offline)
- Training (stateful)
- Inference (stateless)
- Tracking (observability)

### 5. Minimal Dependencies

Core dependencies only:
- **PyTorch**: Model architecture, training, optimization
- **tokenizers** (HuggingFace): BPE tokenization via `ByteLevelBPETokenizer`
- **numpy**: Data handling and array operations
- **tensorboard** (optional): Visualization and experiment tracking

**Note**: Uses `tokenizers` library (fast Rust-based), not the heavier `transformers` library.

## Configuration Schema

### Training Config (`run_exp001.json`)

```json
{
  "meta": {
    "run_name": "run_exp001",
    "description": "...",
    "author": "...",
    "tags": ["baseline", "pretrain"]
  },
  "environment": {
    "device": "mps|cuda|cpu",
    "dtype": "bfloat16|float32",
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
    "artifacts_path": "artifacts/datasets/...",
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
    "max_steps": 1000,
    "learning_rate": 0.0005,
    "scheduler": "cosine",
    "warmup_steps": 200,
    "grad_clip": 1.0,
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "eval_interval": 100,
    "eval_steps": 10,
    "checkpoint_interval": 500,
    "checkpoint_mode": "best_and_latest"
  },
  "observe": {
    "log_interval": 10
  }
}
```

### Inference Config (`infer_exp001.json`)

```json
{
  "meta": {
    "infer_name": "infer_exp001",
    "description": "...",
    "tags": ["exp001"]
  },
  "source": {
    "run_id": null,  // null = auto-detect latest
    "checkpoint": "best|latest"
  },
  "model": { "$from_run": true },
  "tokenizer": { "$from_run": true },
  "generation": {
    "max_new_tokens": 200,
    "temperature": 1.0,
    "top_k": 0,
    "strategy": "top_k"
  },
  "input": {
    "mode": "inline",
    "prompts": ["\n"]
  }
}
```

## Run Directory Structure

### Training Run

```
runs/train/run_exp001_20260301_101611/
├── manifest.json           # Run metadata and status
├── resolved_run.json       # Full resolved config
├── metrics.jsonl          # Per-step metrics
├── checkpoints/
│   ├── best.pt           # Best validation loss
│   └── latest.pt         # Most recent
└── tb/                   # TensorBoard logs
    └── events.out.tfevents.*
```

### Inference Run

```
runs/infer/infer_exp001_20260301_104305/
├── manifest.json          # Run metadata
├── resolved_infer.json    # Full resolved config
├── generations.jsonl      # Generated texts
└── summary.json          # Inference summary
```

## Device Support

The project supports multiple compute devices:
- **MPS** (Apple Silicon): Metal Performance Shaders
- **CUDA** (NVIDIA GPUs): CUDA acceleration
- **CPU**: Fallback for compatibility

Device selection with automatic fallback:
```python
device = get_device('mps')  # Falls back to CPU if MPS unavailable
```

## Experiment Workflow

### 1. Prepare Tokenizer
```bash
# Extract BPE tokenizer from source files (one-time setup)
python setup_tokenizer.py

# This creates:
# - artifacts/tokenizers/tok_bpe_8k/vocab.json (8000 tokens)
# - artifacts/tokenizers/tok_bpe_8k/merges.txt (BPE merge rules)
# - artifacts/tokenizers/tok_bpe_8k/tokenizer_manifest.json
```

### 2. Prepare Dataset
```bash
python src/dataset_prep.py configs/ds_tinystories_pretrain.json
```

### 3. Train Model
```bash
python src/train.py configs/run_exp001.json
```

### 4. Run Inference
```bash
python src/infer.py configs/infer_exp001.json
```

### 5. View Results
```bash
# TensorBoard
tensorboard --logdir runs/train/run_exp001_*/tb

# Metrics
cat runs/train/run_exp001_*/metrics.jsonl

# Generations
cat runs/infer/infer_exp001_*/generations.jsonl
```

### 6. Manage Runs
```bash
# List all runs
python src/manifest.py list

# Rebuild registry
python src/manifest.py rebuild

# Verify run integrity
python src/manifest.py verify runs/train/run_exp001_*
```

## Extension Points

### Adding New Model Architectures

1. Create new model class in `model.py`
2. Update `ModelConfig` with architecture-specific fields
3. Update training/inference to handle new architecture

### Adding New Tokenizers

The project uses HuggingFace's `tokenizers` library for BPE encoding. To add a new tokenizer:

1. Prepare vocab and merges files (or train a new BPE tokenizer)
2. Create setup script similar to `setup_tokenizer.py` to extract artifacts
3. Update `tokenizer.py` to load the new tokenizer type
4. Add tokenizer config to `configs/`
5. Reference the new `tok_id` in dataset and training configs

**Current Implementation**: `ByteLevelBPETokenizer` from HuggingFace
- Library: `tokenizers` (HuggingFace)
- Type: Byte-level BPE
- Vocab size: 8000 tokens

### Adding New Datasets

1. Create dataset config in `configs/`
2. Run `dataset_prep.py` to generate shards
3. Reference dataset ID in training config

### Custom Sampling Strategies

Extend `GPT.generate()` method with new sampling logic:
- Nucleus (top-p) sampling
- Beam search
- Constrained generation

## Performance Considerations

### Memory Optimization
- Gradient accumulation for larger effective batch sizes
- Mixed precision training (bfloat16)
- Gradient checkpointing for deeper models

### Speed Optimization
- DataLoader num_workers for parallel data loading
- Compiled models (torch.compile)
- Efficient attention implementations (Flash Attention)

### Disk I/O
- Binary format (uint16) for fast loading
- Memory-mapped datasets for large data
- Sharded datasets for distributed training

## Monitoring and Debugging

### TensorBoard
```bash
tensorboard --logdir runs/train
```
View:
- Loss curves (train/val)
- Learning rate schedule
- Config as text

### Metrics JSONL
Programmatic access to all logged metrics:
```python
import json
with open('runs/train/run_*/metrics.jsonl') as f:
    metrics = [json.loads(line) for line in f]
```

### Manifest Verification
Check run integrity:
```bash
python src/manifest.py verify runs/train/run_exp001_*
```

## Best Practices

1. **Always validate configs** before training
2. **Use meaningful run names** for organization
3. **Tag experiments** for easy filtering
4. **Monitor validation loss** to detect overfitting
5. **Save checkpoints frequently** for long runs
6. **Verify manifests** after runs complete
7. **Use consistent seeds** for reproducibility
8. **Document experiments** in config descriptions

## Future Enhancements

Potential areas for extension:
- Distributed training (DDP, FSDP)
- Fine-tuning workflows
- Model quantization
- ONNX export
- Evaluation metrics (perplexity, BLEU, etc.)
- Hyperparameter search
- Curriculum learning
- Multi-dataset training

---

## Frequently Asked Questions

### Why is the learning rate changing during training?

Even though you configure `learning_rate: 0.0005` (5e-4), the actual learning rate changes during training using a **Warmup + Cosine Decay** schedule:

1. **Warmup Phase** (steps 0→200): LR increases linearly from 0 to 5e-4
2. **Cosine Decay** (steps 200→1000): LR decreases smoothly from 5e-4 to 5e-5
3. **Minimum** (step 1000+): LR stays at 5e-5

**Why?** This schedule improves training stability and final performance:
- Warmup prevents early instability when weights are random
- High LR in the middle enables fast learning
- Low LR at the end allows fine-tuning and better convergence

**Your configured learning_rate is the MAXIMUM LR**, reached after warmup.

For a detailed explanation with formulas, examples, and visualizations, see: [`docs/learning_rate_schedule.md`](docs/learning_rate_schedule.md)

### How do I use a fixed/constant learning rate instead?

The current implementation uses a scheduled learning rate by default. To use a fixed learning rate:

**Option 1: Config-based (Recommended)**
Add this flag to your training config:
```json
{
  "training": {
    "learning_rate": 0.0003,
    "use_lr_schedule": false
  }
}
```

Then modify `src/train.py` `get_lr()` function to check this flag:
```python
def get_lr(step: int, training: dict) -> float:
    if not training.get('use_lr_schedule', True):
        return training['learning_rate']  # Constant LR
    # ... rest of scheduled LR code
```

**Option 2: Quick modification**
Replace the `get_lr()` function in `src/train.py`:
```python
def get_lr(step: int, training: dict) -> float:
    return training['learning_rate']  # Always constant
```

**Note**: Fixed LR typically results in 5-10% worse final loss compared to scheduled LR. Use lower LR values (e.g., 3e-4 instead of 5e-4) with fixed LR to maintain stability.

For complete implementation details and other schedule types (warmup-only, linear decay, etc.), see: [`docs/learning_rate_schedule.md`](docs/learning_rate_schedule.md)

---

## Step-by-Step Usage Guide

This section provides detailed instructions for running training and inference from scratch.

### Prerequisites

1. **Install Dependencies**
```bash
pip install torch numpy tokenizers tensorboard
```

2. **Verify Project Structure**
```bash
# Ensure you're in the project root (nanoforge/)
ls -la
# Should see: src/, configs/, artifacts/, runs/, data/
```

3. **Setup Tokenizer** (if not already done)
```bash
python setup_tokenizer.py
# Creates artifacts/tokenizers/tok_bpe_8k/
```

### Step 1: Configure a Training Run

Training configurations are JSON files in the `configs/` directory. You can either use an existing config or create a new one.

#### Option A: Use Existing Config

The project includes `configs/run_exp001.json` as a baseline configuration.

#### Option B: Create a New Training Config

Create a new file `configs/my_experiment.json`:

```json
{
  "meta": {
    "run_name": "my_experiment",
    "description": "My custom GPT training run",
    "author": "Your Name",
    "tags": ["custom", "experiment"]
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
    "n_layer": 6,
    "n_head": 6,
    "n_embd": 768,
    "dropout": 0.1,
    "block_size": 128,
    "bias": false,
    "vocab_size": 8000
  },

  "training": {
    "batch_size": 32,
    "max_steps": 2000,
    "learning_rate": 0.0003,
    "scheduler": "cosine",
    "warmup_steps": 100,
    "grad_clip": 1.0,
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "eval_interval": 200,
    "eval_steps": 20,
    "checkpoint_interval": 500,
    "checkpoint_mode": "best_and_latest"
  },

  "observe": {
    "log_interval": 10
  }
}
```

#### Key Configuration Parameters

**Environment**:
- `device`: `"mps"` (Apple Silicon), `"cuda"` (NVIDIA GPU), or `"cpu"`
- `dtype`: `"bfloat16"` (faster, less memory) or `"float32"` (more precise)
- `seed`: Random seed for reproducibility
- `num_workers`: DataLoader workers (0 for single-threaded, 2-4 for multi-core)

**Model Architecture**:
- `n_layer`: Number of transformer layers (4-12 typical)
- `n_head`: Number of attention heads (must divide n_embd evenly)
- `n_embd`: Embedding dimension (384, 768, 1024 typical)
- `dropout`: Dropout rate (0.0-0.2)
- `block_size`: Context length (must match dataset max_seq_len)

**Training Hyperparameters**:
- `batch_size`: Batch size (16-64 typical, adjust based on memory)
- `max_steps`: Total training steps
- `learning_rate`: Peak learning rate (1e-4 to 1e-3 typical)
- `warmup_steps`: Linear warmup steps
- `eval_interval`: Steps between validation evaluations
- `checkpoint_interval`: Steps between checkpoint saves

**Important**: Ensure these match:
- `model.vocab_size` == `tokenizer.vocab_size` == 8000
- `model.block_size` == `dataset.max_seq_len` == 128

### Step 2: Run Training from Command Line

Once your config is ready, start training:

```bash
# From project root (nanoforge/)
python src/train.py configs/my_experiment.json
```

**Expected Output**:
```
Run ID:  my_experiment_20260301_143022
Run dir: runs/train/my_experiment_20260301_143022
Device:  mps  |  Params: 42.3M  |  Steps: 2000
step     0: loss 10.9876  lr 0.00e+00
step    10: loss 8.5432  lr 3.00e-05
step    20: loss 7.2341  lr 6.00e-05
...
         eval  train=5.4321  val=5.5678
         -> best.pt  val=5.5678
...
Done. Run: my_experiment_20260301_143022
```

**What Happens During Training**:
1. Creates timestamped run directory: `runs/train/my_experiment_YYYYMMDD_HHMMSS/`
2. Saves resolved config with metadata
3. Initializes model, optimizer, dataloaders
4. Trains for specified steps
5. Logs metrics every `log_interval` steps
6. Evaluates on train/val every `eval_interval` steps
7. Saves checkpoints:
   - `best.pt`: Best validation loss
   - `latest.pt`: Most recent checkpoint
8. Writes TensorBoard logs to `tb/` subdirectory
9. Completes manifest with summary

**Training Outputs**:
```
runs/train/my_experiment_20260301_143022/
├── manifest.json           # Run metadata and status
├── resolved_run.json       # Full config with environment info
├── metrics.jsonl          # All logged metrics
├── checkpoints/
│   ├── best.pt           # Best validation checkpoint
│   └── latest.pt         # Latest checkpoint
└── tb/                   # TensorBoard logs
    └── events.out.tfevents.*
```

**Monitoring Training**:
- Watch console output for loss and learning rate
- Use TensorBoard for visualizations (see below)
- Check `metrics.jsonl` for programmatic access

**Stopping Training**:
- Press `Ctrl+C` to stop gracefully
- Latest checkpoint will be saved
- Manifest will be marked as failed

### Step 3: Configure an Inference Run

After training completes, create an inference config to generate text.

#### Option A: Auto-Detect Latest Run

Run inference without a config (uses latest training run):
```bash
python src/infer.py
```

#### Option B: Create Explicit Inference Config

Create `configs/my_inference.json`:

```json
{
  "meta": {
    "infer_name": "my_inference",
    "description": "Generate stories from my trained model",
    "tags": ["generation", "my_experiment"]
  },

  "source": {
    "run_id": "my_experiment_20260301_143022",
    "checkpoint": "best"
  },

  "model": { "$from_run": true },
  "tokenizer": { "$from_run": true },

  "generation": {
    "max_new_tokens": 200,
    "temperature": 0.8,
    "top_k": 40,
    "strategy": "top_k"
  },

  "input": {
    "mode": "inline",
    "prompts": [
      "Once upon a time",
      "The little girl",
      "In a magical forest"
    ]
  }
}
```

#### Key Inference Parameters

**Source**:
- `run_id`: Training run ID (or `null` for auto-detect latest)
- `checkpoint`: `"best"` (best val loss) or `"latest"` (most recent)

**Model & Tokenizer**:
- `{ "$from_run": true }`: Inherit from training run (recommended)
- Or specify explicitly if you want to override

**Generation**:
- `max_new_tokens`: Maximum tokens to generate (50-500 typical)
- `temperature`: Sampling temperature
  - `1.0`: Standard sampling
  - `0.7-0.9`: More focused, coherent
  - `1.1-1.5`: More creative, diverse
- `top_k`: Top-k sampling
  - `0`: Disabled (sample from full distribution)
  - `40-50`: Balanced (recommended)
  - `10-20`: More conservative
- `strategy`: Sampling strategy (`"top_k"` currently supported)

**Input**:
- `mode`: `"inline"` (prompts in config) or `"file"` (load from file)
- `prompts`: List of text prompts to generate from

### Step 4: Run Inference from Command Line

Generate text using your trained model:

```bash
# Option 1: Use explicit config
python src/infer.py configs/my_inference.json

# Option 2: Auto-detect latest training run
python src/infer.py
```

**Expected Output**:
```
Infer ID: my_inference_20260301_150530
Checkpoint: runs/train/my_experiment_20260301_143022/checkpoints/best.pt
Generating 3 prompt(s), max_new_tokens=200

======================================================================
Once upon a time there was a little girl named Lily. She loved to 
play outside in the sunshine. One day, she saw a big red ball in 
the park. She ran to get it and kicked it very high. The ball went 
up and up into the sky...
----------------------------------------------------------------------
The little girl was very happy. She had a new toy to play with. 
It was a soft teddy bear with brown fur. She hugged it tight and 
smiled. Her mom said, "That's a nice bear!" The girl nodded...
----------------------------------------------------------------------
In a magical forest, there lived many animals. The birds sang 
beautiful songs in the trees. The rabbits hopped around looking 
for carrots. One day, a friendly fox came to visit...
----------------------------------------------------------------------

Inference complete. Run: my_inference_20260301_150530
```

**Inference Outputs**:
```
runs/infer/my_inference_20260301_150530/
├── manifest.json          # Run metadata
├── resolved_infer.json    # Full config
├── generations.jsonl      # Generated texts
└── summary.json          # Summary statistics
```

**View Generated Texts**:
```bash
# Pretty print generations
cat runs/infer/my_inference_20260301_150530/generations.jsonl | jq .

# Extract just the generated text
cat runs/infer/my_inference_20260301_150530/generations.jsonl | jq -r .generated
```

### Step 5: Run TensorBoard Locally

TensorBoard provides interactive visualizations of training metrics.

#### Start TensorBoard

```bash
# From project root (nanoforge/)
tensorboard --logdir runs/train
```

**Expected Output**:
```
TensorBoard 2.x.x at http://localhost:6006/ (Press CTRL+C to quit)
```

#### Access TensorBoard

1. Open your web browser
2. Navigate to: `http://localhost:6006`
3. You should see the TensorBoard interface

#### TensorBoard Features

**Scalars Tab**:
- `Loss/train_step`: Per-step training loss
- `Loss/train_eval`: Periodic training evaluation loss
- `Loss/val`: Validation loss
- `LR`: Learning rate schedule

**Text Tab**:
- View full resolved configuration

**Time Series Tab**:
- Compare multiple runs
- Smooth curves with slider
- Download data as CSV

#### View Specific Run

To view only a specific training run:

```bash
tensorboard --logdir runs/train/my_experiment_20260301_143022/tb
```

#### Compare Multiple Runs

TensorBoard automatically compares all runs in the directory:

```bash
# Compare all training runs
tensorboard --logdir runs/train

# Each run appears as a separate line in the plots
```

#### Custom Port

If port 6006 is already in use:

```bash
tensorboard --logdir runs/train --port 6007
```

Then access at `http://localhost:6007`

#### TensorBoard Tips

1. **Smoothing**: Use the smoothing slider (left sidebar) to reduce noise in loss curves
2. **Refresh**: Click the refresh button (top right) to load new data
3. **Download**: Click the download icon to export data as CSV/JSON
4. **Filtering**: Use regex in the filter box to show/hide specific metrics
5. **Horizontal Axis**: Switch between "Step", "Relative", and "Wall Time"

#### Stop TensorBoard

Press `Ctrl+C` in the terminal where TensorBoard is running.

### Complete Workflow Example

Here's a complete end-to-end example:

```bash
# 1. Setup (one-time)
python setup_tokenizer.py

# 2. Prepare dataset (if needed)
python src/dataset_prep.py configs/ds_tinystories_pretrain.json

# 3. Start TensorBoard in background
tensorboard --logdir runs/train &

# 4. Train model
python src/train.py configs/run_exp001.json

# 5. Run inference with auto-detect
python src/infer.py

# 6. View results
cat runs/infer/infer_*/generations.jsonl | jq -r .generated

# 7. List all runs
python src/manifest.py list

# 8. Verify run integrity
python src/manifest.py verify runs/train/run_exp001_*
```

### Troubleshooting

**Issue**: `RuntimeError: Working directory doesn't look like project root`
- **Solution**: Run commands from the `nanoforge/` directory, not from `src/`

**Issue**: `FileNotFoundError: artifacts/tokenizers/tok_bpe_8k/vocab.json`
- **Solution**: Run `python setup_tokenizer.py` first

**Issue**: `FileNotFoundError: artifacts/datasets/...`
- **Solution**: Run `python src/dataset_prep.py configs/ds_tinystories_pretrain.json`

**Issue**: `CUDA out of memory` or `MPS out of memory`
- **Solution**: Reduce `batch_size` in training config (try 16 or 8)

**Issue**: Training loss is NaN
- **Solution**: Reduce `learning_rate` (try 1e-4) or increase `warmup_steps`

**Issue**: TensorBoard shows no data
- **Solution**: Wait for first log interval, then refresh TensorBoard

**Issue**: Inference generates gibberish
- **Solution**: Train longer, or reduce `temperature` in inference config

### Advanced Usage

#### Resume Training from Checkpoint

Currently not directly supported, but you can:
1. Load checkpoint in `train.py`
2. Initialize optimizer state from checkpoint
3. Continue training loop

#### Fine-tuning

To fine-tune a pre-trained model:
1. Create new config with `parent_run_id` in lineage
2. Load checkpoint in training script
3. Use lower learning rate (1e-5 to 1e-4)

#### Batch Inference

To generate from many prompts:
1. Create text file with one prompt per line
2. Update inference config: `"mode": "file", "prompts_file": "prompts.txt"`
3. Modify `infer.py` to support file mode

#### Export Model

To export for deployment:
```python
# In Python
import torch
model, ckpt = GPT.from_checkpoint('runs/train/.../checkpoints/best.pt')
torch.save(model.state_dict(), 'model_weights.pt')
```

### Next Steps

After completing the basic workflow:
1. Experiment with different model architectures (layers, heads, dimensions)
2. Try different hyperparameters (learning rate, batch size, warmup)
3. Adjust generation parameters (temperature, top_k)
4. Monitor training curves in TensorBoard
5. Compare multiple runs to find best configuration
6. Use manifest system to track experiment lineage
