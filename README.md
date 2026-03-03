# alg3

Modular PyTorch system for training and evaluating small GPT language models on the TinyStories dataset using Byte-Pair Encoding (BPE) tokenization.

---

## Quick start

### 1. One-time setup

```bash
# Install dependencies
uv sync

# Build the BPE tokenizer from raw data
python setup_tokenizer.py

# Prepare dataset shards
python src/dataset_prep.py configs/ds_tinystories_pretrain.json
```

### 2. Train

```bash
python src/train.py configs/train.json
```

Checkpoints are written to `runs/train/<run_id>/checkpoints/`. Progress is visible in TensorBoard:

```bash
tensorboard --logdir runs/train
```

### 3. Generate stories

```bash
# Batch mode — 3 prompts, saves output JSONL
python src/infer.py configs/infer.json

# Interactive REPL — type prompts at the terminal
python src/infer.py configs/infer_interactive.json
```

Auto-detects the latest training run and loads the best checkpoint.

### 4. Evaluate quality

```bash
python evals.py configs/eval.json
```

Runs inference on test cases from `evals.json`, then scores each generation with an LLM judge (Coherence, Fluency, Creativity). Outputs land in `runs/evals/<eval_id>/`.

---

## Config file reference

All configs are JSON files in `configs/`. Each section below corresponds to a top-level key.

### `meta` — run identity

| Key | Type | Description |
|-----|------|-------------|
| `run_name` / `infer_name` / `eval_name` | string | Human-readable name, used as prefix in the run ID |
| `description` | string | Free-text note for the run registry |
| `tags` | string[] | Labels for filtering in the manifest |

---

### `environment` — hardware and reproducibility  *(train only)*

| Key | Values | Description |
|-----|--------|-------------|
| `device` | `"mps"`, `"cuda"`, `"cpu"` | Compute device. Auto-detected if omitted (MPS → CUDA → CPU) |
| `dtype` | `"bfloat16"`, `"float32"` | Floating-point precision. `bfloat16` is faster on MPS/CUDA |
| `seed` | integer | Random seed for reproducibility |
| `num_workers` | integer | DataLoader workers. Use `0` on MPS (spawning is slow) |

---

### `tokenizer` — vocabulary  *(train only)*

| Key | Description |
|-----|-------------|
| `tok_id` | Artifact ID, must match `dataset.tok_id` and the tokenizer directory name |
| `type` | Tokenizer type — only `"bpe"` is currently supported |
| `vocab_size` | Must equal `model.vocab_size` |
| `artifacts_path` | Path to the tokenizer artifact directory |

---

### `dataset` — training data  *(train only)*

| Key | Description |
|-----|-------------|
| `ds_id` | Artifact ID of the prepared dataset shards |
| `dataset_type` | `"pretrain"` — sliding-window next-token prediction |
| `tok_id` | Must match `tokenizer.tok_id` |
| `artifacts_path` | Path to the dataset artifact directory (binary uint16 shards) |
| `max_seq_len` | Sequence length in tokens. **Must equal `model.block_size`** |
| `train_split` / `val_split` | Fraction of shards used for train/validation (must sum to 1.0) |

---

### `model` — architecture  *(train only; infer/eval inherit via `$from_run`)*

| Key | Description |
|-----|-------------|
| `architecture` | `"gpt"` — standard decoder-only transformer |
| `n_layer` | Number of transformer blocks |
| `n_head` | Number of attention heads per block |
| `n_embd` | Embedding dimension. Must be divisible by `n_head` |
| `dropout` | Dropout probability (applied during training, disabled at inference) |
| `block_size` | Context length in tokens. **Must equal `dataset.max_seq_len`** |
| `bias` | Whether to add bias to linear layers. `false` is standard (GPT-2 style) |
| `vocab_size` | Must equal `tokenizer.vocab_size` |

**Parameter count** scales roughly as `12 × n_layer × n_embd²`. The default (4L, 4H, 1024D) is ~98M parameters.

---

### `training` — optimizer and schedule  *(train only)*

| Key | Description |
|-----|-------------|
| `batch_size` | Sequences per gradient step |
| `max_steps` | Total training steps |
| `learning_rate` | Peak LR (after warmup). Typical range: 3e-4 – 5e-4 |
| `scheduler` | `"cosine"` decays LR to 10% of peak; `"constant"` holds it fixed |
| `warmup_steps` | Steps to linearly ramp LR from 0 to `learning_rate`. Set to `0` for constant LR |
| `grad_clip` | Max gradient norm (1.0 is standard) |
| `weight_decay` | AdamW weight decay |
| `beta1` / `beta2` | AdamW momentum parameters |
| `eval_interval` | Run validation every N steps (and always at step 0 and the final step). Set to `999999` to skip mid-run evals |
| `eval_steps` | Number of validation batches to average |
| `checkpoint_interval` | Save `latest.pt` every N steps. Set to `999999` to only save at the end |
| `checkpoint_mode` | `"best_and_latest"` — saves `best.pt` whenever val loss improves plus `latest.pt` on schedule |

---

### `observe` — logging  *(train only)*

| Key | Default | Description |
|-----|---------|-------------|
| `log_interval` | `10` | Print loss to console every N steps |
| `disable_tensorboard` | `false` | Set to `true` to skip TensorBoard writer |
| `disable_jsonl` | `false` | Set to `true` to skip per-step JSONL metrics file |

Disabling both JSONL and TensorBoard gives the fastest iteration — console output only.

---

### `source` — checkpoint resolution  *(infer and eval)*

| Key | Description |
|-----|-------------|
| `run_id` | Training run ID to load from. `null` = auto-detect the most recent run |
| `checkpoint` | `"best"` loads `best.pt`; `"latest"` loads `latest.pt` |

---

### `model` / `tokenizer` — inheritance  *(infer and eval)*

Setting either to `{ "$from_run": true }` copies the architecture and tokenizer config directly from the source training run's manifest. This ensures inference always matches the training setup exactly.

---

### `generation` — sampling  *(infer and eval)*

| Key | Description |
|-----|-------------|
| `max_new_tokens` | Maximum tokens to generate beyond the prompt |
| `temperature` | Sampling temperature. Higher = more random. `1.0` = unmodified logits |
| `top_k` | Sample from the top-K most likely tokens. `0` = no filtering (full distribution) |
| `strategy` | `"top_k"` is the only implemented strategy |
| `stop_tokens` | List of strings that terminate generation early (e.g. `["</s>", "\n\n"]`) |

---

### `input` — inference mode  *(infer only)*

| Key | Description |
|-----|-------------|
| `interactive` | `true` = REPL loop, reads prompts from stdin |
| `evals` | `true` = read prompts from `evals_file` (deepeval test case format) |
| `prompts` | List of inline prompt strings (used when `interactive` and `evals` are both `false`) |
| `evals_file` | Path to the test cases JSON file (used when `evals: true`) |

The three modes are mutually exclusive: **batch** (`prompts` list), **interactive** (stdin REPL), **evals** (from file).

---

### `judge` — LLM-as-judge  *(eval only)*

| Key | Description |
|-----|-------------|
| `provider` | `"nebius"` or `"together"` — selects the credential resolver |
| `endpoint` | OpenAI-compatible API base URL |
| `model` | Model name passed to the judge API |

Credentials are resolved automatically via `~/Documents/dev/azure/providers.py`:
- **nebius**: Keychain service `nebius-api-key` or env `NEBIUS_API_KEY`
- **together**: Keychain service `together-api-key` or env `TOGETHER_API_KEY`

---

### `metrics` — eval criteria  *(eval only)*

Each entry in the `metrics` array defines one GEval metric:

| Key | Description |
|-----|-------------|
| `name` | Metric label (appears in results output) |
| `criteria` | Natural-language rubric fed verbatim to the LLM judge |
| `threshold` | Minimum score (0–1) for a test case to pass |

---

## Constraints

These must be consistent across sections or the run will fail at startup:

- `model.block_size` == `dataset.max_seq_len`
- `model.vocab_size` == `tokenizer.vocab_size`
- `dataset.tok_id` == `tokenizer.tok_id`
- `model.n_embd` divisible by `model.n_head`

---

## Output structure

```
runs/
  train/<run_id>/
    checkpoints/
      best.pt        # best validation loss checkpoint
      latest.pt      # most recent periodic checkpoint
    manifest.json    # config hash, lineage, summary stats
    metrics.jsonl    # per-step loss (if disable_jsonl: false)
    tensorboard/     # TensorBoard event files (if disable_tensorboard: false)
  infer/<infer_id>/
    generations.jsonl
    manifest.json
  evals/<eval_id>/
    generations.jsonl
    results.jsonl
    summary.json
    manifest.json
  registry.jsonl     # global log of all train and infer runs
```
