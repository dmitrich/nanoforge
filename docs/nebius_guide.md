# Running alg3 on Nebius — Step-by-Step Guide

This guide covers the complete workflow: provisioning a GPU instance on Nebius,
uploading the project, training, monitoring, and retrieving results.

---

## Overview

```
Local machine                          Nebius GPU instance
─────────────────                      ──────────────────────────────
alg3/                                  /home/user/alg3/
  data/raw/          ──── rsync ──▶      data/raw/
  artifacts/         ──── rsync ──▶      artifacts/
  src/               ──── rsync ──▶      src/
  configs/           ──── rsync ──▶      configs/
  notebooks/         ──── rsync ──▶      notebooks/
  runs/              ◀─── rsync ────     runs/train/<run_id>/
```

**What you upload once:** source files + dataset shards (~200 MB)
**What you upload per experiment:** updated `configs/run_exp001.json` (~2 KB)
**What you download after training:** `runs/train/<run_id>/` (checkpoints + metrics)

---

## Prerequisites (local machine)

- Nebius account at [nebius.ai](https://nebius.ai) with billing enabled
- SSH key pair — upload your public key to Nebius before creating an instance
- `rsync` installed (comes with macOS/Linux)
- alg3 project set up locally with dataset already prepared:
  ```bash
  cd /Users/dmitrit/Documents/dev/alg3
  python setup_tokenizer.py
  python src/dataset_prep.py configs/ds_tinystories_pretrain.json
  ```

---

## Step 1 — Create a GPU Instance on Nebius

1. Log in to the [Nebius Console](https://console.nebius.ai)
2. Go to **Compute** → **Create instance**
3. Choose a GPU configuration. Recommended for this project:

   | Option     | GPU          | VRAM  | Good for              |
   |------------|--------------|-------|-----------------------|
   | Budget     | L40S         | 48 GB | experiments, 1k steps |
   | Standard   | H100 SXM     | 80 GB | full training runs    |

4. **Boot image:** select the pre-built **PyTorch** image (Ubuntu 22.04 + CUDA 12.x + PyTorch pre-installed). This skips the GPU driver setup entirely.
5. **Disk:** 100 GB SSD is sufficient.
6. **SSH key:** paste your public key (`~/.ssh/id_rsa.pub` or `~/.ssh/id_ed25519.pub`).
7. Click **Create**. Note the **external IP address** shown in the instance list.

---

## Step 2 — Connect via SSH

```bash
# Replace <IP> with your instance's external IP
ssh user@<IP>

# Verify GPU is visible
nvidia-smi
# Expected: GPU model, driver version, CUDA version

# Verify PyTorch sees CUDA
python3 -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## Step 3 — Upload the Project

Run these commands from your **local machine** inside the alg3 project root.

```bash
cd /Users/dmitrit/Documents/dev/alg3
NEBIUS="user@<IP>"           # replace with your instance IP
REMOTE="~/alg3"              # destination on the remote machine

# Create the remote directory structure
ssh $NEBIUS "mkdir -p $REMOTE/runs/train $REMOTE/runs/infer"

# Upload everything needed for training
# (excludes .venv, __pycache__, and existing run outputs)
rsync -avz --progress \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='runs/train/*/checkpoints' \
  src/ configs/ artifacts/ notebooks/ data/raw/ \
  runs/registry.jsonl \
  $NEBIUS:$REMOTE/

echo "Upload complete"
```

> **Tip:** `artifacts/datasets/` (~200 MB) is the largest upload. On a fast connection
> this takes ~30 seconds. On slow connections, consider uploading just once and reusing.

---

## Step 4 — Set Up the Python Environment on Nebius

SSH into the instance and run:

```bash
ssh user@<IP>
cd ~/alg3

# The PyTorch image already has torch + numpy.
# Install the two additional packages needed:
pip install tokenizers tensorboard --quiet

# Quick sanity check
python3 -c "
import torch, tokenizers
print('torch     :', torch.__version__)
print('CUDA      :', torch.cuda.is_available())
print('GPU       :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')
print('tokenizers:', tokenizers.__version__)
"
```

---

## Step 5 — Launch JupyterLab

On the **remote instance:**

```bash
# Start JupyterLab on port 8888, no browser (headless server)
cd ~/alg3
jupyter lab --no-browser --port=8888 --ip=0.0.0.0
```

Copy the token URL shown in the output — it looks like:
```
http://127.0.0.1:8888/lab?token=abc123...
```

On your **local machine**, open an SSH tunnel:

```bash
# Keep this terminal open while you work
ssh -N -L 8888:localhost:8888 user@<IP>
```

Now open your browser and go to:
```
http://localhost:8888
```

Paste the token from the remote terminal when prompted.

> **Alternative:** if Nebius provides a managed JupyterHub (check the console under
> **AI Notebooks**), you can skip Steps 5's SSH tunnel — just open the hosted URL directly.

---

## Step 6 — Edit Cell 1 in Each Notebook

Before running any notebook, update **Cell 1** (the `os.chdir` cell) to point to the
remote project path. Open each notebook and change:

```python
# BEFORE (local):
os.chdir("/Users/dmitrit/Documents/dev/alg3")

# AFTER (Nebius):
os.chdir("/home/user/alg3")   # adjust if your home dir differs
```

This is the **only change** needed. Everything else — src imports, config paths,
artifact paths — works identically on local and cloud.

---

## Step 7 — Run the Notebooks in Order

### `01_tokenizer.ipynb` — one-time

Extracts `vocab.json` + `merges.txt` into `artifacts/tokenizers/tok_bpe_8k/`.

> **Skip if you already uploaded `artifacts/tokenizers/tok_bpe_8k/` via rsync.**

Run: **Kernel → Restart Kernel and Run All Cells**

---

### `02_dataset.ipynb` — one-time

Splits the binary data into `shard_000.bin` (train) and `shard_001.bin` (val).

> **Skip if you already uploaded `artifacts/datasets/ds_tinystories_tok_bpe8k_T128/` via rsync.**

Run: **Kernel → Restart Kernel and Run All Cells**

---

### `03_train.ipynb` — main experiment

**Cell 3** auto-detects CUDA and prints the GPU name.
**Cell 4** patches the config device to `"cuda"` in-memory — no JSON editing needed.
**Cell 5** runs training. Output appears inline as steps complete.

Expected console output:
```
GPU: NVIDIA H100 SXM5 80GB
VRAM: 80.0 GB

Run ID:  run_exp001_20260301_120000
Run dir: runs/train/run_exp001_20260301_120000
Device:  cuda  |  Params: 66.9M  |  Steps: 1000
step     0: loss 9.1875  lr 0.00e+00
step    10: loss 8.2344  lr 2.50e-05
...
```

**Cell 6** plots the loss curve using `metrics.jsonl` after training completes.

> **To change experiment parameters** (more steps, different LR, etc.), edit
> `configs/run_exp001.json` before running Cell 4.

---

### `04_infer.ipynb` — generate text

- **Cell 3:** `run_inference()` — zero args, auto-loads latest checkpoint.
- **Cell 4:** `run_inference("configs/infer_exp001.json")` — explicit config.
- **Cell 5:** Interactive — edit `PROMPT`, `TEMPERATURE`, `TOP_K` inline and re-run.
- **Cell 6:** Browse all past inference runs with generation previews.

---

## Step 8 — Monitor Training with TensorBoard

While training is running (or after), open a **second terminal** and run:

**On the remote instance:**
```bash
cd ~/alg3
tensorboard --logdir runs/train/ --port=6006 --host=0.0.0.0
```

**On your local machine** (second SSH tunnel):
```bash
ssh -N -L 6006:localhost:6006 user@<IP>
```

Open: [http://localhost:6006](http://localhost:6006)

You'll see real-time `Loss/train_step`, `Loss/val`, and `LR` curves.

---

## Step 9 — Download Results

After training, sync the run folder back to your local machine:

```bash
cd /Users/dmitrit/Documents/dev/alg3
NEBIUS="user@<IP>"
REMOTE="~/alg3"

# Download all training runs (checkpoints + metrics + manifests)
rsync -avz --progress \
  $NEBIUS:$REMOTE/runs/train/ \
  runs/train/

# Update the local registry
python src/manifest.py rebuild

# Verify the downloaded run
python src/manifest.py list
```

The downloaded run folder contains:
```
runs/train/run_exp001_20260301_120000/
  manifest.json          # status=completed, summary with final losses
  resolved_run.json      # full config as-run (device=cuda stamped in)
  metrics.jsonl          # every logged metric, every step
  checkpoints/
    best.pt              # lowest val loss checkpoint
    latest.pt            # final step checkpoint
  tb/                    # TensorBoard event files
```

---

## Step 10 — Run Inference Locally on the Downloaded Checkpoint

After downloading, run inference on your local machine with no code changes:

```bash
cd /Users/dmitrit/Documents/dev/alg3
.venv/bin/python src/infer.py
# auto-finds the latest run (which is now the one you downloaded)
```

Or with the explicit config:

```bash
.venv/bin/python src/infer.py configs/infer_exp001.json
```

---

## Step 11 — Stop the Instance

**Important:** Nebius bills by the hour while the instance is running.

```bash
# From the remote instance, shut down Jupyter first
# Ctrl+C in the Jupyter terminal

# Then from the Nebius Console:
# Compute → Your Instance → Stop
```

> **Tip:** Use **Stop** (not Delete) to preserve the disk. You can restart the same
> instance for the next training run without re-uploading the dataset.

---

## Uploading a New Experiment

For subsequent experiments — just upload the new config and re-run `03_train.ipynb`:

```bash
# Local → upload only the changed config
rsync -avz configs/run_exp002.json user@<IP>:~/alg3/configs/

# Remote: open 03_train.ipynb, change patched_path to point to run_exp002.json, run
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Upload project | `rsync -avz src/ configs/ artifacts/ notebooks/ data/raw/ runs/registry.jsonl user@<IP>:~/alg3/` |
| SSH tunnel (Jupyter) | `ssh -N -L 8888:localhost:8888 user@<IP>` |
| SSH tunnel (TensorBoard) | `ssh -N -L 6006:localhost:6006 user@<IP>` |
| Download results | `rsync -avz user@<IP>:~/alg3/runs/train/ runs/train/` |
| List runs (local) | `python src/manifest.py list` |
| Verify run integrity | `python src/manifest.py verify runs/train/<run_id>` |
| Rebuild registry | `python src/manifest.py rebuild` |
