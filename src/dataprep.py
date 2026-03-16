"""src/dataprep.py — Unified 3-phase data preparation pipeline

Usage:
    python src/dataprep.py config/dataprep.json

Phases:
    1. download  — Stream from HuggingFace → data/raw/<date>_<seq>_<size>tok.txt
    2. tokenize  — Train BPE tokenizer → artifacts/tokenizers/<tok_id>/
    3. encode    — Tokenize .txt → binary shards in artifacts/datasets/<ds_id>/
"""

import hashlib
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path


PROBE_SIZE    = 100
ARTIFACTS_DIR = Path("artifacts")


# ── helpers ───────────────────────────────────────────────────────────────────

def _hash_dict(d: dict) -> str:
    s = json.dumps(d, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()[:12]


# ── Phase 1: Download ─────────────────────────────────────────────────────────

def _probe_stream(ds_iter, text_field: str) -> tuple[list[str], dict, int]:
    """Stream first PROBE_SIZE stories; detect special tokens and newline counts."""
    sample: list[str] = []
    for example in ds_iter:
        text = example[text_field]
        if text:
            sample.append(text)
        if len(sample) >= PROBE_SIZE:
            break

    special_pattern = re.compile(r"<\|[^|]+\|>")
    special_found: dict[str, int] = {}
    for text in sample:
        for match in special_pattern.findall(text):
            special_found[match] = special_found.get(match, 0) + 1

    stories_with_newlines = sum(1 for t in sample if "\n" in t)
    return sample, special_found, stories_with_newlines


def _build_output_path(output_dir: Path, size_mode: str, size_value: int) -> Path:
    today = date.today()
    date_prefix = today.strftime("%m%d")
    if size_mode == "tokens":
        v = size_value
        size_tag = f"{v // 1_000_000}M" if v >= 1_000_000 else f"{v // 1_000}k"
        size_tag += "tok"
    else:
        size_tag = f"{size_value}MB"
    existing = sorted(output_dir.glob(f"{date_prefix}_*_*.txt"))
    seq_num = len(existing) + 1
    return output_dir / f"{date_prefix}_{seq_num:03d}_{size_tag}.txt"


def phase_download(cfg: dict, force: bool) -> Path:
    """Phase 1: Download HuggingFace dataset to a .txt file. Idempotent."""
    output_dir = Path(cfg["output_dir"])
    sidecar    = output_dir / "last_download.json"
    cfg_hash   = _hash_dict(cfg)

    if not force and sidecar.exists():
        with open(sidecar) as f:
            prev = json.load(f)
        if prev.get("cfg_hash") == cfg_hash:
            existing_path = Path(prev["output_path"])
            if existing_path.exists():
                print("── Phase 1: Download [SKIPPED] ─────────────────────────────────────")
                print(f"  (already done: {existing_path})")
                return existing_path

    print("── Phase 1: Download ──────────────────────────────────────────────")
    dataset_name = cfg["dataset_name"]
    split        = cfg["split"]
    text_field   = cfg["text_field"]
    size_mode    = cfg["size"]["mode"]
    size_value   = cfg["size"]["value"]
    sep_token    = cfg["separation"]["token"]

    limit = size_value if size_mode == "tokens" else size_value * 1_048_576

    print(f"Loading '{dataset_name}' ({split}) ...")
    from datasets import load_dataset
    ds_iter = iter(load_dataset(dataset_name, split=split, streaming=True))

    print(f"Probing first {PROBE_SIZE} examples...")
    probed_stories, special_found, nl_count = _probe_stream(ds_iter, text_field)

    if special_found:
        print("  Special tokens detected in story text:")
        for token, count in sorted(special_found.items(), key=lambda x: -x[1]):
            print(f"    {token!r}  ({count} occurrences)")
    else:
        print(f"  No special tokens detected.  Using separator: {sep_token!r} [config]")

    if nl_count:
        print(f"  WARNING: {nl_count}/{PROBE_SIZE} stories contain newlines (paragraph breaks)")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = _build_output_path(output_dir, size_mode, size_value)

    accumulated = 0.0
    story_count = 0

    with output_path.open("w", encoding="utf-8") as out:
        def story_stream():
            yield from probed_stories
            for example in ds_iter:
                text = example[text_field]
                if text:
                    yield text

        for text in story_stream():
            if size_mode == "tokens":
                accumulated += len(text) / 4
            else:
                accumulated += len(text.encode("utf-8"))

            out.write(text)
            out.write(sep_token)
            out.write("\n")
            story_count += 1

            if story_count % 10_000 == 0:
                print(f"  {story_count:,} stories...")

            if accumulated >= limit:
                break

    size_mb    = output_path.stat().st_size / 1_048_576
    est_tokens = accumulated if size_mode == "tokens" else accumulated / 4
    print(f"Wrote {story_count:,} stories | ~{est_tokens:,.0f} est. tokens | {size_mb:.0f} MB → {output_path}")

    with open(sidecar, "w") as f:
        json.dump({"cfg_hash": cfg_hash, "output_path": str(output_path)}, f)

    return output_path


# ── Phase 2: Tokenize ─────────────────────────────────────────────────────────

def _train_bpe(input_txt: Path, vocab_size: int, min_frequency: int, special_tokens: list[str]):
    from tokenizers import ByteLevelBPETokenizer
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[str(input_txt)],
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
    )
    return tokenizer


def _save_tokenizer_artifact(tokenizer, tok_type: str, tok_id: str, vocab_size: int, special_tokens: list[str]) -> Path:
    out_dir = ARTIFACTS_DIR / "tokenizers" / tok_id
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_model(str(out_dir))  # writes vocab.json + merges.txt
    manifest = {
        "tok_id":         tok_id,
        "tok_type":       tok_type,
        "vocab_size":     vocab_size,
        "special_tokens": special_tokens,
        "vocab_path":     str(out_dir / "vocab.json"),
        "merges_path":    str(out_dir / "merges.txt"),
        "loaded_at":      datetime.now().isoformat(),
    }
    with open(out_dir / "tokenizer_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return out_dir


def phase_tokenize(cfg: dict, input_txt: Path, force: bool) -> str:
    """Phase 2: Train BPE tokenizer and save artifact. Idempotent."""
    tok_id        = cfg["tok_id"]
    manifest_path = ARTIFACTS_DIR / "tokenizers" / tok_id / "tokenizer_manifest.json"

    if not force and manifest_path.exists():
        print("── Phase 2: Tokenize [SKIPPED] ─────────────────────────────────────")
        print(f"  (already done: {manifest_path.parent})")
        return tok_id

    print("── Phase 2: Tokenize ──────────────────────────────────────────────")
    tok_type       = cfg["type"]
    vocab_size     = cfg["vocab_size"]
    min_frequency  = cfg.get("min_frequency", 2)
    special_tokens = cfg.get("special_tokens", ["<pad>", "<unk>", "<s>", "</s>"])

    assert vocab_size <= 65535, f"vocab_size {vocab_size} exceeds uint16 max (65535)"

    print(f"Training BPE tokenizer (vocab_size={vocab_size}, min_frequency={min_frequency})...")
    tokenizer = _train_bpe(input_txt, vocab_size, min_frequency, special_tokens)

    out_dir = _save_tokenizer_artifact(tokenizer, tok_type, tok_id, vocab_size, special_tokens)
    print(f"Saved: {out_dir}/")
    return tok_id


# ── Phase 3: Encode ───────────────────────────────────────────────────────────

def _encode_line_by_line(tok, input_txt: Path, max_tokens: int | None) -> list[int]:
    ids: list[int] = []
    last_report = 0
    with open(input_txt, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            ids.extend(tok.encode(line))
            if len(ids) - last_report >= 10_000_000:
                last_report = (len(ids) // 10_000_000) * 10_000_000
                print(f"  {len(ids):,} tokens...")
            if max_tokens is not None and len(ids) >= max_tokens:
                ids = ids[:max_tokens]
                break
    return ids


def _write_shards(ids: list[int], out_dir: Path, train_split: float) -> tuple[int, int]:
    import numpy as np
    split_at  = int(train_split * len(ids))
    train_ids = ids[:split_at]
    val_ids   = ids[split_at:]

    train_path = out_dir / "shard_000.bin"
    val_path   = out_dir / "shard_001.bin"

    np.array(train_ids, dtype=np.uint16).tofile(train_path)
    np.array(val_ids,   dtype=np.uint16).tofile(val_path)

    train_mb = train_path.stat().st_size / 1_048_576
    val_mb   = val_path.stat().st_size   / 1_048_576
    print(f"shard_000.bin: {len(train_ids):,} tokens ({train_mb:.0f} MB)")
    print(f"shard_001.bin: {len(val_ids):,} tokens ({val_mb:.0f} MB)")

    return len(train_ids), len(val_ids)


def phase_encode(cfg: dict, input_txt: Path, tok_id: str, force: bool) -> str:
    """Phase 3: Encode .txt to binary shards. Idempotent."""
    ds_id         = cfg["ds_id"]
    out_dir       = Path(cfg.get("output_dir", "data/clean"))
    manifest_path = out_dir / "dataset_manifest.json"

    if not force and manifest_path.exists():
        print("── Phase 3: Encode [SKIPPED] ────────────────────────────────────────")
        print(f"  (already done: {out_dir})")
        return ds_id

    print("── Phase 3: Encode ────────────────────────────────────────────────")
    max_seq_len = cfg["max_seq_len"]
    train_split = cfg["train_split"]
    dtype       = cfg.get("dtype", "uint16")
    max_tokens  = cfg.get("max_tokens")

    from tokenizer import Tokenizer
    tok = Tokenizer.load(tok_id)

    print(f"Encoding {input_txt}...")
    ids = _encode_line_by_line(tok, input_txt, max_tokens)

    out_dir.mkdir(parents=True, exist_ok=True)

    train_tokens, val_tokens = _write_shards(ids, out_dir, train_split)

    manifest = {
        "ds_id":        ds_id,
        "tok_id":       tok_id,
        "raw_source":   str(input_txt),
        "train_tokens": train_tokens,
        "val_tokens":   val_tokens,
        "max_seq_len":  max_seq_len,
        "dtype":        dtype,
        "created_at":   datetime.now().isoformat(),
    }
    with open(out_dir / "dataset_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved: {out_dir}/")
    return ds_id


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run(config_path: str) -> None:
    with open(config_path) as f:
        cfg = json.load(f)

    phases_cfg  = cfg.get("phases", {})
    do_download = phases_cfg.get("download", True)
    do_tokenize = phases_cfg.get("tokenize", True)
    do_encode   = phases_cfg.get("encode",   True)
    force       = phases_cfg.get("force",    False)

    dl_on  = "ON"  if do_download else "OFF"
    tok_on = "ON"  if do_tokenize else "OFF"
    enc_on = "ON"  if do_encode   else "OFF"
    f_on   = "ON"  if force       else "OFF"

    print("=== nanoforge dataprep ===")
    print(f"Phases: download={dl_on}  tokenize={tok_on}  encode={enc_on}  force={f_on}")
    print()

    dl_cfg  = cfg.get("download",  {})
    tok_cfg = cfg.get("tokenizer", {})
    ds_cfg  = cfg.get("dataset",   {})

    input_txt: Path | None = None

    if do_download:
        input_txt = phase_download(dl_cfg, force)
        print()

    # input_txt override for standalone tokenize/encode without download
    if tok_cfg.get("input_txt"):
        input_txt = Path(tok_cfg["input_txt"])

    tok_id: str | None = None

    if do_tokenize:
        assert input_txt is not None, "input_txt required for tokenize phase (run download first or set tokenizer.input_txt)"
        tok_id = phase_tokenize(tok_cfg, input_txt, force)
        print()

    if do_encode:
        if tok_id is None:
            tok_id = tok_cfg.get("tok_id")
        assert tok_id is not None, "tok_id required for encode phase"
        assert input_txt is not None, "input_txt required for encode phase (run download first or set tokenizer.input_txt)"
        phase_encode(ds_cfg, input_txt, tok_id, force)
        print()

    print("── Done ───────────────────────────────────────────────────────────")
    if input_txt:
        print(f"  txt  → {input_txt}")
    effective_tok_id = tok_id or tok_cfg.get("tok_id", "?")
    print(f"  tok  → {ARTIFACTS_DIR / 'tokenizers' / effective_tok_id}/")
    print(f"  data → {ds_cfg.get('output_dir', 'data/clean')}/")


if __name__ == "__main__":
    run(sys.argv[1])
