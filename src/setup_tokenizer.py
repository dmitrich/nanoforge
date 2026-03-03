"""
One-time setup: extract BPE tokenizer files from local source into artifacts/.
All source files live inside this project under data/raw/ — no external dependencies.

Run from alg3/ project root:
    python setup_tokenizer.py
"""
import json
import pathlib

model_path = pathlib.Path("data/raw/98m_8k_bpe.model")
vocab_path  = pathlib.Path("data/raw/98m_8k_bpe.vocab")
out_dir     = pathlib.Path("artifacts/tokenizers/tok_bpe_8k")
out_dir.mkdir(parents=True, exist_ok=True)

# Extract merges from JSON model file
model_data = json.loads(model_path.read_text(encoding='utf-8'))
merges_txt = model_data["merges"]

# Build vocab dict: token → index (order from .vocab file)
vocab = {}
with open(vocab_path, encoding='utf-8') as f:
    for idx, line in enumerate(f):
        parts = line.strip().split('\t')
        if parts:
            vocab[parts[0]] = idx

(out_dir / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding='utf-8')
(out_dir / "merges.txt").write_text(merges_txt, encoding='utf-8')

print(f"vocab.json : {len(vocab)} tokens  → {out_dir / 'vocab.json'}")
print(f"merges.txt : written              → {out_dir / 'merges.txt'}")
