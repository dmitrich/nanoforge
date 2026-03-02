import json
import sys
import numpy as np
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import DatasetConfig


class DatasetPrep:

    @classmethod
    def run(cls, cfg: DatasetConfig):
        out_dir = Path('artifacts/datasets') / cfg.ds_id
        out_dir.mkdir(parents=True, exist_ok=True)

        raw = cfg.raw_source
        if raw.endswith('.bin'):
            data = np.fromfile(raw, dtype=np.uint16)
        else:
            # text file: load tokenizer and encode
            from tokenizer import Tokenizer
            tok  = Tokenizer.load(cfg.tok_id)
            text = Path(raw).read_text(encoding='utf-8')
            ids  = tok.encode(text)
            data = np.array(ids, dtype=np.uint16)

        n          = int(cfg.train_split * len(data))
        train_data = data[:n]
        val_data   = data[n:]

        train_data.tofile(out_dir / 'shard_000.bin')
        val_data.tofile(out_dir / 'shard_001.bin')

        manifest = {
            'ds_id':        cfg.ds_id,
            'tok_id':       cfg.tok_id,
            'raw_source':   raw,
            'train_tokens': int(len(train_data)),
            'val_tokens':   int(len(val_data)),
            'max_seq_len':  cfg.max_seq_len,
            'dtype':        cfg.dtype,
            'created_at':   datetime.now().isoformat(),
        }
        with open(out_dir / 'dataset_manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)

        print(f"Dataset '{cfg.ds_id}': {len(train_data):,} train / {len(val_data):,} val tokens")
        print(f"Output: {out_dir}")


if __name__ == '__main__':
    cfg = DatasetConfig.load(sys.argv[1])
    DatasetPrep.run(cfg)
