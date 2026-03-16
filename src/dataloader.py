# src/dataloader.py — Sharded binary dataset loader

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

_PIN_MEMORY = torch.cuda.is_available()


class ShardedDataset(Dataset):
    _SHARDS = {'train': 'shard_000.bin', 'val': 'shard_001.bin'}

    def __init__(self, data_dir: Path, split: str, block_size: int):
        shard_path = Path(data_dir) / self._SHARDS[split]
        raw = np.fromfile(shard_path, dtype=np.uint16).astype(np.int64)
        self.data       = torch.from_numpy(raw)
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx     : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


def build_dataloaders(dataset_cfg: dict, training_cfg: dict):
    data_dir    = Path(dataset_cfg['data_dir'])
    batch_size  = training_cfg['batch_size']
    num_workers = training_cfg.get('num_workers', 0)

    with open(data_dir / 'dataset_manifest.json') as f:
        manifest = json.load(f)
    block_size = manifest['max_seq_len']

    train_ds = ShardedDataset(data_dir, 'train', block_size)
    val_ds   = ShardedDataset(data_dir, 'val',   block_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=_PIN_MEMORY)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=_PIN_MEMORY)
    return train_loader, val_loader
