import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class ShardedDataset(Dataset):
    _SHARD = {'train': 'shard_000.bin', 'val': 'shard_001.bin'}

    def __init__(self, ds_id: str, split: str, block_size: int):
        shard_path = Path('artifacts/datasets') / ds_id / self._SHARD[split]
        raw        = np.fromfile(shard_path, dtype=np.uint16).astype(np.int64)
        self.data       = torch.from_numpy(raw)
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.block_size]
        y = self.data[idx + 1: idx + self.block_size + 1]
        return x, y


def build_dataloaders(dataset_cfg: dict, training_cfg: dict):
    ds_id       = dataset_cfg['ds_id']
    block_size  = dataset_cfg['max_seq_len']
    batch_size  = training_cfg['batch_size']
    num_workers = training_cfg.get('num_workers', 0)

    train_ds = ShardedDataset(ds_id, 'train', block_size)
    val_ds   = ShardedDataset(ds_id, 'val',   block_size)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, drop_last=True,
    )
    return train_loader, val_loader
