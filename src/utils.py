import json
import hashlib
import random
from pathlib import Path

import numpy as np
import torch


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def append_jsonl(path, record):
    with open(path, 'a') as f:
        f.write(json.dumps(record) + '\n')


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def hash_dict(d):
    s = json.dumps(d, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def hash_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()[:12]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(requested):
    if requested == 'mps' and torch.backends.mps.is_available():
        return 'mps'
    if requested == 'cuda' and torch.cuda.is_available():
        return 'cuda'
    return 'cpu'
