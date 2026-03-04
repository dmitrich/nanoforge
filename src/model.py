import json
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from torch.nn import functional as F

from safetensors import safe_open
from safetensors.torch import save_file


CHECKPOINT_EXT = '.safetensors'
LEGACY_CHECKPOINT_EXT = '.pt'


def _checkpoint_name(stem: str, ext: str = CHECKPOINT_EXT) -> str:
    return f'{stem}{ext}'


def resolve_checkpoint_path(ckpt_dir, ckpt_key: str) -> str:
    from pathlib import Path

    ckpt_dir = Path(ckpt_dir)

    if ckpt_key == 'best':
        candidates = [
            ckpt_dir / _checkpoint_name('best'),
            ckpt_dir / _checkpoint_name('best', LEGACY_CHECKPOINT_EXT),
            ckpt_dir / _checkpoint_name('latest'),
            ckpt_dir / _checkpoint_name('latest', LEGACY_CHECKPOINT_EXT),
        ]
    elif ckpt_key == 'latest':
        candidates = [
            ckpt_dir / _checkpoint_name('latest'),
            ckpt_dir / _checkpoint_name('latest', LEGACY_CHECKPOINT_EXT),
        ]
    else:
        candidates = [
            ckpt_dir / _checkpoint_name(ckpt_key),
            ckpt_dir / _checkpoint_name(ckpt_key, LEGACY_CHECKPOINT_EXT),
        ]

    for path in candidates:
        if path.exists():
            return str(path)
    return str(candidates[0])


def _to_saveable_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().contiguous()


def _encode_metadata(value, prefix: str, tensors: dict):
    if torch.is_tensor(value):
        tensor_name = prefix
        tensors[tensor_name] = _to_saveable_tensor(value)
        return {'__tensor__': tensor_name}
    if isinstance(value, dict):
        return {str(k): _encode_metadata(v, f'{prefix}.{k}', tensors) for k, v in value.items()}
    if isinstance(value, list):
        return [_encode_metadata(v, f'{prefix}.{i}', tensors) for i, v in enumerate(value)]
    if isinstance(value, tuple):
        return {'__tuple__': [_encode_metadata(v, f'{prefix}.{i}', tensors) for i, v in enumerate(value)]}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f'Unsupported checkpoint metadata type: {type(value)!r}')


def _decode_metadata(value, tensors: dict):
    if isinstance(value, dict):
        if '__tensor__' in value:
            return tensors[value['__tensor__']]
        if '__tuple__' in value:
            return tuple(_decode_metadata(v, tensors) for v in value['__tuple__'])
        return {k: _decode_metadata(v, tensors) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode_metadata(v, tensors) for v in value]
    return value


@dataclass
class ModelConfig:
    architecture: str
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float
    block_size: int
    bias: bool
    vocab_size: int

    @classmethod
    def from_dict(cls, d):
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in fields})

    def to_dict(self):
        return asdict(self)


class Head(nn.Module):
    def __init__(self, cfg, head_size):
        super().__init__()
        self.key   = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.query = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.value = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(cfg.block_size, cfg.block_size)))
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        head_size = cfg.n_embd // cfg.n_head
        self.heads   = nn.ModuleList([Head(cfg, head_size) for _ in range(cfg.n_head)])
        self.proj    = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedFoward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.ReLU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.sa   = MultiHeadAttention(cfg)
        self.ffwd = FeedFoward(cfg)
        self.ln1  = nn.LayerNorm(cfg.n_embd)
        self.ln2  = nn.LayerNorm(cfg.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding_table    = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f   = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=0, stop_token_ids=None):
        """
        Generate tokens autoregressively.
        
        Args:
            idx: Starting token indices (B, T)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            stop_token_ids: List of token IDs that trigger early stopping
        
        Returns:
            Generated token indices (B, T+generated)
        """
        if stop_token_ids is None:
            stop_token_ids = []
        
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            if temperature != 1.0:
                logits = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
            # Check for stop tokens
            if stop_token_ids and idx_next.item() in stop_token_ids:
                break
        
        return idx

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def save_checkpoint(self, path, step, optimizer, val_loss):
        model_tensors = {
            f'model.{name}': _to_saveable_tensor(tensor)
            for name, tensor in self.state_dict().items()
        }
        optimizer_tensors = {}
        optimizer_state = _encode_metadata(optimizer.state_dict(), 'optimizer', optimizer_tensors)
        metadata = {
            'nanoforge_checkpoint_format': 'v1',
            'cfg': json.dumps(self.cfg.to_dict()),
            'step': json.dumps(step),
            'val_loss': json.dumps(float(val_loss)),
            'optimizer': json.dumps(optimizer_state),
        }
        save_file({**model_tensors, **optimizer_tensors}, str(path), metadata=metadata)

    @classmethod
    def from_checkpoint(cls, path):
        path = str(path)
        if path.endswith(LEGACY_CHECKPOINT_EXT):
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            cfg  = ModelConfig.from_dict(ckpt['cfg'])
            model = cls(cfg)
            model.load_state_dict(ckpt['model'])
            return model, ckpt

        with safe_open(path, framework='pt', device='cpu') as f:
            metadata = f.metadata() or {}
            tensors = {key: f.get_tensor(key) for key in f.keys()}

        cfg = ModelConfig.from_dict(json.loads(metadata['cfg']))
        model = cls(cfg)
        model_state = {
            key.removeprefix('model.'): tensor
            for key, tensor in tensors.items()
            if key.startswith('model.')
        }
        model.load_state_dict(model_state)

        optimizer_meta = json.loads(metadata.get('optimizer', '{}'))
        ckpt = {
            'model': model_state,
            'cfg': cfg.to_dict(),
            'step': json.loads(metadata.get('step', '0')),
            'val_loss': json.loads(metadata.get('val_loss', '0.0')),
            'optimizer': _decode_metadata(optimizer_meta, tensors),
        }
        return model, ckpt
