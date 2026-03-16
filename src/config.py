import json
import platform
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import torch

from model import resolve_checkpoint_path


def _assert_project_root():
    required = ['src', 'config', 'artifacts', 'runs']
    missing = [d for d in required if not Path(d).exists()]
    if missing:
        raise RuntimeError(
            f"Working directory '{Path.cwd()}' doesn't look like project root.\n"
            f"Missing: {missing}\n"
            f"Run from the nanoforge/ project root."
        )


_assert_project_root()


# ─── Tokenizer Config ────────────────────────────────────────────────────────

@dataclass
class TokenizerConfig:
    tok_id: str
    type: str
    vocab_size: int
    special_tokens: list
    training_corpus: object
    min_frequency: int = 2

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(**d)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)


# ─── Dataset Config ───────────────────────────────────────────────────────────

@dataclass
class DatasetConfig:
    ds_id: str
    dataset_type: str
    tok_id: str
    raw_source: str
    max_seq_len: int
    train_split: float
    val_split: float
    dtype: str
    max_tokens: int = None

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        d = {k: v for k, v in d.items() if not k.startswith('_')}
        return cls(**d)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)


# ─── Run Config ───────────────────────────────────────────────────────────────

@dataclass
class RunConfig:
    meta: dict
    environment: dict
    tokenizer: dict
    dataset: dict
    model: dict
    training: dict
    observability: dict

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(**d)

    def validate(self):
        tok_id = self.tokenizer['tok_id']
        tok_manifest_path = Path('artifacts/tokenizers') / tok_id / 'tokenizer_manifest.json'
        with open(tok_manifest_path) as f:
            tok_manifest = json.load(f)

        data_dir = Path(self.dataset['data_dir'])
        with open(data_dir / 'dataset_manifest.json') as f:
            ds_manifest = json.load(f)

        assert self.model['vocab_size'] == tok_manifest['vocab_size'], (
            f"model.vocab_size {self.model['vocab_size']} != "
            f"tokenizer vocab_size {tok_manifest['vocab_size']} (from manifest)"
        )
        assert self.model['block_size'] == ds_manifest['max_seq_len'], (
            f"model.block_size {self.model['block_size']} != "
            f"dataset max_seq_len {ds_manifest['max_seq_len']} (from manifest)"
        )
        assert ds_manifest['tok_id'] == tok_id, (
            f"dataset was encoded with tok_id '{ds_manifest['tok_id']}' "
            f"but config specifies tokenizer.tok_id '{tok_id}'"
        )
        assert self.model['n_embd'] % self.model['n_head'] == 0, (
            f"n_embd {self.model['n_embd']} not divisible by n_head {self.model['n_head']}"
        )

    def resolve(self, run_id):
        """Return a dict with run_id + runtime metadata stamped in."""
        d = asdict(self)
        d['_resolved'] = {
            'run_id':       run_id,
            'timestamp':    datetime.now().isoformat(),
            'torch_version': torch.__version__,
            'python':       sys.version.split()[0],
            'platform':     platform.platform(),
        }
        return d

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @property
    def run_name(self):
        return self.meta['run_name']


# ─── Infer Config ─────────────────────────────────────────────────────────────

@dataclass
class InferConfig:
    meta: dict
    source: dict        # run_id, checkpoint
    model: dict         # injected from resolved_run.json
    tokenizer: dict     # injected from resolved_run.json
    generation: dict
    input: dict

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)

        # Resolve run_id: explicit or latest
        run_id = d.get('source', {}).get('run_id')
        if run_id:
            run_dir = Path('runs/train') / run_id
        else:
            runs = [p for p in Path('runs/train').iterdir() if p.is_dir()]
            run_dir = sorted(runs, key=lambda p: p.stat().st_mtime)[-1]
            run_id = run_dir.name

        resolved = json.load(open(run_dir / 'resolved_run.json'))

        # Inject $from_run sections
        model_section = d.get('model', {})
        if model_section.get('$from_run'):
            d['model'] = resolved['model']

        tok_section = d.get('tokenizer', {})
        if tok_section.get('$from_run'):
            d['tokenizer'] = resolved['tokenizer']

        # Ensure source is set
        d.setdefault('source', {})
        d['source']['run_id'] = run_id

        return cls(
            meta=d.get('meta', {}),
            source=d['source'],
            model=d['model'],
            tokenizer=d['tokenizer'],
            generation=d.get('generation', {'max_new_tokens': 200, 'temperature': 1.0, 'top_k': 0}),
            input=d.get('input', {'prompts': ['\n']}),
        )

    def resolve(self, infer_id):
        d = asdict(self)
        d['_resolved'] = {
            'infer_id':  infer_id,
            'timestamp': datetime.now().isoformat(),
        }
        return d

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @property
    def infer_name(self):
        return self.meta.get('infer_name', 'infer')

    @property
    def checkpoint_path(self):
        run_id   = self.source['run_id']
        ckpt_key = self.source.get('checkpoint', 'best')
        ckpt_dir = Path('runs/train') / run_id / 'checkpoints'
        return resolve_checkpoint_path(ckpt_dir, ckpt_key)
