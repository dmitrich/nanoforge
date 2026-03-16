# src/tokenizer.py — Tokenizer inference class

import json
from pathlib import Path


ARTIFACTS_DIR = Path("artifacts")


class Tokenizer:
    def __init__(self, tokenizer, manifest: dict):
        self._tokenizer = tokenizer
        self._manifest  = manifest

    @classmethod
    def load(cls, tok_id: str) -> "Tokenizer":
        manifest_path = ARTIFACTS_DIR / "tokenizers" / tok_id / "tokenizer_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Tokenizer manifest not found: {manifest_path}")
        with open(manifest_path) as f:
            manifest = json.load(f)

        from tokenizers import ByteLevelBPETokenizer
        tokenizer = ByteLevelBPETokenizer.from_file(
            manifest["vocab_path"],
            manifest["merges_path"],
        )
        return cls(tokenizer, manifest)

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tokenizer.decode(ids)

    def get_stop_token_ids(self, stop_strings: list[str]) -> list[int]:
        result = []
        for s in stop_strings:
            ids = self.encode(s)
            if ids:
                result.append(ids[0])
        return result

    @property
    def vocab_size(self) -> int:
        return self._manifest["vocab_size"]
