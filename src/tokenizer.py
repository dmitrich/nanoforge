import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import TokenizerConfig


class Tokenizer:
    def __init__(self, tok_id, _tokenizer, vocab_size):
        self.tok_id     = tok_id
        self.vocab_size = vocab_size
        self._tok       = _tokenizer

    @classmethod
    def load(cls, tok_id):
        tok_dir     = Path('artifacts/tokenizers') / tok_id
        vocab_path  = tok_dir / 'vocab.json'
        merges_path = tok_dir / 'merges.txt'

        from tokenizers import ByteLevelBPETokenizer
        tok = ByteLevelBPETokenizer(
            vocab=str(vocab_path),
            merges=str(merges_path),
        )

        vocab_size = tok.get_vocab_size()

        manifest = {
            'tok_id':      tok_id,
            'vocab_size':  vocab_size,
            'vocab_path':  str(vocab_path),
            'merges_path': str(merges_path),
            'loaded_at':   datetime.now().isoformat(),
        }
        with open(tok_dir / 'tokenizer_manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)

        return cls(tok_id, tok, vocab_size)

    def encode(self, text):
        if not text:
            return [0]
        return self._tok.encode(text).ids

    def decode(self, ids):
        return self._tok.decode(ids)

    def encode_batch(self, texts):
        return [self.encode(t) for t in texts]
    
    def get_token_id(self, text):
        """Get the token ID for a specific text string."""
        tokens = self.encode(text)
        return tokens[0] if len(tokens) == 1 else None
    
    def get_stop_token_ids(self, stop_strings):
        """
        Convert stop strings to token IDs.
        
        Args:
            stop_strings: List of strings that should trigger stopping
        
        Returns:
            List of token IDs
        """
        stop_ids = []
        for stop_str in stop_strings:
            tokens = self.encode(stop_str)
            # For single-token stops, add the token ID
            if len(tokens) == 1:
                stop_ids.append(tokens[0])
            # For multi-token stops, we'll just use the first token as approximation
            # (proper multi-token stopping would require sequence matching)
            elif len(tokens) > 1:
                stop_ids.append(tokens[0])
        return stop_ids


if __name__ == '__main__':
    cfg = TokenizerConfig.load(sys.argv[1])
    print(f"Tokenizer train not implemented for pre-built BPE. Use setup_tokenizer.py.")
