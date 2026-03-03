# Dependencies: evals.py

Full dependency tree for `src/evals.py`.

```
evals.py
├── STDLIB: json, sys, os, datetime, pathlib
├── THIRD-PARTY
│   ├── torch
│   ├── deepeval  (GEval, LLMTestCase, LLMTestCaseParams)
│   └── openai    (OpenAI, AzureOpenAI — used transitively via judge.py)
└── LOCAL (src/)
    ├── judge.py       — JudgeLLM + build_judge; wraps any OpenAI-compatible API
    │   └── deps: os, platform, subprocess, deepeval, openai
    ├── manifest.py    — run tracking, registry (generate_run_id, create/complete/fail_manifest)
    │   └── deps: utils.py
    ├── model.py       — GPT model definition (GPT.from_checkpoint)
    │   └── deps: torch
    ├── tokenizer.py   — BPE tokenizer wrapper (Tokenizer.load)
    │   └── deps: tokenizers (lazy), config.py
    │       └── config.py — dataclass configs (TokenizerConfig, InferConfig, etc.)
    └── utils.py       — write_json, append_jsonl, get_device, hash_dict
        └── deps: json, hashlib, numpy, torch
```

## External credentials

`judge.py` → `build_judge` fetches API keys via:
- macOS Keychain (`security find-generic-password`)
- Environment variable (e.g. `NEBIUS_API_KEY`)
- `~/Documents/dev/azure/providers.py` (project-local helper, not in repo)

## NOT used by evals.py

| Module | Reason excluded |
|--------|-----------------|
| `infer.py` | Independent script; shares model/tokenizer/manifest/utils but is never imported |
| `config.py` | Used only by `tokenizer.py` (TokenizerConfig) and `infer.py` (InferConfig) |
| `train.py` | Training-only; no runtime dependency |
| `dataloader.py` / `dataset_prep.py` | Dataset pipeline; not needed at eval time |
| `tracker.py` | TensorBoard/JSONL metrics; training-only |
