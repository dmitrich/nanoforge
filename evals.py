"""
evals.py — deepeval evaluation runner for alg3 GPT models

Usage:
    python evals.py configs/evals_stories.json   # explicit config
    python evals.py                              # latest training run + evals.json defaults

Each run produces:
    runs/evals/<eval_id>/
        manifest.json        — lineage, status, config hash
        resolved_eval.json   — fully resolved config stamped at runtime
        generations.jsonl    — raw inference output per test case
        results.jsonl        — per-case metric scores + judge reasoning
        summary.json         — aggregate pass/fail and per-metric stats

Judge credentials are resolved via ~/Documents/dev/azure/providers.py:
    - "nebius"   → Keychain: nebius-api-key   / env: NEBIUS_API_KEY
    - "together" → Keychain: together-api-key  / env: TOGETHER_API_KEY
    - "azure"    → Keychain: azure-openai-api-key / env: AZURE_OPENAI_API_KEY

    The eval config's "judge" section controls which provider, endpoint, and
    model to use. Falls back to values in the provider's config file if not set.
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from deepeval.metrics import GEval
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from openai import AzureOpenAI, OpenAI

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from manifest import generate_run_id, create_manifest, complete_manifest, fail_manifest
from model import GPT
from tokenizer import Tokenizer
from utils import append_jsonl, get_device, write_json


# ─── Judge LLM (provider-agnostic) ───────────────────────────────────────────

class JudgeLLM(DeepEvalBaseLLM):
    """DeepEvalBaseLLM wrapper for any OpenAI-compatible client (Azure, Nebius, Together)."""

    def __init__(self, client, model_name: str):
        # Set attributes before super().__init__() because it calls load_model()
        self.client     = client
        self.model_name = model_name
        super().__init__(model=model_name)

    def load_model(self):
        return self.client

    def generate(self, prompt: str, *args, **kwargs) -> str:
        schema = kwargs.get('schema')
        extra  = {"response_format": {"type": "json_object"}} if schema else {}
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **extra,
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str, *args, **kwargs) -> str:
        return self.generate(prompt, *args, **kwargs)

    def get_model_name(self) -> str:
        return self.model_name


def _build_judge(judge_cfg: dict) -> JudgeLLM:
    """
    Build a JudgeLLM from the eval config's "judge" section.

    Supported providers:  nebius | together | azure
    Credentials are resolved via ~/Documents/dev/azure/providers.py
    (Keychain on macOS, env var fallback).

    Judge config fields (all optional — fall back to provider config files):
        provider  : "nebius" | "together" | "azure"
        endpoint  : API base URL
        model     : model / deployment name
        api_version: Azure only
    """
    sys.path.insert(0, os.path.expanduser('~/Documents/dev/azure'))
    from providers import get_api_key, load_config as load_provider_config

    provider = judge_cfg.get("provider", "nebius")

    api_key = get_api_key(provider)
    if not api_key:
        raise ValueError(
            f"{provider} API key not found. "
            f"Set the appropriate env var or add to Keychain via providers.py."
        )

    prov_cfg = load_provider_config(provider)

    if provider == "azure":
        endpoint    = judge_cfg.get("endpoint")    or prov_cfg.get("endpoint")
        model_name  = judge_cfg.get("model")       or prov_cfg.get("deployment_name")
        api_version = judge_cfg.get("api_version") or prov_cfg.get("api_version", "2024-02-15-preview")
        client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )
    else:
        endpoint   = judge_cfg.get("endpoint") or prov_cfg.get("endpoint")
        model_name = judge_cfg.get("model")    or prov_cfg.get("current_model") or prov_cfg.get("default_model")
        client = OpenAI(api_key=api_key, base_url=endpoint)

    return JudgeLLM(client, model_name)


# ─── Config resolution ────────────────────────────────────────────────────────

def _resolve_run_refs(raw: dict) -> tuple:
    """
    Resolve $from_run references in model/tokenizer sections.
    Returns (resolved_raw, checkpoint_path, run_id).
    """
    source = raw.get("source", {})
    run_id = source.get("run_id")

    if run_id:
        run_dir = Path("runs/train") / run_id
    else:
        runs    = [p for p in Path("runs/train").iterdir() if p.is_dir()]
        run_dir = sorted(runs, key=lambda p: p.stat().st_mtime)[-1]
        run_id  = run_dir.name

    resolved_run = json.load(open(run_dir / "resolved_run.json"))

    if raw.get("model",     {}).get("$from_run"):
        raw["model"]     = resolved_run["model"]
    if raw.get("tokenizer", {}).get("$from_run"):
        raw["tokenizer"] = resolved_run["tokenizer"]

    raw.setdefault("source", {})["run_id"] = run_id

    ckpt_key = source.get("checkpoint", "best")
    ckpt_dir = run_dir / "checkpoints"
    if ckpt_key == "best":
        ckpt = ckpt_dir / "best.pt"
        ckpt_path = str(ckpt if ckpt.exists() else ckpt_dir / "latest.pt")
    elif ckpt_key == "latest":
        ckpt_path = str(ckpt_dir / "latest.pt")
    else:
        ckpt_path = str(ckpt_dir / f"{ckpt_key}.pt")

    return raw, ckpt_path, run_id


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_evals(config_path: str = None):
    if config_path is None:
        raw = {
            "meta":      {"eval_name": "evals"},
            "source":    {"run_id": None, "checkpoint": "best"},
            "model":     {"$from_run": True},
            "tokenizer": {"$from_run": True},
            "generation": {"max_new_tokens": 200, "temperature": 0.8, "top_k": 40},
            "judge":     {"provider": "nebius", "endpoint": "https://api.tokenfactory.nebius.com/v1/", "model": "meta-llama/Llama-3.3-70B-Instruct"},
            "test_cases_file": "evals.json",
            "metrics": [
                {
                    "name":      "Coherence",
                    "criteria":  "The story flows logically and is internally consistent.",
                    "threshold": 0.5,
                },
                {
                    "name":      "Fluency",
                    "criteria":  "The text is grammatically correct and reads naturally.",
                    "threshold": 0.5,
                },
            ],
        }
    else:
        with open(config_path) as f:
            raw = json.load(f)

    raw, ckpt_path, run_id = _resolve_run_refs(raw)

    eval_name       = raw.get("meta", {}).get("eval_name", "evals")
    judge_cfg       = raw.get("judge", {})
    metrics_cfg     = raw.get("metrics", [])
    test_cases_file = raw.get("test_cases_file", "evals.json")
    generation      = raw.get("generation", {})
    model_cfg       = raw["model"]

    # ── Create run directory ──────────────────────────────────────────────────
    eval_id  = generate_run_id(eval_name)
    eval_dir = Path("runs/evals") / eval_id
    eval_dir.mkdir(parents=True, exist_ok=True)

    resolved_eval = {**raw, "_resolved": {
        "eval_id":   eval_id,
        "timestamp": datetime.now().isoformat(),
    }}
    write_json(eval_dir / "resolved_eval.json", resolved_eval)
    create_manifest(eval_dir, eval_id, "eval", resolved_eval, lineage={"from_run": run_id})

    print(f"Eval ID:    {eval_id}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Tests file: {test_cases_file}")
    print(f"Metrics:    {[m['name'] for m in metrics_cfg]}")

    try:
        # ── Inference ─────────────────────────────────────────────────────────
        model, _ = GPT.from_checkpoint(ckpt_path)
        device   = get_device(model_cfg.get("device", "mps"))
        model    = model.to(device).eval()

        tok_id    = raw["tokenizer"]["tok_id"]
        tokenizer = Tokenizer.load(tok_id)

        max_new        = generation.get("max_new_tokens", 200)
        temperature    = generation.get("temperature", 0.8)
        top_k          = generation.get("top_k", 40)
        stop_strings   = generation.get("stop_tokens", [])
        stop_token_ids = tokenizer.get_stop_token_ids(stop_strings) if stop_strings else []

        with open(test_cases_file) as f:
            test_cases_raw = json.load(f)["test_cases"]

        print(f"\n--- Inference ({len(test_cases_raw)} test cases) ---")
        generations = []
        for tc in test_cases_raw:
            prompt = tc["input"]
            tokens = tokenizer.encode(prompt)
            idx    = torch.tensor([tokens], dtype=torch.long, device=device)
            with torch.no_grad():
                out = model.generate(
                    idx, max_new,
                    temperature=temperature,
                    top_k=top_k,
                    stop_token_ids=stop_token_ids,
                )
            text  = tokenizer.decode(out[0].tolist())
            entry = {"test_id": tc["id"], "input": prompt, "actual_output": text}
            generations.append(entry)
            append_jsonl(eval_dir / "generations.jsonl", entry)
            print(f"  [{tc['id']}] {prompt!r} → {len(text)} chars")

        # ── Build judge ───────────────────────────────────────────────────────
        print("\n--- Building judge ---")
        judge = _build_judge(judge_cfg)
        print(f"Judge: {judge.get_model_name()}")

        # ── Build GEval metrics from config ───────────────────────────────────
        metrics = [
            GEval(
                name=m["name"],
                criteria=m["criteria"],
                evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
                threshold=m.get("threshold", 0.5),
                model=judge,
                async_mode=False,
            )
            for m in metrics_cfg
        ]

        # ── Score each test case ──────────────────────────────────────────────
        print(f"\n--- Evaluating ({len(generations)} × {len(metrics)} metrics) ---")
        all_results = []
        for gen in generations:
            test_case    = LLMTestCase(input=gen["input"], actual_output=gen["actual_output"])
            case_metrics = {}
            for metric in metrics:
                metric.measure(test_case)
                case_metrics[metric.name] = {
                    "score":  round(metric.score, 4),
                    "reason": metric.reason,
                    "passed": metric.is_successful(),
                }

            result = {
                "test_id":       gen["test_id"],
                "input":         gen["input"],
                "actual_output": gen["actual_output"],
                "metrics":       case_metrics,
                "passed":        all(v["passed"] for v in case_metrics.values()),
            }
            all_results.append(result)
            append_jsonl(eval_dir / "results.jsonl", result)

            status = "PASS" if result["passed"] else "FAIL"
            print(f"  [{gen['test_id']}] {status}")
            for name, s in case_metrics.items():
                print(f"    {name}: {s['score']:.2f}  {s['reason'][:80]}")

        # ── Aggregate summary ─────────────────────────────────────────────────
        n_total  = len(all_results)
        n_passed = sum(1 for r in all_results if r["passed"])

        metrics_summary = {}
        for metric in metrics:
            scores = [r["metrics"][metric.name]["score"] for r in all_results]
            n_pass = sum(1 for r in all_results if r["metrics"][metric.name]["passed"])
            metrics_summary[metric.name] = {
                "mean":      round(sum(scores) / len(scores), 3),
                "min":       round(min(scores), 3),
                "max":       round(max(scores), 3),
                "pass_rate": round(n_pass / n_total, 3),
            }

        summary = {
            "eval_id":         eval_id,
            "checkpoint":      ckpt_path,
            "total":           n_total,
            "passed":          n_passed,
            "failed":          n_total - n_passed,
            "pass_rate":       round(n_passed / n_total, 3) if n_total else 0.0,
            "metrics_summary": metrics_summary,
        }
        write_json(eval_dir / "summary.json", summary)
        complete_manifest(eval_dir, summary)

        print(f"\n=== Eval complete: {eval_id} ===")
        print(f"Pass rate: {n_passed}/{n_total} ({summary['pass_rate']:.0%})")
        for name, s in metrics_summary.items():
            print(f"  {name}: mean={s['mean']:.2f}  pass_rate={s['pass_rate']:.0%}")
        print(f"Output: {eval_dir}/")

    except Exception as e:
        fail_manifest(eval_dir, str(e))
        raise


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_evals(config_path)
