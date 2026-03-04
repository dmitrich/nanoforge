"""
src/evals.py — deepeval evaluation runner for nanoforge GPT models

Usage:
    python src/evals.py configs/eval.json   # runs all judges defined in eval.json
    python src/evals.py                     # latest training run + built-in defaults

Each run produces:
    runs/evals/<eval_id>/
        manifest.json          — lineage, status, config hash
        resolved_eval.json     — fully resolved config stamped at runtime
        generations.jsonl      — raw inference output per test case (shared across judges)
        <provider>/
            results.jsonl      — per-case metric scores + judge reasoning
            summary.json       — aggregate pass/fail and per-metric stats

Judges and providers are configured in configs/eval.json:
    "providers" : credentials and endpoints for each provider
    "judges"    : list of {provider, model} entries to run
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

sys.path.insert(0, str(Path(__file__).parent))

from judge import build_judge
from manifest import generate_run_id, create_manifest, complete_manifest, fail_manifest
from model import GPT, resolve_checkpoint_path
from observability import (
    EVAL_ROOT_SPAN_NAME,
    EVAL_SAMPLE_SPAN_NAME,
    GENERATION_SPAN_NAME,
    JUDGE_EVAL_SPAN_NAME,
    build_common_tags,
    build_common_trace_metadata,
    flush_langfuse,
    get_langfuse_client,
)
from tokenizer import Tokenizer
from utils import append_jsonl, get_device, write_json


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
    ckpt_path = resolve_checkpoint_path(ckpt_dir, ckpt_key)

    return raw, ckpt_path, run_id


# ─── Comparison table ─────────────────────────────────────────────────────────

_PROVIDER_LABELS = {
    "nebius":   "Nebius",
    "together": "Together AI",
    "azure":    "Azure",
}

def _judge_label(judge_cfg: dict) -> str:
    provider = judge_cfg.get("provider", "nebius")
    return _PROVIDER_LABELS.get(provider, provider.title())


def _score_name(metric_name: str, judge_cfg: dict) -> str:
    return f"{metric_name} [{_judge_label(judge_cfg)}]"


def _box_table(headers: list, rows: list) -> str:
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def _divider(left, mid, right):
        parts = ["─" * (w + 2) for w in col_widths]
        return left + mid.join(parts) + right

    def _data_row(cells):
        parts = [f" {cell.ljust(col_widths[i])} " for i, cell in enumerate(cells)]
        return "│" + "│".join(parts) + "│"

    def _header_row(cells):
        parts = [f" {cell.center(col_widths[i])} " for i, cell in enumerate(cells)]
        return "│" + "│".join(parts) + "│"

    lines = [_divider("┌", "┬", "┐"), _header_row(headers), _divider("├", "┼", "┤")]
    for i, row in enumerate(rows):
        lines.append(_data_row(row))
        if i < len(rows) - 1:
            lines.append(_divider("├", "┼", "┤"))
    lines.append(_divider("└", "┴", "┘"))
    return "\n".join(lines)


def _print_comparison(results: list):
    labels = [r["judge_label"] for r in results]

    def _short_model(name):
        base = name.split("/")[-1]
        for suffix in ["-Instruct-Turbo", "-Instruct", "-Turbo"]:
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        return base

    model_names  = [r.get("judge_model", "") for r in results]
    short_names  = [_short_model(m) for m in model_names if m]
    unique_names = list(dict.fromkeys(short_names))
    if len(unique_names) == 1:
        title = f"Judge comparison: {unique_names[0]} on {' vs '.join(labels)}"
    else:
        title = f"Judge comparison: {' vs '.join(labels)}"

    print(f"\n{title}")

    metric_names = list(results[0]["metrics_summary"].keys())
    headers = ["Metric"] + [f"{lbl} (mean / pass%)" for lbl in labels]

    judge_row = ["Judge model"]
    for r in results:
        model = r.get("judge_model", "")
        judge_row.append(model.split("/")[-1] if model else "unknown")

    rows = [judge_row]
    for metric in metric_names:
        row = [metric]
        for r in results:
            ms = r["metrics_summary"].get(metric, {})
            row.append(f"{ms.get('mean', 0):.2f} / {ms.get('pass_rate', 0):.0%}")
        rows.append(row)

    overall_row = ["Overall pass rate"]
    for r in results:
        n, t = r["passed"], r["total"]
        overall_row.append(f"{n}/{t} ({round(n / t * 100) if t else 0}%)")
    rows.append(overall_row)

    print(_box_table(headers, rows))

    pass_counts = [r["passed"] for r in results]
    totals      = [r["total"]  for r in results]
    if len(set(pass_counts)) == 1:
        n, t = pass_counts[0], totals[0]
        pct  = round(n / t * 100) if t else 0
        word = "Both judges agree" if len(results) == 2 else "All judges agree"
        print(f"{word} on the headline: {n}/{t} stories pass ({pct}%)")
    else:
        parts = [f"{lbl}: {r['passed']}/{r['total']}" for lbl, r in zip(labels, results)]
        print(f"Judges disagree on the headline — {', '.join(parts)}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_evals(config_path: str = None):
    if config_path is None:
        raw = {
            "meta":      {"eval_name": "eval"},
            "source":    {"run_id": None, "checkpoint": "best"},
            "model":     {"$from_run": True},
            "tokenizer": {"$from_run": True},
            "generation": {"max_new_tokens": 200, "temperature": 0.8, "top_k": 40},
            "providers": {
                "nebius": {
                    "endpoint":          "https://api.tokenfactory.nebius.com/v1/",
                    "keychain_service":  "nebius-api-key",
                    "env_var":           "NEBIUS_API_KEY",
                    "default_model":     "meta-llama/Llama-3.3-70B-Instruct",
                },
            },
            "judges": [
                {"provider": "nebius", "model": "meta-llama/Llama-3.3-70B-Instruct"},
            ],
            "test_cases_file": "evals.json",
            "metrics": [
                {"name": "Coherence", "criteria": "The story flows logically and is internally consistent.", "threshold": 0.5},
                {"name": "Fluency",   "criteria": "The text is grammatically correct and reads naturally.",  "threshold": 0.5},
            ],
        }
    else:
        with open(config_path) as f:
            raw = json.load(f)

    raw, ckpt_path, run_id = _resolve_run_refs(raw)

    eval_name       = raw.get("meta", {}).get("eval_name", "eval")
    providers_cfg   = raw.get("providers", {})
    judges_cfg      = raw.get("judges", [])
    metrics_cfg     = raw.get("metrics", [])
    generation      = raw.get("generation", {})
    model_cfg       = raw["model"]

    # Resolve test_cases_file relative to this script's directory (src/) if not absolute
    tcf = raw.get("test_cases_file", "evals.json")
    test_cases_path = Path(tcf) if Path(tcf).is_absolute() else Path(__file__).parent / tcf

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
    langfuse = get_langfuse_client()
    eval_trace = langfuse.trace(
        id=eval_id,
        name=EVAL_ROOT_SPAN_NAME,
        session_id=run_id,
        input={"config_path": config_path, "resolved_eval": resolved_eval},
        metadata=build_common_trace_metadata(
            experiment_id=run_id,
            run_type="evaluation",
            source_run_id=run_id,
            model_checkpoint=str(ckpt_path),
            extra={
                "eval.run_id": eval_id,
                "test_cases_file": str(test_cases_path),
                "judge.count": len(judges_cfg),
                "metrics": [m["name"] for m in metrics_cfg],
            },
        ),
        tags=build_common_tags("evaluation", run_id, eval_name),
    )
    eval_span = langfuse.span(
        trace_id=eval_trace.id,
        name=EVAL_ROOT_SPAN_NAME,
        input={"checkpoint": str(ckpt_path), "test_cases_file": str(test_cases_path)},
        metadata={
            "source_run_id": run_id,
            "judge_count": len(judges_cfg),
            "metric_names": [m["name"] for m in metrics_cfg],
        },
    )

    print(f"Eval ID:    {eval_id}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Tests file: {test_cases_path}")
    print(f"Judges:     {[j['provider'] for j in judges_cfg]}")
    print(f"Metrics:    {[m['name'] for m in metrics_cfg]}")

    try:
        # ── Inference (once, shared across all judges) ─────────────────────────
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

        with open(test_cases_path) as f:
            test_cases_raw = json.load(f)["test_cases"]

        print(f"\n--- Inference ({len(test_cases_raw)} test cases) ---")
        generations = []
        for tc in test_cases_raw:
            prompt = tc["input"]
            sample_span = langfuse.span(
                trace_id=eval_trace.id,
                parent_observation_id=eval_span.id,
                name=EVAL_SAMPLE_SPAN_NAME,
                input={"test_id": tc["id"], "prompt": prompt},
            )
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
            generation = langfuse.generation(
                trace_id=eval_trace.id,
                parent_observation_id=sample_span.id,
                name=GENERATION_SPAN_NAME,
                model=f"nanoforge:{run_id}",
                model_parameters={
                    "temperature": temperature,
                    "top_k": top_k,
                    "max_new_tokens": max_new,
                },
                input=prompt,
                output=text,
                metadata={
                    "test_id": tc["id"],
                    "checkpoint": str(ckpt_path),
                    "source_run_id": run_id,
                    "stop_tokens": stop_strings,
                },
            )
            generation.end()
            sample_span.end(output={"generated_chars": len(text)})
            entry = {"test_id": tc["id"], "input": prompt, "actual_output": text}
            entry["langfuse_generation_id"] = generation.id
            entry["langfuse_trace_id"] = generation.trace_id
            generations.append(entry)
            append_jsonl(eval_dir / "generations.jsonl", entry)
            print(f"  [{tc['id']}] {prompt!r} → {len(text)} chars")

        # ── Score with each judge ──────────────────────────────────────────────
        all_judge_results = []

        for judge_cfg in judges_cfg:
            provider = judge_cfg["provider"]
            print(f"\n--- Judge: {provider} ---")

            judge = build_judge(judge_cfg, providers_cfg)
            print(f"Model: {judge.get_model_name()}")

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

            judge_dir = eval_dir / provider
            judge_dir.mkdir(exist_ok=True)

            print(f"Evaluating ({len(generations)} × {len(metrics)} metrics)")
            all_results = []
            for gen in generations:
                judge_span = langfuse.span(
                    trace_id=eval_trace.id,
                    parent_observation_id=gen["langfuse_generation_id"],
                    name=JUDGE_EVAL_SPAN_NAME,
                    input={
                        "provider": provider,
                        "judge_model": judge.get_model_name(),
                        "test_id": gen["test_id"],
                    },
                    metadata={
                        "judge.provider": provider,
                        "judge.label": _judge_label(judge_cfg),
                        "judge.model": judge.get_model_name(),
                        "metric_names": [m["name"] for m in metrics_cfg],
                    },
                )
                test_case    = LLMTestCase(input=gen["input"], actual_output=gen["actual_output"])
                case_metrics = {}
                for metric in metrics:
                    metric.measure(test_case)
                    case_metrics[metric.name] = {
                        "score":  round(metric.score, 4),
                        "reason": metric.reason,
                        "passed": metric.is_successful(),
                    }
                    langfuse.score(
                        trace_id=gen["langfuse_trace_id"],
                        observation_id=gen["langfuse_generation_id"],
                        name=_score_name(metric.name, judge_cfg),
                        value=round(metric.score, 4),
                        comment=f"{judge.get_model_name()}: {metric.reason}",
                    )

                result = {
                    "test_id":       gen["test_id"],
                    "input":         gen["input"],
                    "actual_output": gen["actual_output"],
                    "metrics":       case_metrics,
                    "passed":        all(v["passed"] for v in case_metrics.values()),
                }
                all_results.append(result)
                append_jsonl(judge_dir / "results.jsonl", result)
                judge_span.end(output=result["metrics"])

                status = "PASS" if result["passed"] else "FAIL"
                print(f"  [{gen['test_id']}] {status}")
                for name, s in case_metrics.items():
                    print(f"    {name}: {s['score']:.2f}  {s['reason'][:80]}")

            # Per-judge summary
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

            judge_summary = {
                "eval_id":         eval_id,
                "provider":        provider,
                "model":           judge.get_model_name(),
                "checkpoint":      ckpt_path,
                "total":           n_total,
                "passed":          n_passed,
                "failed":          n_total - n_passed,
                "pass_rate":       round(n_passed / n_total, 3) if n_total else 0.0,
                "metrics_summary": metrics_summary,
            }
            write_json(judge_dir / "summary.json", judge_summary)

            print(f"  Pass rate: {n_passed}/{n_total} ({judge_summary['pass_rate']:.0%})")

            all_judge_results.append({
                "eval_id":         eval_id,
                "judge_label":     _judge_label(judge_cfg),
                "judge_model":     judge.get_model_name(),
                "total":           n_total,
                "passed":          n_passed,
                "metrics_summary": metrics_summary,
            })

        # ── Top-level manifest / comparison summary ────────────────────────────
        comparison = {
            "eval_id":    eval_id,
            "checkpoint": ckpt_path,
            "judges":     [
                {"provider": r["judge_label"], "model": r["judge_model"],
                 "pass_rate": round(r["passed"] / r["total"], 3) if r["total"] else 0.0}
                for r in all_judge_results
            ],
        }
        write_json(eval_dir / "summary.json", comparison)
        eval_span.end(output=comparison)
        eval_trace.update(output=comparison)
        flush_langfuse()
        complete_manifest(eval_dir, comparison)

        print(f"\n=== Eval complete: {eval_id} ===")
        print(f"Output: {eval_dir}/")

        if len(all_judge_results) > 1:
            _print_comparison(all_judge_results)
        else:
            r = all_judge_results[0]
            print(f"Pass rate: {r['passed']}/{r['total']} ({r['passed']/r['total']:.0%})")
            for name, s in r["metrics_summary"].items():
                print(f"  {name}: mean={s['mean']:.2f}  pass_rate={s['pass_rate']:.0%}")

    except Exception as e:
        eval_span.end(
            level="ERROR",
            status_message=str(e),
            output={"error": str(e)},
        )
        eval_trace.update(output={"error": str(e)})
        flush_langfuse()
        fail_manifest(eval_dir, str(e))
        raise


if __name__ == "__main__":
    import os
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    # Resolve config path to absolute BEFORE changing directory
    if config_path:
        config_path = str(Path(config_path).resolve())
    # Always run relative to project root regardless of where script is invoked from
    os.chdir(Path(__file__).parent.parent)
    run_evals(config_path)
