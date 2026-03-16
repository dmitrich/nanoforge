"""
src/observability.py — Langfuse observability integration for nanoforge

Provides a Langfuse client with graceful fallback to a no-op stub when
credentials are not configured. All downstream code calls the same API
regardless of whether Langfuse is active.
"""

import os

# ─── Span / event name constants ──────────────────────────────────────────────

CHECKPOINT_EVENT_NAME     = "checkpoint"
TRAINING_EVAL_SPAN_NAME   = "training_eval"
TRAINING_ROOT_SPAN_NAME   = "training"
EVAL_ROOT_SPAN_NAME       = "evaluation"
EVAL_SAMPLE_SPAN_NAME     = "eval_sample"
GENERATION_SPAN_NAME      = "generation"
JUDGE_EVAL_SPAN_NAME      = "judge_eval"


# ─── No-op stub ───────────────────────────────────────────────────────────────

class _NoOpObservation:
    """Returned by all _NoOpClient methods. Silently ignores all operations."""

    id       = "noop"
    trace_id = "noop"

    def end(self, **kwargs):
        return self

    def update(self, **kwargs):
        return self


class _NoOpClient:
    """Silently replaces the real Langfuse client when credentials are absent."""

    def trace(self, **kwargs) -> _NoOpObservation:
        return _NoOpObservation()

    def span(self, **kwargs) -> _NoOpObservation:
        return _NoOpObservation()

    def generation(self, **kwargs) -> _NoOpObservation:
        return _NoOpObservation()

    def score(self, **kwargs) -> None:
        pass

    def flush(self) -> None:
        pass


# ─── Module-level singleton ────────────────────────────────────────────────────

_client = None


def get_langfuse_client(enabled: bool = True):
    """
    Return the module-level Langfuse client, initialising it on first call.

    If `enabled` is False, returns a no-op stub immediately without checking
    credentials. Otherwise, if LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, and
    LANGFUSE_HOST are all set, a real Langfuse client is returned. Falls back
    to a silent no-op stub when credentials are absent or `enabled` is False.
    """
    global _client
    if not enabled:
        return _NoOpClient()
    if _client is not None:
        return _client

    required = ("LANGFUSE_SECRET_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_HOST")
    if all(os.environ.get(k) for k in required):
        try:
            from langfuse import Langfuse
            _client = Langfuse()
            return _client
        except Exception:
            pass

    _client = _NoOpClient()
    return _client


def flush_langfuse() -> None:
    """Flush any queued Langfuse events. No-op if the client is a stub."""
    if _client is not None:
        _client.flush()


# ─── Metadata helpers ─────────────────────────────────────────────────────────

def build_common_trace_metadata(
    experiment_id: str,
    run_type: str,
    source_run_id: str,
    extra: dict = None,
    model_checkpoint=None,
) -> dict:
    """Return a flat metadata dict stamped onto every trace."""
    meta = {
        "experiment_id":  experiment_id,
        "run_type":       run_type,
        "source_run_id":  source_run_id,
    }
    if model_checkpoint is not None:
        meta["model_checkpoint"] = str(model_checkpoint)
    if extra:
        meta.update(extra)
    return meta


def build_common_tags(run_type: str, run_id: str, run_name: str) -> list:
    """Return a list of tag strings attached to every trace."""
    return [t for t in [run_type, run_id, run_name] if t]
