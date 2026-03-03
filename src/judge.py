"""
src/judge.py — LLM judge for deepeval evaluations

Provides JudgeLLM (DeepEvalBaseLLM subclass) and build_judge factory.

Provider credentials are resolved from:
  1. Environment variable  (env_var field in provider config)
  2. macOS Keychain        (keychain_service field in provider config)

Supported providers: nebius | together | azure
"""

import os
import platform
import subprocess

from deepeval.models.base_model import DeepEvalBaseLLM
from openai import AzureOpenAI, OpenAI


# ─── Credential resolution ────────────────────────────────────────────────────

def _get_api_key(env_var: str, keychain_service: str) -> str | None:
    """Resolve API key: env var first, then macOS Keychain."""
    key = os.environ.get(env_var)
    if key:
        return key
    if platform.system() == 'Darwin':
        try:
            result = subprocess.run(
                ['security', 'find-generic-password', '-s', keychain_service, '-w'],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
    return None


# ─── Judge LLM ───────────────────────────────────────────────────────────────

class JudgeLLM(DeepEvalBaseLLM):
    """DeepEvalBaseLLM wrapper for any OpenAI-compatible judge (Nebius, Together, Azure)."""

    def __init__(self, client, model_name: str):
        # Must set attributes before super().__init__() — it calls load_model() immediately
        self.client     = client
        self.model_name = model_name
        super().__init__(model=model_name)

    def load_model(self):
        return self.client

    def generate(self, prompt: str, *args, **kwargs) -> str:
        schema = kwargs.get('schema')
        extra  = {'response_format': {'type': 'json_object'}} if schema else {}
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{'role': 'user', 'content': prompt}],
            **extra,
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str, *args, **kwargs) -> str:
        return self.generate(prompt, *args, **kwargs)

    def get_model_name(self) -> str:
        return self.model_name


# ─── Factory ──────────────────────────────────────────────────────────────────

def build_judge(judge_cfg: dict, providers_cfg: dict) -> JudgeLLM:
    """
    Build a JudgeLLM from a judge entry and the providers dict (both from eval.json).

    judge_cfg fields:
        provider : "nebius" | "together" | "azure"
        model    : model / deployment name (overrides provider default_model)

    providers_cfg: the "providers" section of eval.json, keyed by provider name.
    """
    provider = judge_cfg.get('provider', 'nebius')
    prov_cfg = providers_cfg.get(provider)
    if not prov_cfg:
        raise ValueError(
            f"Provider '{provider}' not found in config's providers section. "
            f"Available: {list(providers_cfg.keys())}"
        )

    api_key = _get_api_key(
        env_var=prov_cfg['env_var'],
        keychain_service=prov_cfg['keychain_service'],
    )
    if not api_key:
        raise ValueError(
            f"{provider} API key not found. "
            f"Set {prov_cfg['env_var']} env var or add to Keychain:\n"
            f"  security add-generic-password -s {prov_cfg['keychain_service']} -w YOUR_KEY"
        )

    endpoint   = prov_cfg.get('endpoint')
    model_name = judge_cfg.get('model') or prov_cfg.get('default_model')

    if provider == 'azure':
        api_version = prov_cfg.get('api_version', '2024-02-15-preview')
        client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )
    else:
        client = OpenAI(api_key=api_key, base_url=endpoint)

    return JudgeLLM(client, model_name)
