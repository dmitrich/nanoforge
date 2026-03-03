"""
src/judge.py — LLM judge for deepeval evaluations

Provides JudgeLLM (DeepEvalBaseLLM subclass) and _build_judge factory.

Provider configs live in configs/{provider}_config.json.
Credentials are resolved from:
  1. Environment variable  (env_var field in provider config)
  2. macOS Keychain        (keychain_service field in provider config)

Supported providers: nebius | together | azure
"""

import json
import os
import platform
import subprocess
from pathlib import Path

from deepeval.models.base_model import DeepEvalBaseLLM
from openai import AzureOpenAI, OpenAI

_CONFIGS_DIR = Path(__file__).parent.parent / 'configs'


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


def _load_provider_config(provider: str) -> dict:
    path = _CONFIGS_DIR / f'{provider}_config.json'
    if not path.exists():
        raise FileNotFoundError(
            f"Provider config not found: {path}\n"
            f"Expected one of: nebius_config.json, together_config.json, azure_config.json"
        )
    with open(path) as f:
        return json.load(f)


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

def build_judge(judge_cfg: dict) -> JudgeLLM:
    """
    Build a JudgeLLM from an eval config's "judge" section.

    Provider defaults come from configs/{provider}_config.json.
    API keys are resolved from env var, then macOS Keychain.

    judge_cfg fields (all optional — fall back to provider config):
        provider    : "nebius" | "together" | "azure"
        endpoint    : API base URL
        model       : model / deployment name
        api_version : Azure only
    """
    provider = judge_cfg.get('provider', 'nebius')
    prov_cfg = _load_provider_config(provider)

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

    endpoint   = judge_cfg.get('endpoint') or prov_cfg.get('endpoint')
    model_name = judge_cfg.get('model')    or prov_cfg.get('default_model')

    if provider == 'azure':
        api_version = judge_cfg.get('api_version') or prov_cfg.get('api_version', '2024-02-15-preview')
        client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )
    else:
        client = OpenAI(api_key=api_key, base_url=endpoint)

    return JudgeLLM(client, model_name)
