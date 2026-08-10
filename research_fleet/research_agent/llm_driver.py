"""LLM driver for the research agents — the reasoning brain.

The agent's *thinking* (forming hypotheses, interpreting backtest results,
writing findings) is done by Claude via the code-factory's SingleStore-hosted
Anthropic gateway: AWS Bedrock ``Converse`` against a custom ``LLM_ENDPOINT``
with an UNSIGNED boto3 client + ``Authorization: Bearer <JWT>`` header. This
mirrors the demo backend's ``llm.py`` exactly (a known-good path).

This is deliberately a thin, swappable driver. On the EC2 fleet the research
agent is launched *through OpenClaw-in-NemoClaw*; where NemoClaw can route
OpenClaw's inference at our endpoint it does so, and this driver is the same
Anthropic-Messages-shaped brain used for structured sub-calls (hypothesis JSON,
finding text). Keeping it isolated means the inference path is defined in ONE
place.
"""

from __future__ import annotations

import json
import os
from typing import Any

import boto3
from botocore import UNSIGNED
from botocore.config import Config

LLM_ENDPOINT = os.environ.get("LLM_ENDPOINT", "")
MODELS = {
    "opus": (os.environ.get("OPUS_MODEL", ""), os.environ.get("OPUS_KEY", "")),
    "sonnet": (os.environ.get("SONNET_MODEL", ""), os.environ.get("SONNET_KEY", "")),
    "haiku": (os.environ.get("HAIKU_MODEL", ""), os.environ.get("HAIKU_KEY", "")),
}


def _client(api_key: str):
    client = boto3.client(
        "bedrock-runtime", region_name="us-east-1", endpoint_url=LLM_ENDPOINT,
        aws_access_key_id="placeholder", aws_secret_access_key="placeholder",
        config=Config(signature_version=UNSIGNED, retries={"max_attempts": 3}),
    )

    def inject(request, **_):
        request.headers["Authorization"] = f"Bearer {api_key}"
        request.headers.pop("X-Amz-Date", None)
        request.headers.pop("X-Amz-Security-Token", None)

    em = client._endpoint._event_emitter
    for evt in ("before-send.bedrock-runtime.Converse",
                "before-send.bedrock-runtime.ConverseStream"):
        em.register_first(evt, inject)
    return client


def complete(prompt: str, *, model: str = "sonnet", system: str | None = None,
             max_tokens: int = 1500, temperature: float = 0.4) -> str:
    """One-shot completion. Returns assistant text."""
    model_id, key = MODELS.get(model, MODELS["sonnet"])
    if not model_id or not key or not LLM_ENDPOINT:
        raise RuntimeError(f"LLM not configured for model '{model}'")
    client = _client(key)
    kwargs: dict[str, Any] = {
        "modelId": model_id,
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
    }
    if system:
        kwargs["system"] = [{"text": system}]
    resp = client.converse(**kwargs)
    parts = resp["output"]["message"]["content"]
    return "".join(p.get("text", "") for p in parts).strip()


def complete_json(prompt: str, *, model: str = "sonnet", system: str | None = None,
                  max_tokens: int = 1500) -> dict:
    """Completion that must return a JSON object; robustly parses it."""
    sys_json = (system or "") + "\nRespond with ONLY a single valid JSON object, no prose, no markdown fences."
    txt = complete(prompt, model=model, system=sys_json.strip(), max_tokens=max_tokens, temperature=0.2)
    return _extract_json(txt)


def _extract_json(txt: str) -> dict:
    txt = txt.strip()
    if txt.startswith("```"):
        txt = txt.split("```", 2)[1] if "```" in txt[3:] else txt.strip("`")
        if txt.startswith("json"):
            txt = txt[4:]
    start, end = txt.find("{"), txt.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(txt[start:end + 1])
        except Exception:
            pass
    return {}
