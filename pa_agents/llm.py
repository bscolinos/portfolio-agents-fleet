"""SingleStore-hosted Claude client (Bedrock Converse, unsigned + JWT bearer).

Used for the agents' *reflections*: after each rebalance an agent asks Claude to
distill what happened into a natural-language learning it will re-read (via
semantic memory recall) on its next run. Mirrors the factory's llm.py pattern.
"""

from __future__ import annotations

import boto3
from botocore.config import Config
from botocore import UNSIGNED

from .config import settings


def _client(jwt: str):
    session = boto3.Session()
    client = session.client(
        "bedrock-runtime",
        region_name="us-east-1",
        endpoint_url=settings.llm_endpoint,
        config=Config(signature_version=UNSIGNED, retries={"max_attempts": 2}),
    )

    def _inject_auth(request, **_kwargs):
        request.headers["Authorization"] = f"Bearer {jwt}"

    client.meta.events.register_first("before-send.bedrock-runtime.*", _inject_auth)
    return client


def reflect(prompt: str, *, model: str | None = None, jwt: str | None = None,
            max_tokens: int = 400) -> str:
    """Single-shot completion for agent reflections/summaries."""
    model = model or settings.haiku_model
    jwt = jwt or settings.haiku_key
    if not model or not jwt or not settings.llm_endpoint:
        return ""  # LLM optional; agents still run without reflections
    try:
        client = _client(jwt)
        resp = client.converse(
            modelId=model,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0.3},
        )
        parts = resp["output"]["message"]["content"]
        return "".join(p.get("text", "") for p in parts).strip()
    except Exception:
        return ""
