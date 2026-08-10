from __future__ import annotations
import os
from typing import Iterator, Literal

import boto3
from botocore import UNSIGNED
from botocore.config import Config

ModelName = Literal["opus", "sonnet", "haiku"]


def _key(model: ModelName) -> str:
    return {"opus": os.environ["OPUS_KEY"], "sonnet": os.environ["SONNET_KEY"], "haiku": os.environ["HAIKU_KEY"]}[model]


def _model_id(model: ModelName) -> str:
    return {"opus": os.environ["OPUS_MODEL"], "sonnet": os.environ["SONNET_MODEL"], "haiku": os.environ["HAIKU_MODEL"]}[model]


def _make_client(api_key: str):
    cfg = Config(signature_version=UNSIGNED)
    client = boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        endpoint_url=os.environ["LLM_ENDPOINT"],
        aws_access_key_id="placeholder",
        aws_secret_access_key="placeholder",
        config=cfg,
    )

    def inject(request, **_):
        request.headers["Authorization"] = f"Bearer {api_key}"
        request.headers.pop("X-Amz-Date", None)
        request.headers.pop("X-Amz-Security-Token", None)

    em = client._endpoint._event_emitter
    for evt in (
        "before-send.bedrock-runtime.Converse",
        "before-send.bedrock-runtime.ConverseStream",
        "before-send.bedrock-runtime.InvokeModel",
        "before-send.bedrock-runtime.InvokeModelWithResponseStream",
    ):
        em.register_first(evt, inject)
    return client


def _normalize(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        role = m["role"]
        if role == "system":
            continue
        content = m["content"]
        if isinstance(content, str):
            out.append({"role": role, "content": [{"text": content}]})
        else:
            out.append({"role": role, "content": content})
    return out


def chat(model: ModelName, messages: list[dict], *, system: str | None = None, max_tokens: int | None = None) -> str:
    client = _make_client(_key(model))
    kw: dict = {"modelId": _model_id(model), "messages": _normalize(messages)}
    if system:
        kw["system"] = [{"text": system}]
    if max_tokens is not None:
        kw["inferenceConfig"] = {"maxTokens": max_tokens}
    resp = client.converse(**kw)
    parts = resp.get("output", {}).get("message", {}).get("content", []) or []
    return "".join(p.get("text", "") for p in parts)


def chat_stream(model: ModelName, messages: list[dict], *, system: str | None = None, max_tokens: int | None = None) -> Iterator[str]:
    client = _make_client(_key(model))
    kw: dict = {"modelId": _model_id(model), "messages": _normalize(messages)}
    if system:
        kw["system"] = [{"text": system}]
    if max_tokens is not None:
        kw["inferenceConfig"] = {"maxTokens": max_tokens}
    resp = client.converse_stream(**kw)
    for ev in resp["stream"]:
        delta = ev.get("contentBlockDelta", {}).get("delta", {}).get("text")
        if delta:
            yield delta
