from __future__ import annotations
import os
from openai import OpenAI


def _client() -> OpenAI:
    return OpenAI(api_key=os.environ["QWEN_KEY"], base_url=os.environ["LLM_ENDPOINT"])


def embed(texts: list[str] | str) -> list[list[float]]:
    if isinstance(texts, str):
        texts = [texts]
    resp = _client().embeddings.create(model=os.environ["EMBEDDING_MODEL"], input=texts)
    items = sorted(resp.data, key=lambda d: getattr(d, "index", 0))
    return [list(it.embedding) for it in items]
