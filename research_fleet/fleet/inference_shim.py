"""OpenAI-compatible -> SingleStore Bedrock-Converse inference shim.

Why this exists: NemoClaw/OpenClaw can be onboarded with a `custom` provider that
speaks the OpenAI Chat Completions API against an arbitrary base URL with a
Bearer key. But our Anthropic models are served by a SingleStore gateway that
speaks AWS **Bedrock Converse** (unsigned boto3 client + `Authorization: Bearer
<JWT>`), NOT the OpenAI or public Anthropic wire format, and NemoClaw's built-in
Bedrock adapter is hard-gated to real AWS hostnames. So this tiny local shim:

    OpenClaw --(OpenAI /v1/chat/completions, Bearer JWT)--> this shim
    this shim --(Bedrock Converse, unsigned + Bearer JWT)--> SingleStore gateway

It runs on the EC2 host, reachable from inside the OpenShell sandbox at
`host.openshell.internal:11500`. Model routing: the OpenAI `model` field is
mapped to a SingleStore Claude model id (opus/sonnet/haiku aliases or a raw id).

Deliberately dependency-light: stdlib http.server + boto3 (already needed).
Auth: the shim requires the SAME bearer token OpenClaw sends to equal one of the
configured model JWTs, so a stray localhost caller can't use it blindly.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import boto3
from botocore import UNSIGNED
from botocore.config import Config

LLM_ENDPOINT = os.environ["LLM_ENDPOINT"]
PORT = int(os.environ.get("SHIM_PORT", "11500"))

# alias -> (model_id, jwt). Populated from env the userdata writes.
MODELS = {
    "opus": (os.environ.get("OPUS_MODEL", ""), os.environ.get("OPUS_KEY", "")),
    "sonnet": (os.environ.get("SONNET_MODEL", ""), os.environ.get("SONNET_KEY", "")),
    "haiku": (os.environ.get("HAIKU_MODEL", ""), os.environ.get("HAIKU_KEY", "")),
}
# also allow addressing a model by its raw id
_BY_ID = {mid: (mid, key) for (mid, key) in MODELS.values() if mid}
DEFAULT_MODEL = os.environ.get("SHIM_DEFAULT_MODEL", "sonnet")


def _resolve(model: str) -> tuple[str, str]:
    if not model:
        return MODELS[DEFAULT_MODEL]
    if model in MODELS:
        return MODELS[model]
    if model in _BY_ID:
        return _BY_ID[model]
    # substring match (e.g. "claude-3-5-sonnet" -> our sonnet)
    low = model.lower()
    for alias in ("opus", "sonnet", "haiku"):
        if alias in low and MODELS[alias][0]:
            return MODELS[alias]
    return MODELS[DEFAULT_MODEL]


def _client(jwt: str):
    c = boto3.client("bedrock-runtime", region_name="us-east-1", endpoint_url=LLM_ENDPOINT,
                     aws_access_key_id="x", aws_secret_access_key="x",
                     config=Config(signature_version=UNSIGNED, retries={"max_attempts": 3}))

    def inject(request, **_):
        request.headers["Authorization"] = f"Bearer {jwt}"
        request.headers.pop("X-Amz-Date", None)
        request.headers.pop("X-Amz-Security-Token", None)

    em = c._endpoint._event_emitter
    for evt in ("before-send.bedrock-runtime.Converse", "before-send.bedrock-runtime.ConverseStream"):
        em.register_first(evt, inject)
    return c


def _to_converse(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    """OpenAI chat messages -> Bedrock (messages, system)."""
    conv, system = [], []
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if isinstance(content, list):  # OpenAI content-parts
            text = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        else:
            text = content or ""
        if role == "system":
            system.append({"text": text})
        elif role in ("user", "assistant"):
            conv.append({"role": role, "content": [{"text": text}]})
    if not conv:
        conv = [{"role": "user", "content": [{"text": ""}]}]
    # Bedrock requires the convo to start with a user turn
    if conv[0]["role"] != "user":
        conv.insert(0, {"role": "user", "content": [{"text": "(continue)"}]})
    return conv, system


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, obj: dict):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            self._send(200, {"object": "list", "data": [
                {"id": a, "object": "model", "owned_by": "singlestore"} for a in MODELS if MODELS[a][0]]})
        elif self.path.rstrip("/") in ("/health", "/healthz", ""):
            self._send(200, {"ok": True})
        else:
            self._send(404, {"error": {"message": "not found"}})

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send(404, {"error": {"message": "only /v1/chat/completions"}})
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            self._send(400, {"error": {"message": f"bad json: {e}"}})
            return

        model_req = payload.get("model", "")
        model_id, jwt = _resolve(model_req)
        if not model_id or not jwt:
            self._send(500, {"error": {"message": f"model '{model_req}' not configured in shim"}})
            return
        messages = payload.get("messages", [])
        max_tokens = int(payload.get("max_tokens") or payload.get("max_completion_tokens") or 2048)
        temperature = float(payload.get("temperature", 0.4))
        conv, system = _to_converse(messages)
        kwargs = {"modelId": model_id, "messages": conv,
                  "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature}}
        if system:
            kwargs["system"] = system
        try:
            resp = _client(jwt).converse(**kwargs)
            parts = resp["output"]["message"]["content"]
            text = "".join(p.get("text", "") for p in parts)
            usage = resp.get("usage", {})
        except Exception as e:
            self._send(502, {"error": {"message": f"bedrock converse failed: {e}"}})
            return

        self._send(200, {
            "id": f"chatcmpl-{uuid.uuid4().hex[:20]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_req or model_id,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": usage.get("inputTokens", 0),
                      "completion_tokens": usage.get("outputTokens", 0),
                      "total_tokens": usage.get("totalTokens", 0)},
        })


def main():
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"inference-shim listening on 0.0.0.0:{PORT} -> {LLM_ENDPOINT}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
