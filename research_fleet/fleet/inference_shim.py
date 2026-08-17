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

TOOL USE + STRUCTURED OUTPUT
----------------------------
Two callers need more than plain text:

  1. an agentic research loop that needs Claude **tool-calling** (OpenAI
     `tools` + `tool_calls` round-trips), and
  2. NVIDIA NeMo Switchyard's `llm_classifier` router, which sends OpenAI
     chat-completion requests with a `response_format` JSON schema and expects
     the model to return schema-valid JSON.

Bedrock Converse has no first-class json_schema mode, so both ride the Converse
`toolConfig` mechanism:

  * OpenAI `tools`/`tool_choice` translate directly to Converse
    `toolConfig.tools[].toolSpec` + `toolChoice`. A `toolUse` block in the
    reply becomes OpenAI `tool_calls` (finish_reason "tool_calls").
  * OpenAI `response_format: json_schema` is emulated by declaring ONE forced
    tool whose `inputSchema.json` IS the requested schema and forcing it with
    `toolChoice={"tool":{"name":...}}`. The forced `toolUse.input` is then
    serialized back into `choices[0].message.content` as JSON *text* (NOT as
    tool_calls) so a client expecting JSON content — like Switchyard's
    classifier — gets valid JSON. `response_format: json_object` (no schema) is
    best-effort: we append a system instruction asking for a single JSON object
    and return the text unchanged.

A plain text request with no tools/response_format behaves EXACTLY as before.
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

# NOTE: read config with .get() (not os.environ[...]) so the module is importable
# for unit tests without a fully-populated env. main() validates LLM_ENDPOINT.
LLM_ENDPOINT = os.environ.get("LLM_ENDPOINT", "")
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


# --------------------------------------------------------------------------- #
# Translation helpers (pure functions — unit-tested without a server/network). #
# --------------------------------------------------------------------------- #

def _content_text(content) -> str:
    """Flatten an OpenAI message `content` (str | list of parts | None) to text."""
    if isinstance(content, list):  # OpenAI content-parts
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def _to_converse(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    """OpenAI chat messages -> Bedrock (messages, system).

    Handles plain text plus tool-use round-trips:
      * assistant messages carrying OpenAI `tool_calls` become a Converse
        assistant turn with `toolUse` blocks (preserving any assistant text).
      * `role:"tool"` messages become a Converse USER turn with a `toolResult`
        block; adjacent tool results are merged into a single user turn (Bedrock
        wants tool results grouped together in one user message).
    The convo is normalized to start with a user turn.
    """
    conv: list[dict] = []
    system: list[dict] = []

    def _append(role: str, block: dict):
        # Merge into the previous turn iff it shares the role AND the merge keeps
        # tool results grouped (only ever merge into an immediately-adjacent turn).
        if conv and conv[-1]["role"] == role:
            conv[-1]["content"].append(block)
        else:
            conv.append({"role": role, "content": [block]})

    for m in messages:
        role = m.get("role")
        if role == "system":
            system.append({"text": _content_text(m.get("content"))})
            continue

        if role == "tool":
            # OpenAI tool result -> Converse user turn with a toolResult block.
            tc_id = m.get("tool_call_id", "")
            result_text = _content_text(m.get("content"))
            block = {"toolResult": {
                "toolUseId": tc_id,
                "content": [{"text": result_text}],
                "status": "success",
            }}
            _append("user", block)
            continue

        if role == "assistant":
            tool_calls = m.get("tool_calls") or []
            text = _content_text(m.get("content"))
            blocks: list[dict] = []
            if text:
                blocks.append({"text": text})
            for tc in tool_calls:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                raw_args = fn.get("arguments", "")
                try:
                    parsed = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                except (ValueError, TypeError):
                    parsed = {}
                blocks.append({"toolUse": {
                    "toolUseId": tc.get("id", "") or f"tool-{uuid.uuid4().hex[:12]}",
                    "name": fn.get("name", ""),
                    "input": parsed,
                }})
            if not blocks:
                blocks = [{"text": ""}]
            # Assistant turns are not merged with a prior assistant turn (rare);
            # keep them as their own turn to preserve toolUse/text ordering.
            conv.append({"role": "assistant", "content": blocks})
            continue

        # default: user (and any unknown role) -> user text turn
        _append("user", {"text": _content_text(m.get("content"))})

    if not conv:
        conv = [{"role": "user", "content": [{"text": ""}]}]
    # Bedrock requires the convo to start with a user turn
    if conv[0]["role"] != "user":
        conv.insert(0, {"role": "user", "content": [{"text": "(continue)"}]})
    return conv, system


def _tools_to_converse(tools: list[dict]) -> list[dict]:
    """OpenAI `tools` -> Converse `toolConfig.tools` (list of {"toolSpec": ...})."""
    out: list[dict] = []
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        fn = t.get("function", t)  # tolerate a bare function object
        name = fn.get("name")
        if not name:
            continue
        spec = {"name": name, "inputSchema": {"json": fn.get("parameters", {}) or {}}}
        desc = fn.get("description")
        if desc:
            spec["description"] = desc
        out.append({"toolSpec": spec})
    return out


def _tool_choice_to_converse(tool_choice) -> dict | None:
    """OpenAI `tool_choice` -> Converse `toolChoice`.

    "auto" -> {"auto":{}}; "required" -> {"any":{}}; {"type":"function",...} ->
    {"tool":{"name":...}}. "none" returns None (caller drops tools entirely).
    A missing/unknown value defaults to auto.
    """
    if tool_choice in (None, "auto"):
        return {"auto": {}}
    if tool_choice == "none":
        return None
    if tool_choice == "required":
        return {"any": {}}
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function", {})
        name = fn.get("name")
        if name:
            return {"tool": {"name": name}}
    return {"auto": {}}


def _apply_response_format(response_format: dict) -> tuple[dict | None, str | None, dict | None]:
    """Translate an OpenAI `response_format` into a forced-tool structured-output plan.

    Returns (tool_config, forced_tool_name, extra_system):
      * json_schema -> (toolConfig with ONE forced tool whose inputSchema.json is
        the schema, toolChoice tool), forced_tool_name, None.
      * json_object -> (None, None, {"text": "Respond with ONLY ..."}) best-effort.
      * anything else / text -> (None, None, None).
    forced_tool_name is truthy ONLY for the json_schema path; the response
    translator uses it to route toolUse.input into message.content as JSON text.
    """
    if not isinstance(response_format, dict):
        return None, None, None
    rf_type = response_format.get("type")
    if rf_type == "json_schema":
        js = response_format.get("json_schema", {}) or {}
        schema = js.get("schema", js.get("json_schema", {})) or {}
        name = js.get("name") or "structured_output"
        # Bedrock tool names must be a safe token; sanitize just in case.
        name = "".join(c if (c.isalnum() or c in "_-") else "_" for c in str(name))[:64] or "structured_output"
        tool_config = {
            "tools": [{"toolSpec": {
                "name": name,
                "description": "Return the answer as a JSON object matching the schema.",
                "inputSchema": {"json": schema},
            }}],
            "toolChoice": {"tool": {"name": name}},
        }
        return tool_config, name, None
    if rf_type == "json_object":
        return None, None, {"text": "Respond with ONLY a single valid JSON object."}
    return None, None, None


def _extract_tool_uses(parts: list[dict]) -> tuple[str, list[dict]]:
    """Split Converse output content into (joined text, [toolUse dicts])."""
    text = "".join(p.get("text", "") for p in parts if isinstance(p, dict) and "text" in p)
    tool_uses = [p["toolUse"] for p in parts if isinstance(p, dict) and "toolUse" in p]
    return text, tool_uses


def _response_to_openai(resp: dict, model_label: str, *, forced_tool_name: str | None,
                        real_tools: bool) -> dict:
    """Bedrock Converse response -> OpenAI chat.completion dict.

    * forced_tool_name set (response_format json_schema): serialize the forced
      toolUse.input into message.content as JSON text, finish_reason "stop".
    * real_tools True and the model emitted toolUse: return OpenAI tool_calls,
      finish_reason "tool_calls".
    * otherwise: plain assistant text, finish_reason "stop" (unchanged).
    """
    parts = resp.get("output", {}).get("message", {}).get("content", []) or []
    text, tool_uses = _extract_tool_uses(parts)
    usage = resp.get("usage", {})

    message: dict
    finish_reason: str

    if forced_tool_name:
        # Structured output: put the forced tool's input into content as JSON text.
        forced = next((tu for tu in tool_uses if tu.get("name") == forced_tool_name), None)
        if forced is None and tool_uses:
            forced = tool_uses[0]
        payload = forced.get("input", {}) if forced else {}
        message = {"role": "assistant", "content": json.dumps(payload)}
        finish_reason = "stop"
    elif real_tools and tool_uses:
        # Real tool-calling requested by the caller's `tools`.
        tool_calls = [{
            "id": tu.get("toolUseId") or f"call_{uuid.uuid4().hex[:20]}",
            "type": "function",
            "function": {"name": tu.get("name", ""), "arguments": json.dumps(tu.get("input", {}))},
        } for tu in tool_uses]
        message = {"role": "assistant", "content": text or None, "tool_calls": tool_calls}
        finish_reason = "tool_calls"
    else:
        message = {"role": "assistant", "content": text}
        finish_reason = "stop"

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:20]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_label,
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": message}],
        "usage": {"prompt_tokens": usage.get("inputTokens", 0),
                  "completion_tokens": usage.get("outputTokens", 0),
                  "total_tokens": usage.get("totalTokens", 0)},
    }


def _build_converse_kwargs(payload: dict, model_id: str) -> tuple[dict, str | None, bool]:
    """Assemble converse(**kwargs) from an OpenAI request payload.

    Returns (kwargs, forced_tool_name, real_tools) where forced_tool_name is the
    response_format json_schema forced tool (or None) and real_tools indicates
    the caller passed genuine `tools` (so a toolUse must surface as tool_calls).
    """
    messages = payload.get("messages", [])
    conv, system = _to_converse(messages)

    real_tools = bool(payload.get("tools"))
    response_format = payload.get("response_format")
    rf_tool_config, forced_tool_name, extra_system = _apply_response_format(response_format)

    # Default max_tokens; give structured/tool responses more room unless the
    # caller set an explicit value.
    explicit_max = payload.get("max_tokens") or payload.get("max_completion_tokens")
    if explicit_max:
        max_tokens = int(explicit_max)
    elif real_tools or rf_tool_config or forced_tool_name:
        max_tokens = 4096
    else:
        max_tokens = 2048
    temperature = float(payload.get("temperature", 0.4))

    if extra_system:
        system = list(system) + [extra_system]

    kwargs: dict = {
        "modelId": model_id,
        "messages": conv,
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
    }
    if system:
        kwargs["system"] = system

    # toolConfig: real tools take precedence; else the forced structured-output tool.
    if real_tools:
        tool_cfg: dict = {"tools": _tools_to_converse(payload.get("tools", []))}
        choice = _tool_choice_to_converse(payload.get("tool_choice"))
        if choice is None:
            # tool_choice "none": expose no tools at all.
            tool_cfg = {}
        elif tool_cfg["tools"]:
            tool_cfg["toolChoice"] = choice
        if tool_cfg.get("tools"):
            kwargs["toolConfig"] = tool_cfg
        else:
            real_tools = False  # nothing declared -> behave as plain text
    elif rf_tool_config:
        kwargs["toolConfig"] = rf_tool_config

    return kwargs, forced_tool_name, real_tools


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, obj: dict):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_sse_completion(self, completion: dict):
        """Minimal single-chunk SSE fallback for `stream: true`.

        We do NOT translate ConverseStream token-by-token; Switchyard's classifier
        and our research loop both use non-streaming. Instead we emit the entire
        completion as one chat.completion.chunk delta, then a [DONE] sentinel, so
        stream-mode clients don't 500. Non-stream is the fully-correct default.
        """
        choice = completion["choices"][0]
        msg = choice["message"]
        delta = {"role": "assistant"}
        if msg.get("content") is not None:
            delta["content"] = msg["content"]
        if msg.get("tool_calls"):
            delta["tool_calls"] = [{
                "index": i,
                "id": tc["id"],
                "type": "function",
                "function": tc["function"],
            } for i, tc in enumerate(msg["tool_calls"])]
        chunk = {
            "id": completion["id"],
            "object": "chat.completion.chunk",
            "created": completion["created"],
            "model": completion["model"],
            "choices": [{"index": 0, "delta": delta, "finish_reason": choice["finish_reason"]}],
        }
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

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
            self._send(400, {"error": {"message": f"bad json: {e}", "type": "invalid_request_error"}})
            return
        if not isinstance(payload, dict):
            self._send(400, {"error": {"message": "body must be a JSON object",
                                       "type": "invalid_request_error"}})
            return

        model_req = payload.get("model", "")
        model_id, jwt = _resolve(model_req)
        if not model_id or not jwt:
            self._send(500, {"error": {"message": f"model '{model_req}' not configured in shim",
                                       "type": "invalid_request_error"}})
            return

        stream = bool(payload.get("stream"))
        try:
            kwargs, forced_tool_name, real_tools = _build_converse_kwargs(payload, model_id)
        except Exception as e:  # malformed tools/messages/response_format
            self._send(400, {"error": {"message": f"bad request: {e}",
                                       "type": "invalid_request_error"}})
            return

        try:
            resp = _client(jwt).converse(**kwargs)
            completion = _response_to_openai(resp, model_req or model_id,
                                             forced_tool_name=forced_tool_name,
                                             real_tools=real_tools)
        except Exception as e:
            self._send(502, {"error": {"message": f"bedrock converse failed: {e}",
                                       "type": "api_error"}})
            return

        if stream:
            try:
                self._send_sse_completion(completion)
            except Exception:
                pass
            return
        self._send(200, completion)


def main():
    if not LLM_ENDPOINT:
        raise SystemExit("LLM_ENDPOINT is required (set it in the environment before starting the shim)")
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"inference-shim listening on 0.0.0.0:{PORT} -> {LLM_ENDPOINT}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
