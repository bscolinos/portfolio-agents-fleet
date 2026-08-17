"""Switchyard transport — a Bedrock-``Converse``-shaped adapter over Switchyard.

The agentic research loop (:mod:`agentic_loop`) drives a Bedrock ``Converse``
tool-use conversation against a boto3 ``bedrock-runtime`` client. This module
provides a DROP-IN, duck-compatible stand-in for that client whose sole public
surface the loop uses is ``.converse(modelId=, messages=, toolConfig=, system=,
inferenceConfig=) -> dict`` returning a Bedrock-Converse-shaped response dict.

Instead of talking Bedrock, it talks to the **NVIDIA NeMo Switchyard** router at
its OpenAI Chat Completions endpoint. Switchyard classifies each request's
complexity and routes it to a model TIER (haiku/sonnet/opus) via the local
inference shim (see ``fleet/routes.toml`` + ``fleet/inference_shim.py``). So the
loop keeps its exact Converse-handling logic while gaining per-turn tier routing.

Wire path::

    agentic_loop --(Converse-shaped .converse(...))--> SwitchyardTransport
    SwitchyardTransport --(OpenAI /v1/chat/completions, model="research")--> Switchyard
    Switchyard --(classify -> pick tier)--> inference shim --> SingleStore gateway

Design notes:

  * Deliberately dependency-light: POSTs with stdlib :mod:`urllib` (mirrors the
    shim's stdlib-only style), so this path works even where ``requests`` is not
    guaranteed. Timeout ~120s.
  * The ``modelId`` the loop passes is IGNORED for routing (Switchyard decides the
    tier); it is accepted for signature-compatibility only. The OpenAI ``model``
    field sent to Switchyard is the ROUTE id (``SWITCHYARD_ROUTE``, default
    "research").
  * Tool-use round-trips are translated in BOTH directions so ids stay stable
    across turns: a Converse ``toolUse`` becomes an OpenAI assistant ``tool_call``
    (same id), and the loop's next-turn ``toolResult`` (keyed by that id) becomes
    an OpenAI ``role:"tool"`` message with a matching ``tool_call_id``.
  * The chosen tier (the OpenAI response ``model`` field, e.g. "haiku-fast",
    "sonnet", "opus") is stashed on the returned dict under ``_switchyard_model``
    so the loop can log which tier handled each turn — the whole point of routing.
  * On HTTP/transport error this RAISES; the loop's ``_converse`` retry/try-except
    handles it exactly as it does a Bedrock failure.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
import uuid
from typing import Any, Callable

# Where Switchyard's OpenAI Chat Completions endpoint lives, and the route id the
# request addresses. Both env-overridable; defaults match routes.toml / the node.
SWITCHYARD_URL = os.environ.get("SWITCHYARD_URL", "http://127.0.0.1:4000/v1/chat/completions")
SWITCHYARD_ROUTE = os.environ.get("SWITCHYARD_ROUTE", "research")

# Network timeout for a single Switchyard call (seconds). A converse turn with a
# real backtest downstream can be slow; give it room but bound it.
_HTTP_TIMEOUT = 120.0


# --------------------------------------------------------------------------- #
# Converse request -> OpenAI Chat Completions (pure functions, unit-tested).  #
# --------------------------------------------------------------------------- #

def _text_of(blocks: list[dict]) -> str:
    """Join the ``text`` blocks of a Converse content list."""
    return "".join(b.get("text", "") for b in blocks if isinstance(b, dict) and "text" in b)


def _stringify_tool_result(content: list[dict]) -> str:
    """Flatten a Converse ``toolResult.content`` list to a single string.

    The loop answers a tool with ``content=[{"json": <result dict>}]``; it may
    also carry ``{"text": ...}`` blocks. JSON blocks are serialized compactly so
    the model reads a faithful, parseable result; text blocks pass through.
    """
    parts: list[str] = []
    for b in content or []:
        if not isinstance(b, dict):
            continue
        if "json" in b:
            parts.append(json.dumps(b["json"]))
        elif "text" in b:
            parts.append(b["text"])
    return "".join(parts)


def _messages_to_openai(messages: list[dict], system: list[dict] | None) -> list[dict]:
    """Bedrock (messages, system) -> OpenAI chat messages.

    * ``system=[{"text":...}]`` -> a single leading ``{"role":"system",...}`` msg.
    * user/assistant ``{"text":...}`` blocks -> ``content`` text.
    * assistant ``toolUse`` blocks -> assistant ``tool_calls`` (id/name/arguments),
      preserving the SAME toolUseId as the OpenAI tool_call id so pairing holds.
    * user ``toolResult`` blocks -> one ``{"role":"tool","tool_call_id",...}`` msg
      per result (OpenAI wants one tool message per tool result).
    """
    out: list[dict] = []

    sys_text = _text_of(system or [])
    if sys_text:
        out.append({"role": "system", "content": sys_text})

    for m in messages or []:
        role = m.get("role")
        blocks = m.get("content") or []

        if role == "assistant":
            text = _text_of(blocks)
            tool_calls = []
            for b in blocks:
                if not isinstance(b, dict) or "toolUse" not in b:
                    continue
                tu = b["toolUse"]
                tool_calls.append({
                    "id": tu.get("toolUseId", "") or f"tooluse-{uuid.uuid4().hex}",
                    "type": "function",
                    "function": {
                        "name": tu.get("name", ""),
                        "arguments": json.dumps(tu.get("input", {}) or {}),
                    },
                })
            msg: dict[str, Any] = {"role": "assistant", "content": text or None}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            out.append(msg)
            continue

        # user (and any unknown role): text + one tool message per toolResult.
        tool_results = [b["toolResult"] for b in blocks
                        if isinstance(b, dict) and "toolResult" in b]
        text = _text_of(blocks)
        if text or not tool_results:
            # keep a user turn for the text (or an empty one if truly empty)
            out.append({"role": "user", "content": text})
        for tr in tool_results:
            out.append({
                "role": "tool",
                "tool_call_id": tr.get("toolUseId", ""),
                "content": _stringify_tool_result(tr.get("content") or []),
            })

    return out


def _tools_to_openai(tool_config: dict | None) -> list[dict]:
    """Converse ``toolConfig.tools`` (toolSpec list) -> OpenAI ``tools`` list."""
    out: list[dict] = []
    for t in (tool_config or {}).get("tools", []) or []:
        if not isinstance(t, dict):
            continue
        spec = t.get("toolSpec", t)
        name = spec.get("name")
        if not name:
            continue
        fn: dict[str, Any] = {
            "name": name,
            "parameters": (spec.get("inputSchema", {}) or {}).get("json", {}) or {},
        }
        desc = spec.get("description")
        if desc:
            fn["description"] = desc
        out.append({"type": "function", "function": fn})
    return out


def _to_openai_request(*, model: str, messages: list[dict], system: list[dict] | None,
                       tool_config: dict | None, inference_config: dict | None) -> dict:
    """Assemble the full OpenAI Chat Completions payload for Switchyard."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": _messages_to_openai(messages, system),
    }
    tools = _tools_to_openai(tool_config)
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    inf = inference_config or {}
    if inf.get("maxTokens") is not None:
        payload["max_tokens"] = int(inf["maxTokens"])
    if inf.get("temperature") is not None:
        payload["temperature"] = float(inf["temperature"])
    return payload


# --------------------------------------------------------------------------- #
# OpenAI Chat Completions response -> Converse response (pure, unit-tested).  #
# --------------------------------------------------------------------------- #

def _to_converse_response(oai: dict) -> dict:
    """OpenAI chat.completion dict -> Bedrock-Converse-shaped response dict.

    * ``tool_calls`` present -> ``stopReason:"tool_use"`` with an optional leading
      text block plus one ``{"toolUse":{"toolUseId","name","input"}}`` per call.
      The toolUseId reuses the OpenAI tool_call id (synthesized if absent) so it
      round-trips cleanly when the loop echoes the assistant turn and answers with
      a toolResult keyed by that id.
    * else -> ``stopReason:"end_turn"`` with a single ``{"text": content}`` block.
    * ``usage`` maps prompt/completion/total -> input/output/total tokens.
    * the chosen tier (OpenAI ``model`` field) is stashed at ``_switchyard_model``.
    """
    choices = oai.get("choices") or [{}]
    choice = choices[0] if choices else {}
    message = choice.get("message", {}) or {}
    tool_calls = message.get("tool_calls") or []

    content_blocks: list[dict] = []
    text = message.get("content")
    if text:
        content_blocks.append({"text": text})

    if tool_calls:
        for tc in tool_calls:
            fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
            raw_args = fn.get("arguments", "")
            try:
                parsed = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            except (ValueError, TypeError):
                parsed = {}
            content_blocks.append({"toolUse": {
                "toolUseId": (tc.get("id") if isinstance(tc, dict) else None) or f"tooluse-{uuid.uuid4().hex}",
                "name": fn.get("name", ""),
                "input": parsed if isinstance(parsed, dict) else {},
            }})
        stop_reason = "tool_use"
    else:
        if not content_blocks:
            content_blocks = [{"text": ""}]
        stop_reason = "end_turn"

    usage = oai.get("usage", {}) or {}
    resp: dict[str, Any] = {
        "output": {"message": {"role": "assistant", "content": content_blocks}},
        "stopReason": stop_reason,
        "usage": {
            "inputTokens": usage.get("prompt_tokens", 0),
            "outputTokens": usage.get("completion_tokens", 0),
            "totalTokens": usage.get("total_tokens", 0),
        },
    }
    # Preserve the tier Switchyard chose (e.g. "haiku-fast"/"sonnet"/"opus").
    resp["_switchyard_model"] = oai.get("model")
    return resp


# --------------------------------------------------------------------------- #
# HTTP sender (stdlib urllib) — injectable for tests.                         #
# --------------------------------------------------------------------------- #

def _post_json(url: str, payload: dict, *, timeout: float = _HTTP_TIMEOUT) -> dict:
    """POST ``payload`` as JSON to ``url`` and return the parsed JSON response.

    Raises on any HTTP/transport/decoding error so the caller's retry/try-except
    (in :func:`agentic_loop._converse`) handles it like a Bedrock failure.
    """
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 — trusted local endpoint
        raw = r.read()
    return json.loads(raw or b"{}")


class SwitchyardTransport:
    """A boto3-``bedrock-runtime``-duck that routes ``.converse`` through Switchyard.

    Only ``.converse(**kwargs)`` is implemented — the sole method the agentic loop
    calls on its client. It accepts the exact Converse kwargs the loop passes
    (``modelId``, ``messages``, ``toolConfig``, ``system``, ``inferenceConfig``),
    translates to an OpenAI Chat Completions request against Switchyard, and
    translates the response back into a Bedrock-Converse-shaped dict.

    Parameters
    ----------
    url : str
        Switchyard chat-completions URL (default :data:`SWITCHYARD_URL` / env).
    route : str
        Switchyard route id sent as the OpenAI ``model`` (default
        :data:`SWITCHYARD_ROUTE` / env "research").
    sender : callable, optional
        ``(url, payload) -> dict`` HTTP poster; injectable for network-free tests.
        Defaults to :func:`_post_json`.
    """

    def __init__(self, *, url: str | None = None, route: str | None = None,
                 sender: Callable[[str, dict], dict] | None = None):
        self.url = url or SWITCHYARD_URL
        self.route = route or SWITCHYARD_ROUTE
        self._sender = sender or _post_json

    def converse(self, *, modelId: str | None = None, messages: list[dict] | None = None,  # noqa: N803 — boto3 kwarg name
                 toolConfig: dict | None = None, system: list[dict] | None = None,  # noqa: N803
                 inferenceConfig: dict | None = None, **_ignored) -> dict:  # noqa: N803
        """Duck-compatible Converse call routed through Switchyard.

        ``modelId`` is accepted for signature-compatibility but IGNORED for
        routing — Switchyard classifies the request and picks the tier. Extra
        kwargs are tolerated and ignored so the loop can evolve without breaking
        this transport.
        """
        payload = _to_openai_request(
            model=self.route,
            messages=messages or [],
            system=system,
            tool_config=toolConfig,
            inference_config=inferenceConfig,
        )
        oai = self._sender(self.url, payload)
        return _to_converse_response(oai)


def make_transport(*, url: str | None = None, route: str | None = None,
                   sender: Callable[[str, dict], dict] | None = None) -> SwitchyardTransport:
    """Factory for a :class:`SwitchyardTransport` (reads env defaults when unset)."""
    return SwitchyardTransport(url=url, route=route, sender=sender)
