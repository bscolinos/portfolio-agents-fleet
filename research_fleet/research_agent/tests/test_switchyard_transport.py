"""Tests for switchyard_transport — the Converse-shaped adapter over Switchyard.

These are pure, network-free tests. They exercise BOTH translation directions
(Converse request -> OpenAI, OpenAI response -> Converse), prove tool-use id
round-trip stability, and drive a full `run_cycle` with a FAKE http poster
injected so no socket opens. A companion test proves the DEFAULT (bedrock)
transport still takes the Bedrock client path unchanged.

Run: python -m pytest research_fleet/research_agent/tests/test_switchyard_transport.py -q
"""

from __future__ import annotations

import json

import pytest

from research_fleet.research_agent import agentic_loop as al
from research_fleet.research_agent import agent_tools as at
from research_fleet.research_agent import switchyard_transport as sx


# ---------------------------------------------------------------------------
# Converse request -> OpenAI
# ---------------------------------------------------------------------------

def test_system_block_becomes_leading_system_message():
    msgs = sx._messages_to_openai(
        [{"role": "user", "content": [{"text": "Begin a research cycle."}]}],
        [{"text": "You are a quant."}],
    )
    assert msgs[0] == {"role": "system", "content": "You are a quant."}
    assert msgs[1] == {"role": "user", "content": "Begin a research cycle."}


def test_tools_toolconfig_to_openai_shape():
    tool_config = {"tools": at.TOOL_SPECS}
    tools = sx._tools_to_openai(tool_config)
    assert len(tools) == len(at.TOOL_SPECS)
    first = tools[0]
    assert first["type"] == "function"
    fn = first["function"]
    # name/description/parameters=inputSchema.json all carried across
    assert fn["name"] == at.TOOL_SPECS[0]["toolSpec"]["name"]
    assert fn["description"] == at.TOOL_SPECS[0]["toolSpec"]["description"]
    assert fn["parameters"] == at.TOOL_SPECS[0]["toolSpec"]["inputSchema"]["json"]


def test_assistant_tooluse_and_user_toolresult_to_openai():
    # A Converse history: user kickoff, assistant emits a toolUse, user answers
    # with a matching toolResult (same id) — the exact shape run_cycle builds.
    messages = [
        {"role": "user", "content": [{"text": "Begin."}]},
        {"role": "assistant", "content": [
            {"text": "I'll backtest."},
            {"toolUse": {"toolUseId": "tu-7", "name": "run_backtest",
                         "input": {"strategy_family": "momentum", "params": {}}}},
        ]},
        {"role": "user", "content": [
            {"toolResult": {"toolUseId": "tu-7",
                            "content": [{"json": {"sharpe": 0.42}}],
                            "status": "success"}},
        ]},
    ]
    out = sx._messages_to_openai(messages, None)

    # assistant turn -> assistant with tool_calls (matching id + json.dumps input)
    asst = next(m for m in out if m["role"] == "assistant")
    assert asst["content"] == "I'll backtest."
    tc = asst["tool_calls"][0]
    assert tc["id"] == "tu-7"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "run_backtest"
    assert json.loads(tc["function"]["arguments"]) == {"strategy_family": "momentum", "params": {}}

    # toolResult -> role:"tool" message with matching tool_call_id + stringified result
    tool_msg = next(m for m in out if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "tu-7"
    assert json.loads(tool_msg["content"]) == {"sharpe": 0.42}


def test_request_maxtokens_and_temperature_and_route():
    payload = sx._to_openai_request(
        model="research",
        messages=[{"role": "user", "content": [{"text": "hi"}]}],
        system=[{"text": "sys"}],
        tool_config={"tools": at.TOOL_SPECS},
        inference_config={"maxTokens": 2048, "temperature": 0.3},
    )
    assert payload["model"] == "research"
    assert payload["max_tokens"] == 2048
    assert payload["temperature"] == 0.3
    assert payload["tool_choice"] == "auto"
    assert payload["tools"]  # tools present


# ---------------------------------------------------------------------------
# OpenAI response -> Converse
# ---------------------------------------------------------------------------

def _oai_tool_call_resp(name: str, tool_id: str, args: dict, *, model="sonnet",
                        text: str | None = None) -> dict:
    msg = {"role": "assistant", "content": text,
           "tool_calls": [{"id": tool_id, "type": "function",
                           "function": {"name": name, "arguments": json.dumps(args)}}]}
    return {"model": model, "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": msg}]}


def _oai_plain_resp(text: str, *, model="haiku-fast") -> dict:
    return {"model": model, "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}]}


def test_tool_call_response_to_converse():
    oai = _oai_tool_call_resp("run_backtest", "call_abc",
                              {"strategy_family": "momentum", "params": {"lookback_days": 63}},
                              model="opus", text="Backtesting now.")
    resp = sx._to_converse_response(oai)
    assert resp["stopReason"] == "tool_use"
    content = resp["output"]["message"]["content"]
    # leading text block preserved, then the toolUse block
    assert content[0] == {"text": "Backtesting now."}
    tu = content[1]["toolUse"]
    assert tu["toolUseId"] == "call_abc"
    assert tu["name"] == "run_backtest"
    assert tu["input"] == {"strategy_family": "momentum", "params": {"lookback_days": 63}}
    # usage mapped + tier preserved
    assert resp["usage"] == {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15}
    assert resp["_switchyard_model"] == "opus"


def test_plain_response_to_converse():
    resp = sx._to_converse_response(_oai_plain_resp("Momentum did not beat 1/N.", model="haiku-fast"))
    assert resp["stopReason"] == "end_turn"
    assert resp["output"]["message"]["content"] == [{"text": "Momentum did not beat 1/N."}]
    assert resp["usage"]["totalTokens"] == 10
    assert resp["_switchyard_model"] == "haiku-fast"


def test_tool_call_without_id_synthesizes_stable_id():
    oai = {"model": "sonnet", "choices": [{"finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": None,
        "tool_calls": [{"type": "function",
                        "function": {"name": "recall_findings", "arguments": "{}"}}]}}]}
    resp = sx._to_converse_response(oai)
    tid = resp["output"]["message"]["content"][0]["toolUse"]["toolUseId"]
    assert tid and tid.startswith("tooluse-")


# ---------------------------------------------------------------------------
# Round-trip id stability
# ---------------------------------------------------------------------------

def test_tooluse_id_round_trips_cleanly():
    # 1) Switchyard returns a tool_call with a synthesized id -> Converse toolUse.
    resp = sx._to_converse_response(
        _oai_tool_call_resp("run_backtest", "call_xyz", {"strategy_family": "value", "params": {}}))
    assistant_msg = resp["output"]["message"]
    tool_use_id = assistant_msg["content"][-1]["toolUse"]["toolUseId"]
    assert tool_use_id == "call_xyz"

    # 2) The loop echoes that assistant turn verbatim and appends a toolResult
    #    keyed by that id. Translate the whole history BACK to OpenAI.
    history = [
        {"role": "user", "content": [{"text": "Begin."}]},
        assistant_msg,
        {"role": "user", "content": [
            {"toolResult": {"toolUseId": tool_use_id,
                            "content": [{"json": {"sharpe": 0.9}}], "status": "success"}}]},
    ]
    oai_msgs = sx._messages_to_openai(history, None)
    asst = next(m for m in oai_msgs if m["role"] == "assistant")
    tool = next(m for m in oai_msgs if m["role"] == "tool")
    # the id emitted on the way out lines up on the way back in
    assert asst["tool_calls"][0]["id"] == tool["tool_call_id"] == "call_xyz"


# ---------------------------------------------------------------------------
# Full transport + full run_cycle with an injected FAKE http poster (no network)
# ---------------------------------------------------------------------------

def test_transport_returns_converse_shape_with_fake_sender():
    def fake_sender(url, payload):
        assert payload["model"] == "research"  # route id sent as model
        return _oai_plain_resp("hello", model="sonnet")

    t = sx.make_transport(sender=fake_sender)
    resp = t.converse(modelId="ignored", messages=[{"role": "user", "content": [{"text": "hi"}]}],
                      toolConfig={"tools": at.TOOL_SPECS}, system=[{"text": "sys"}],
                      inferenceConfig={"maxTokens": 100, "temperature": 0.3})
    assert resp["stopReason"] == "end_turn"
    assert resp["output"]["message"]["content"][0]["text"] == "hello"
    assert resp["_switchyard_model"] == "sonnet"


def test_run_cycle_completes_with_switchyard_transport(monkeypatch):
    # No DB, no real dispatch, no network — one tool call then a final answer.
    monkeypatch.setattr(al.rdb, "log_activity", lambda *a, **k: None)

    def fake_dispatch(name, tool_input, *, agent_id, task_id=None):
        if name == "run_backtest":
            return {"sharpe": 0.42, "beats_benchmark": False}
        return {"ok": True, "id": f"{name}-id"}
    monkeypatch.setattr(al.at, "dispatch", fake_dispatch)

    # A scripted Switchyard: first turn routes to opus + emits a tool call, second
    # turn routes to haiku-fast + gives the final answer.
    scripted = [
        _oai_tool_call_resp("run_backtest", "call_1",
                            {"strategy_family": "momentum", "params": {}}, model="opus"),
        _oai_plain_resp("Momentum Sharpe 0.42; did not beat 1/N.", model="haiku-fast"),
    ]
    sent: list[dict] = []

    def fake_sender(url, payload):
        sent.append(payload)
        return scripted.pop(0)

    transport = sx.make_transport(sender=fake_sender)
    out = al.run_cycle("unit-agent", "momentum", model="sonnet", max_steps=8, client=transport)

    assert out["capped"] is False
    assert out["final_text"].startswith("Momentum Sharpe")
    assert "run_backtest" in out["tool_calls"]
    # the tier chosen per turn was recorded (visibility is the whole point)
    assert out["tiers"] == ["opus", "haiku-fast"]

    # the SECOND request the loop sent back to Switchyard carried the assistant
    # tool_call + a role:"tool" result with the SAME id — proves round-trip pairing.
    second = sent[1]
    asst = next(m for m in second["messages"] if m["role"] == "assistant" and m.get("tool_calls"))
    tool = next(m for m in second["messages"] if m["role"] == "tool")
    assert asst["tool_calls"][0]["id"] == tool["tool_call_id"] == "call_1"


def test_transport_raises_on_sender_error_so_loop_retry_handles_it():
    def boom_sender(url, payload):
        raise RuntimeError("switchyard down")

    t = sx.make_transport(sender=boom_sender)
    with pytest.raises(RuntimeError, match="switchyard down"):
        t.converse(modelId="x", messages=[{"role": "user", "content": [{"text": "hi"}]}],
                   toolConfig=None, system=None, inferenceConfig={"maxTokens": 10})


# ---------------------------------------------------------------------------
# DEFAULT (bedrock) path is unchanged — run_cycle with no transport flag still
# builds the Bedrock client via llm_driver._client, NEVER the Switchyard transport.
# ---------------------------------------------------------------------------

def test_default_transport_uses_bedrock_client_path(monkeypatch):
    monkeypatch.setattr(al.rdb, "log_activity", lambda *a, **k: None)
    # Make the bedrock path buildable + observable.
    monkeypatch.setattr(al.llm_driver, "LLM_ENDPOINT", "http://fake-endpoint")
    monkeypatch.setattr(al.llm_driver, "MODELS",
                        {"sonnet": ("sonnet-model-id", "jwt-xyz"), "haiku": ("haiku-id", "jwt-h")})

    built = {"bedrock": 0, "switchyard": 0}

    class FakeBedrock:
        def converse(self, **kwargs):
            # a plain final turn so the cycle ends immediately
            return {"stopReason": "end_turn",
                    "output": {"message": {"role": "assistant", "content": [{"text": "done"}]}}}

    def fake_bedrock_client(key):
        built["bedrock"] += 1
        assert key == "jwt-xyz"  # focus model's JWT used, unchanged behavior
        return FakeBedrock()

    def fake_make_transport(*a, **k):
        built["switchyard"] += 1
        raise AssertionError("Switchyard transport must NOT be built on the default path")

    monkeypatch.setattr(al.llm_driver, "_client", fake_bedrock_client)
    monkeypatch.setattr(al.switchyard_transport, "make_transport", fake_make_transport)

    # No transport kwarg => default "bedrock".
    out = al.run_cycle("unit-agent", "momentum", model="sonnet", max_steps=4)

    assert out["transport"] == "bedrock"
    assert built["bedrock"] == 1 and built["switchyard"] == 0
    assert out["tiers"] == []           # no tier routing on the bedrock path
    assert out["final_text"] == "done"
