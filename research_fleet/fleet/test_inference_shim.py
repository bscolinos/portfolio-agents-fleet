"""Unit tests for the OpenAI<->Bedrock-Converse translation in inference_shim.

These exercise the pure translation functions with NO server and NO network.
The module imports boto3 (installed) but never opens a socket at import time.

Run: python -m pytest research_fleet/fleet/test_inference_shim.py -q
"""

from __future__ import annotations

import json

import inference_shim as shim


# --------------------------------------------------------------------------- #
# tools -> Converse toolConfig                                                 #
# --------------------------------------------------------------------------- #

def test_openai_tools_to_converse_shape():
    tools = [{
        "type": "function",
        "function": {
            "name": "run_backtest",
            "description": "Backtest a strategy over a date range.",
            "parameters": {
                "type": "object",
                "properties": {"strategy": {"type": "string"}, "years": {"type": "integer"}},
                "required": ["strategy"],
            },
        },
    }]
    out = shim._tools_to_converse(tools)
    assert len(out) == 1
    spec = out[0]["toolSpec"]
    assert spec["name"] == "run_backtest"
    assert spec["description"] == "Backtest a strategy over a date range."
    # the OpenAI `parameters` schema must land verbatim under inputSchema.json
    assert spec["inputSchema"]["json"]["required"] == ["strategy"]
    assert spec["inputSchema"]["json"]["properties"]["years"]["type"] == "integer"


def test_tool_choice_translation():
    assert shim._tool_choice_to_converse("auto") == {"auto": {}}
    assert shim._tool_choice_to_converse(None) == {"auto": {}}
    assert shim._tool_choice_to_converse("required") == {"any": {}}
    assert shim._tool_choice_to_converse("none") is None
    assert shim._tool_choice_to_converse(
        {"type": "function", "function": {"name": "run_backtest"}}
    ) == {"tool": {"name": "run_backtest"}}


# --------------------------------------------------------------------------- #
# messages -> Converse (tool-use round-trip)                                   #
# --------------------------------------------------------------------------- #

def test_assistant_tool_calls_and_tool_results_roundtrip():
    messages = [
        {"role": "system", "content": "You are a quant."},
        {"role": "user", "content": "Backtest momentum."},
        {"role": "assistant", "content": "Running it.",
         "tool_calls": [
             {"id": "call_1", "type": "function",
              "function": {"name": "run_backtest", "arguments": '{"strategy": "momentum"}'}},
             {"id": "call_2", "type": "function",
              "function": {"name": "run_backtest", "arguments": '{"strategy": "value"}'}},
         ]},
        {"role": "tool", "tool_call_id": "call_1", "content": '{"sharpe": 1.4}'},
        {"role": "tool", "tool_call_id": "call_2", "content": '{"sharpe": 0.9}'},
    ]
    conv, system = shim._to_converse(messages)

    assert system == [{"text": "You are a quant."}]
    # user, assistant(text+2 toolUse), user(2 merged toolResults)
    assert [t["role"] for t in conv] == ["user", "assistant", "user"]

    asst = conv[1]["content"]
    assert asst[0] == {"text": "Running it."}
    tool_uses = [b["toolUse"] for b in asst if "toolUse" in b]
    assert [tu["toolUseId"] for tu in tool_uses] == ["call_1", "call_2"]
    assert tool_uses[0]["name"] == "run_backtest"
    assert tool_uses[0]["input"] == {"strategy": "momentum"}  # arguments parsed from JSON string

    # consecutive tool results merged into a single user turn, ids preserved
    results = conv[2]["content"]
    assert len(results) == 2
    assert results[0]["toolResult"]["toolUseId"] == "call_1"
    assert results[0]["toolResult"]["status"] == "success"
    assert results[0]["toolResult"]["content"] == [{"text": '{"sharpe": 1.4}'}]
    assert results[1]["toolResult"]["toolUseId"] == "call_2"


def test_convo_normalized_to_start_with_user():
    conv, _ = shim._to_converse([{"role": "assistant", "content": "hi"}])
    assert conv[0]["role"] == "user"


# --------------------------------------------------------------------------- #
# response_format json_schema -> forced tool                                   #
# --------------------------------------------------------------------------- #

def test_response_format_json_schema_forces_single_tool():
    schema = {
        "type": "object",
        "properties": {"p_solve": {"type": "number"}, "capability_boundary": {"type": "string"}},
        "required": ["p_solve", "capability_boundary"],
    }
    rf = {"type": "json_schema", "json_schema": {"name": "verdict", "schema": schema}}
    tool_config, forced_name, extra_system = shim._apply_response_format(rf)

    assert forced_name == "verdict"
    assert extra_system is None
    assert tool_config["toolChoice"] == {"tool": {"name": "verdict"}}
    assert len(tool_config["tools"]) == 1
    assert tool_config["tools"][0]["toolSpec"]["name"] == "verdict"
    assert tool_config["tools"][0]["toolSpec"]["inputSchema"]["json"] == schema


def test_response_format_json_object_appends_system():
    tool_config, forced_name, extra_system = shim._apply_response_format({"type": "json_object"})
    assert tool_config is None
    assert forced_name is None
    assert "JSON" in extra_system["text"]


def _fake_converse_resp(content_blocks, usage=None):
    return {
        "output": {"message": {"content": content_blocks}},
        "usage": usage or {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
        "stopReason": "tool_use",
    }


def test_forced_tool_response_becomes_json_content_not_tool_calls():
    resp = _fake_converse_resp([
        {"toolUse": {"toolUseId": "t1", "name": "verdict",
                     "input": {"p_solve": 0.92, "capability_boundary": "trivial"}}},
    ])
    out = shim._response_to_openai(resp, "haiku", forced_tool_name="verdict", real_tools=False)
    msg = out["choices"][0]["message"]
    assert "tool_calls" not in msg
    assert out["choices"][0]["finish_reason"] == "stop"
    parsed = json.loads(msg["content"])  # content must be valid JSON text
    assert parsed["p_solve"] == 0.92
    assert parsed["capability_boundary"] == "trivial"


# --------------------------------------------------------------------------- #
# real tools -> OpenAI tool_calls                                              #
# --------------------------------------------------------------------------- #

def test_real_tool_use_response_becomes_tool_calls():
    resp = _fake_converse_resp([
        {"text": "Let me run that."},
        {"toolUse": {"toolUseId": "tu_42", "name": "run_backtest",
                     "input": {"strategy": "momentum"}}},
    ])
    out = shim._response_to_openai(resp, "haiku", forced_tool_name=None, real_tools=True)
    choice = out["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    msg = choice["message"]
    assert msg["content"] == "Let me run that."
    assert len(msg["tool_calls"]) == 1
    tc = msg["tool_calls"][0]
    assert tc["id"] == "tu_42"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "run_backtest"
    assert json.loads(tc["function"]["arguments"]) == {"strategy": "momentum"}


# --------------------------------------------------------------------------- #
# plain text passthrough (regression)                                          #
# --------------------------------------------------------------------------- #

def test_plain_text_passthrough_unchanged():
    resp = _fake_converse_resp([{"text": "Hello, world."}])
    out = shim._response_to_openai(resp, "sonnet", forced_tool_name=None, real_tools=False)
    choice = out["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"] == {"role": "assistant", "content": "Hello, world."}
    assert out["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    assert out["object"] == "chat.completion"


def test_build_converse_kwargs_plain_text_has_no_toolconfig():
    payload = {"model": "sonnet", "messages": [{"role": "user", "content": "hi"}]}
    kwargs, forced_name, real_tools = shim._build_converse_kwargs(payload, "claude-x")
    assert "toolConfig" not in kwargs
    assert forced_name is None
    assert real_tools is False
    assert kwargs["inferenceConfig"]["maxTokens"] == 2048  # unchanged default


def test_build_converse_kwargs_real_tools_sets_toolconfig_and_bumps_tokens():
    payload = {
        "model": "haiku",
        "messages": [{"role": "user", "content": "backtest it"}],
        "tools": [{"type": "function", "function": {"name": "run_backtest",
                                                    "parameters": {"type": "object"}}}],
        "tool_choice": "auto",
    }
    kwargs, forced_name, real_tools = shim._build_converse_kwargs(payload, "claude-x")
    assert real_tools is True
    assert forced_name is None
    assert kwargs["toolConfig"]["toolChoice"] == {"auto": {}}
    assert kwargs["toolConfig"]["tools"][0]["toolSpec"]["name"] == "run_backtest"
    assert kwargs["inferenceConfig"]["maxTokens"] == 4096  # bumped for tool use


def test_build_converse_kwargs_response_format_forces_tool():
    payload = {
        "model": "haiku",
        "messages": [{"role": "user", "content": "estimate"}],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "verdict",
            "schema": {"type": "object", "properties": {"p_solve": {"type": "number"}}},
        }},
    }
    kwargs, forced_name, real_tools = shim._build_converse_kwargs(payload, "claude-x")
    assert forced_name == "verdict"
    assert real_tools is False
    assert kwargs["toolConfig"]["toolChoice"] == {"tool": {"name": "verdict"}}


def test_tool_choice_none_drops_tools():
    payload = {
        "model": "haiku",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
        "tool_choice": "none",
    }
    kwargs, forced_name, real_tools = shim._build_converse_kwargs(payload, "claude-x")
    assert "toolConfig" not in kwargs
    assert real_tools is False


def test_explicit_max_tokens_respected_with_tools():
    payload = {
        "model": "haiku",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
        "max_tokens": 128,
    }
    kwargs, _, _ = shim._build_converse_kwargs(payload, "claude-x")
    assert kwargs["inferenceConfig"]["maxTokens"] == 128
