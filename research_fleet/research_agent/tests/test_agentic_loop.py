"""Tests for agentic_loop — the host-side Claude tool-use research loop.

These tests drive the tool-use loop with a FAKE converse client and a stubbed
``agent_tools.dispatch`` so NO network call and NO DB write happens. They prove
the loop's contract:

  * a matching toolResult (same toolUseId) is sent for every toolUse block,
  * the loop terminates when stopReason != "tool_use" and returns a summary,
  * the max_steps cap terminates a runaway (always-tool_use) model with capped=True,
  * multiple toolUse blocks in one assistant turn all get matching toolResults,
  * write-tool calls are tallied into summary["wrote"], errors are surfaced but
    never raise out of run_cycle.

One optional live end-to-end test is env-gated (RUN_LIVE_AGENTIC=1) so CI/local
never spends tokens by default.
"""

from __future__ import annotations

import os

import pytest

from research_fleet.research_agent import agentic_loop as al
from research_fleet.research_agent import agent_tools as at


# ---------------------------------------------------------------------------
# Fakes: a converse client scripted turn-by-turn + a dispatch stub (no DB).
# ---------------------------------------------------------------------------

def _tool_use_resp(name: str, tool_use_id: str, tool_input: dict | None = None) -> dict:
    """A canned assistant turn with a single toolUse block (stopReason tool_use)."""
    return {
        "stopReason": "tool_use",
        "output": {"message": {
            "role": "assistant",
            "content": [
                {"text": f"I'll call {name}."},
                {"toolUse": {"toolUseId": tool_use_id, "name": name,
                             "input": tool_input or {}}},
            ],
        }},
    }


def _multi_tool_use_resp(calls: list[tuple[str, str]]) -> dict:
    """An assistant turn with several toolUse blocks (name, id) — stopReason tool_use."""
    content: list[dict] = [{"text": "Calling several tools."}]
    for name, tid in calls:
        content.append({"toolUse": {"toolUseId": tid, "name": name, "input": {}}})
    return {"stopReason": "tool_use",
            "output": {"message": {"role": "assistant", "content": content}}}


def _final_resp(text: str = "Done. Momentum did not beat 1/N net of cost.") -> dict:
    return {"stopReason": "end_turn",
            "output": {"message": {"role": "assistant", "content": [{"text": text}]}}}


class ScriptedClient:
    """A fake Bedrock client whose .converse returns queued responses in order.

    Records every messages list it was called with, so tests can assert that the
    toolResult the loop appended matches the toolUse id from the prior turn.
    """

    def __init__(self, responses: list[dict]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def converse(self, **kwargs):
        # Snapshot the messages list (loop appends to the same object between calls).
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        if self._responses:
            return self._responses.pop(0)
        # Default to a final turn so a mis-scripted test terminates rather than KeyErrors.
        return _final_resp("(fallback final)")


class AlwaysToolUseClient:
    """A fake client that ALWAYS asks for a tool — used to prove the max_steps cap."""

    def __init__(self):
        self.calls: list[dict] = []
        self._n = 0

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        self._n += 1
        return _tool_use_resp("run_backtest", f"tu-{self._n}",
                              {"strategy_family": "momentum", "params": {}})


@pytest.fixture(autouse=True)
def _no_db_activity(monkeypatch):
    """Neutralize the best-effort activity logging so tests never touch the DB."""
    monkeypatch.setattr(al.rdb, "log_activity", lambda *a, **k: None)


@pytest.fixture
def _stub_dispatch(monkeypatch):
    """Stub agent_tools.dispatch so no real backtest/DB write runs; record calls."""
    seen: list[dict] = []

    def fake_dispatch(name, tool_input, *, agent_id, task_id=None):
        seen.append({"name": name, "input": tool_input, "agent_id": agent_id})
        if name == "run_backtest":
            return {"sharpe": 0.42, "beats_benchmark": False, "strategy_family": "momentum"}
        if name == "ask_analyst":
            return {"available": True, "sql": "SELECT 1", "rows": [[1]], "row_count": 1,
                    "text": "one", "error": None}
        if name in ("write_hypothesis", "write_experiment", "write_finding"):
            return {"ok": True, "id": f"{name}-id", "table": name}
        if name == "set_hypothesis_status":
            return {"ok": True, "status": "inconclusive"}
        return {"ok": True}

    monkeypatch.setattr(al.at, "dispatch", fake_dispatch)
    return seen


# ---------------------------------------------------------------------------
# Core loop behavior
# ---------------------------------------------------------------------------

def test_loop_sends_matching_toolresult_and_terminates(_stub_dispatch):
    client = ScriptedClient([
        _tool_use_resp("run_backtest", "tu-1",
                       {"strategy_family": "momentum", "params": {"lookback_days": 63}}),
        _final_resp("Momentum Sharpe 0.42; did not beat 1/N."),
    ])
    out = al.run_cycle("unit-agent", "momentum", model="haiku", max_steps=8, client=client)

    # (b) terminated when stopReason != tool_use
    assert out["capped"] is False
    assert out["final_text"].startswith("Momentum Sharpe")
    # (c) summary with steps>=1 and the tool name recorded
    assert out["steps"] >= 1
    assert "run_backtest" in out["tool_calls"]
    assert out["agent_id"] == "unit-agent" and out["focus"] == "momentum"

    # (a) the second converse call carried a user turn whose toolResult id matches tu-1
    second_call_messages = client.calls[1]["messages"]
    tool_result_turn = second_call_messages[-1]
    assert tool_result_turn["role"] == "user"
    tr = tool_result_turn["content"][0]["toolResult"]
    assert tr["toolUseId"] == "tu-1"
    assert tr["status"] == "success"
    assert tr["content"][0]["json"]["sharpe"] == 0.42

    # toolConfig + system were passed on every converse call
    for call in client.calls:
        assert call["toolConfig"]["tools"] is at.TOOL_SPECS
        assert call["system"][0]["text"]


def test_max_steps_cap_terminates_runaway():
    client = AlwaysToolUseClient()
    # dispatch is the real one but run_backtest("momentum", {}) won't be hit enough to
    # matter; stub it anyway to avoid any live work.
    import research_fleet.research_agent.agentic_loop as mod
    orig = mod.at.dispatch
    mod.at.dispatch = lambda name, ti, *, agent_id, task_id=None: {"sharpe": 0.0}
    try:
        out = al.run_cycle("unit-agent", "momentum", model="haiku", max_steps=4, client=client)
    finally:
        mod.at.dispatch = orig

    assert out["capped"] is True
    assert out["steps"] == 4
    # exactly max_steps converse calls, proving no infinite loop / no runaway spend
    assert len(client.calls) == 4


def test_multiple_tooluse_blocks_get_matching_toolresults(_stub_dispatch):
    client = ScriptedClient([
        _multi_tool_use_resp([("recall_findings", "a1"), ("query_sweep", "a2"),
                              ("list_recent_experiments", "a3")]),
        _final_resp("Reviewed prior knowledge; nothing to add this cycle."),
    ])
    out = al.run_cycle("unit-agent", "momentum", model="haiku", max_steps=8, client=client)

    # every toolUse recorded
    assert out["tool_calls"] == ["recall_findings", "query_sweep", "list_recent_experiments"]

    # the follow-up user turn has 3 toolResult blocks with ids matching a1/a2/a3
    follow_up = client.calls[1]["messages"][-1]
    assert follow_up["role"] == "user"
    ids = [b["toolResult"]["toolUseId"] for b in follow_up["content"]]
    assert ids == ["a1", "a2", "a3"]
    assert all("toolResult" in b for b in follow_up["content"])


def test_write_tools_tallied_and_status_ok(_stub_dispatch):
    client = ScriptedClient([
        _tool_use_resp("write_hypothesis", "h1"),
        _tool_use_resp("run_backtest", "b1"),
        _tool_use_resp("write_experiment", "e1"),
        _tool_use_resp("write_finding", "f1"),
        _tool_use_resp("set_hypothesis_status", "s1"),
        _final_resp("Arc complete."),
    ])
    out = al.run_cycle("unit-agent", "momentum", model="haiku", max_steps=16, client=client)
    assert out["wrote"] == {"hypotheses": 1, "experiments": 1, "findings": 1}
    assert out["capped"] is False
    assert out["final_text"] == "Arc complete."
    # every write toolResult was marked success
    for call in client.calls[1:]:
        last = call["messages"][-1]
        if last["role"] == "user":
            for b in last["content"]:
                assert b["toolResult"]["status"] == "success"


def test_ask_analyst_tallied_in_summary(_stub_dispatch):
    client = ScriptedClient([
        _tool_use_resp("ask_analyst", "aq1",
                       {"question": "avg sharpe by family"}),
        _final_resp("Used Aura Analyst to compare families."),
    ])
    out = al.run_cycle("unit-agent", "momentum", model="haiku", max_steps=8, client=client)
    assert out["analyst_queries"] == 1
    assert "ask_analyst" in out["tool_calls"]
    # ask_analyst is NOT a write — it must not inflate the write tally
    assert out["wrote"] == {"hypotheses": 0, "experiments": 0, "findings": 0}


def test_ask_analyst_unavailable_not_tallied(monkeypatch):
    # When Aura is not configured, dispatch returns {available: False}; that is a
    # clean skip, not a query, and must not be counted.
    def dispatch_unavailable(name, tool_input, *, agent_id, task_id=None):
        if name == "ask_analyst":
            return {"available": False, "note": "not configured"}
        return {"ok": True}

    monkeypatch.setattr(al.at, "dispatch", dispatch_unavailable)
    client = ScriptedClient([
        _tool_use_resp("ask_analyst", "aq1", {"question": "q"}),
        _final_resp("Aura unavailable; proceeded without it."),
    ])
    out = al.run_cycle("unit-agent", "momentum", model="haiku", max_steps=8, client=client)
    assert out["analyst_queries"] == 0
    # an {available:false} result is not an error — the toolResult status is success
    tr = client.calls[1]["messages"][-1]["content"][0]["toolResult"]
    assert tr["status"] == "success"


def test_dispatch_error_marks_toolresult_error_and_not_tallied(monkeypatch):
    def erroring_dispatch(name, tool_input, *, agent_id, task_id=None):
        if name == "write_finding":
            return {"ok": False, "error": "invalid strategy_family"}
        return {"sharpe": 0.1}

    monkeypatch.setattr(al.at, "dispatch", erroring_dispatch)
    client = ScriptedClient([
        _tool_use_resp("write_finding", "f1"),
        _final_resp("Could not write finding."),
    ])
    out = al.run_cycle("unit-agent", "momentum", model="haiku", max_steps=8, client=client)

    # error result must NOT be counted as a successful write
    assert out["wrote"]["findings"] == 0
    tr = client.calls[1]["messages"][-1]["content"][0]["toolResult"]
    assert tr["status"] == "error"


def test_converse_hard_failure_ends_cycle_gracefully(monkeypatch):
    class BoomClient:
        def converse(self, **kwargs):
            raise RuntimeError("gateway exploded")

    # no retry sleeping in the test
    monkeypatch.setattr(al, "_CONVERSE_BACKOFF", 0.0)
    out = al.run_cycle("unit-agent", "momentum", model="haiku", max_steps=8,
                       client=BoomClient())
    assert "error" in out and "gateway exploded" in out["error"]
    # it did not raise, and the summary is still well-formed
    assert out["wrote"] == {"hypotheses": 0, "experiments": 0, "findings": 0}
    assert out["capped"] is False


def test_unconfigured_model_returns_error_without_client(monkeypatch):
    # No client injected + empty endpoint => graceful error, no crash.
    monkeypatch.setattr(al.llm_driver, "LLM_ENDPOINT", "")
    out = al.run_cycle("unit-agent", "momentum", model="sonnet", max_steps=4)
    assert "error" in out and "not configured" in out["error"]


def test_no_tooluse_block_but_tool_use_stop_ends_gracefully(_stub_dispatch):
    # Malformed: stopReason tool_use but the content has no toolUse block.
    weird = {"stopReason": "tool_use",
             "output": {"message": {"role": "assistant",
                                    "content": [{"text": "hmm no tool here"}]}}}
    client = ScriptedClient([weird])
    out = al.run_cycle("unit-agent", "momentum", model="haiku", max_steps=8, client=client)
    assert out["capped"] is False
    assert out["final_text"] == "hmm no tool here"
    # only one converse call — it did not send an empty user turn
    assert len(client.calls) == 1


# ---------------------------------------------------------------------------
# Optional live end-to-end (env-gated; spends tokens + writes/cleans DB rows)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(os.environ.get("RUN_LIVE_AGENTIC") != "1",
                    reason="live LLM+DB test; set RUN_LIVE_AGENTIC=1 to run")
def test_live_one_cycle_end_to_end():
    agent = "test-agentic-live-DELETEME"
    from research_fleet.research_agent import research_db as rdb
    try:
        out = al.run_cycle(agent, "momentum", model="haiku", max_steps=8)
        assert out["steps"] >= 1
        assert "run_backtest" in out["tool_calls"]
    finally:
        for table in ("research_findings", "research_experiments",
                      "research_hypotheses", "research_activity", "research_agents"):
            try:
                rdb.execute(f"DELETE FROM {table} WHERE agent_id=%s", (agent,))
            except Exception:
                pass
