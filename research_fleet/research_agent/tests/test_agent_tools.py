"""Tests for agent_tools — the tool suite for the autonomous research loop.

Covers:
  * TOOL_SPECS is a valid Bedrock toolSpec list (each has toolSpec.name +
    inputSchema.json.type=="object"), names are unique, and every declared read
    + write tool is present.
  * dispatch runs the REAL backtest (numeric sharpe) — the anti-fabrication
    guarantee — and returns a dict for a bad family instead of raising.
  * dispatch never raises: bad family, bad payloads, unknown tool all return dicts.
  * read tools (recall_findings, query_sweep) return real data without raising.
  * write tools inject agent_id host-side and go through the validated path; a bad
    strategy_family write returns an error dict (not a raise).

DB-backed tests guard-skip when SingleStore is unreachable, so the structural
tests still run anywhere. Every row this suite writes is tagged with a sentinel
agent_id and DELETED in teardown; the module asserts 0 leftover rows at the end.
"""

from __future__ import annotations

import pytest

from research_fleet.research_agent import agent_tools as at


# The sentinel agent_id every write in this suite uses, so teardown can find + purge.
TEST_AGENT = "test-tools-DELETEME"


def _db_reachable() -> bool:
    try:
        from research_fleet.research_agent import research_db as rdb
        rdb.query("SELECT 1 AS ok")
        return True
    except Exception:
        return False


DB_OK = _db_reachable()
_skip_no_db = pytest.mark.skipif(not DB_OK, reason="SingleStore unreachable")


# ---------------------------------------------------------------------------
# Structural: TOOL_SPECS is a valid Bedrock toolSpec list (no DB needed)
# ---------------------------------------------------------------------------

def test_tool_specs_are_valid_bedrock_toolspecs():
    assert isinstance(at.TOOL_SPECS, list) and at.TOOL_SPECS
    names = []
    for spec in at.TOOL_SPECS:
        assert isinstance(spec, dict) and "toolSpec" in spec
        ts = spec["toolSpec"]
        assert isinstance(ts.get("name"), str) and ts["name"]
        assert isinstance(ts.get("description"), str) and ts["description"]
        schema = ts.get("inputSchema", {}).get("json")
        assert isinstance(schema, dict), f"{ts['name']} missing inputSchema.json"
        assert schema.get("type") == "object", f"{ts['name']} schema.type must be 'object'"
        assert isinstance(schema.get("properties"), dict)
        assert isinstance(schema.get("required", []), list)
        names.append(ts["name"])
    # names unique
    assert len(names) == len(set(names)), f"duplicate tool names: {names}"


def test_all_expected_tools_present():
    names = set(s["toolSpec"]["name"] for s in at.TOOL_SPECS)
    expected = {
        "recall_findings", "run_backtest", "query_sweep", "list_recent_experiments",
        "write_hypothesis", "write_experiment", "write_finding", "set_hypothesis_status",
    }
    assert expected <= names, f"missing tools: {expected - names}"


# ---------------------------------------------------------------------------
# dispatch never raises on bad input (no DB needed for these paths)
# ---------------------------------------------------------------------------

def test_bad_family_backtest_returns_error_not_raise():
    out = at.dispatch("run_backtest", {"strategy_family": "bogus", "params": {}},
                      agent_id="unit")
    assert isinstance(out, dict)
    assert "error" in out
    assert out.get("sharpe") is None  # no fabricated metric


def test_unknown_tool_returns_error_dict():
    out = at.dispatch("does_not_exist", {}, agent_id="unit")
    assert isinstance(out, dict) and "error" in out


def test_non_dict_input_returns_error_dict():
    out = at.dispatch("run_backtest", None, agent_id="unit")  # type: ignore[arg-type]
    assert isinstance(out, dict) and "error" in out


def test_bad_family_write_returns_error_not_raise():
    out = at.dispatch("write_hypothesis",
                      {"statement": "x", "strategy_family": "not_a_family"},
                      agent_id=TEST_AGENT)
    assert isinstance(out, dict)
    assert out.get("ok") is False and "error" in out
    # no id was minted for an invalid write
    assert "id" not in out


# ---------------------------------------------------------------------------
# Read tools over the real DB (guard-skipped if unreachable)
# ---------------------------------------------------------------------------

@_skip_no_db
def test_run_backtest_returns_real_numeric_sharpe():
    out = at.dispatch("run_backtest",
                      {"strategy_family": "momentum",
                       "params": {"lookback_days": 63, "top_n": 20},
                       "start": "2020-01-01", "end": "2024-12-31", "universe_n": 40},
                      agent_id="unit")
    assert isinstance(out, dict) and "error" not in out
    assert isinstance(out["sharpe"], (int, float)), "real backtest must return a numeric sharpe"
    # real backtester contract: benchmark comparison + honesty caveat present
    assert "beats_benchmark" in out
    assert "data_caveats" in out and isinstance(out["data_caveats"], str)


@_skip_no_db
def test_recall_findings_returns_without_raising():
    out = at.dispatch("recall_findings", {"query": "momentum edge", "k": 3}, agent_id="unit")
    assert isinstance(out, dict) and "error" not in out
    assert isinstance(out.get("findings"), list)
    assert len(out["findings"]) <= 3


@_skip_no_db
def test_query_sweep_returns_real_rows():
    out = at.dispatch("query_sweep", {"limit": 5}, agent_id="unit")
    assert isinstance(out, dict) and "error" not in out
    rows = out.get("rows")
    assert isinstance(rows, list) and rows, "sweep_results should return rows"
    assert len(rows) <= 5
    # OOS-sorted by default; rows carry the sweep's IS/OOS fields
    for r in rows:
        assert "family" in r and "oos_sharpe" in r and "is_sharpe" in r


@_skip_no_db
def test_list_recent_experiments_returns_without_raising():
    out = at.dispatch("list_recent_experiments", {"limit": 5}, agent_id="unit")
    assert isinstance(out, dict) and "error" not in out
    assert isinstance(out.get("experiments"), list)
    assert len(out["experiments"]) <= 5


# ---------------------------------------------------------------------------
# Write tool over the real DB: injects agent_id, returns id, cleaned up
# ---------------------------------------------------------------------------

@_skip_no_db
def test_write_hypothesis_injects_agent_and_returns_id():
    out = at.dispatch("write_hypothesis",
                      {"statement": "DELETEME test hypothesis",
                       "strategy_family": "momentum", "confidence": 0.5},
                      agent_id=TEST_AGENT, task_id="task-DELETEME")
    assert isinstance(out, dict)
    assert out.get("ok") is True
    assert isinstance(out.get("id"), str) and out["id"]
    # confirm the row was written under the INJECTED agent_id (not something spoofed)
    from research_fleet.research_agent import research_db as rdb
    rows = rdb.query("SELECT agent_id, task_id FROM research_hypotheses WHERE hypothesis_id=%s",
                     (out["id"],))
    assert rows and rows[0]["agent_id"] == TEST_AGENT
    assert rows[0]["task_id"] == "task-DELETEME"


# ---------------------------------------------------------------------------
# system_prompt
# ---------------------------------------------------------------------------

def test_system_prompt_is_strong_directive():
    sp = at.system_prompt("momentum")
    assert isinstance(sp, str) and len(sp) > 200
    low = sp.lower()
    assert "momentum" in low
    assert "run_backtest" in low and "recall_findings" in low
    assert "1/n" in low or "benchmark" in low


# ---------------------------------------------------------------------------
# Teardown: purge every row this suite could have written; assert 0 remain.
# ---------------------------------------------------------------------------

def teardown_module(module):
    if not DB_OK:
        return
    from research_fleet.research_agent import research_db as rdb
    for table, col in (
        ("research_hypotheses", "agent_id"),
        ("research_experiments", "agent_id"),
        ("research_findings", "agent_id"),
        ("research_activity", "agent_id"),
    ):
        try:
            rdb.execute(f"DELETE FROM {table} WHERE {col}=%s", (TEST_AGENT,))
        except Exception:
            pass
    # assert nothing left behind under the sentinel
    for table in ("research_hypotheses", "research_experiments", "research_findings",
                  "research_activity"):
        n = rdb.query(f"SELECT COUNT(*) c FROM {table} WHERE agent_id=%s", (TEST_AGENT,))[0]["c"]
        assert n == 0, f"{table} still has {n} rows for {TEST_AGENT}"
