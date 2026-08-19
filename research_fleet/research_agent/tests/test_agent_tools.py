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
        "ask_analyst",
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
# ask_analyst — the Aura Analyst tool (no DB / no network; analyst is stubbed)
# ---------------------------------------------------------------------------

def test_ask_analyst_unconfigured_returns_available_false(monkeypatch):
    # Aura not configured => honest {available: False}, no fabricated SQL, no raise.
    monkeypatch.setattr(at.analyst, "available", lambda: False)
    # audit must not be attempted (and even if it were, must not raise) — stub it.
    monkeypatch.setattr(at.wt, "record_analyst_query", lambda **k: {"ok": True})
    out = at.dispatch("ask_analyst", {"question": "what is the average sharpe?"},
                      agent_id="unit")
    assert isinstance(out, dict)
    assert out.get("available") is False
    assert out.get("sql") is None
    assert "error" not in out  # not-configured is a clean skip, not an error


def test_ask_analyst_missing_question_returns_error(monkeypatch):
    monkeypatch.setattr(at.analyst, "available", lambda: True)
    out = at.dispatch("ask_analyst", {"question": "   "}, agent_id="unit")
    assert isinstance(out, dict) and "error" in out


def test_ask_analyst_shapes_result_and_audits(monkeypatch):
    # Stub a configured Aura that returns a real-looking flattened payload.
    monkeypatch.setattr(at.analyst, "available", lambda: True)

    def fake_ask(question, output_modes=None, agent_id="", **k):
        return {"sql": "SELECT strategy_family, AVG(sharpe) FROM research_experiments GROUP BY 1",
                "confidence": 0.91, "tables_used": ["research_experiments"],
                "columns": ["strategy_family", "avg_sharpe"],
                "rows": [["momentum", 0.42], ["low_vol", 0.55]],
                "row_count": 2, "text": "low_vol has the highest average sharpe.",
                "error": None, "cached": False, "latency_ms": 123.4}

    captured = {}

    def fake_audit(**kw):
        captured.update(kw)
        return {"ok": True, "id": "aq-x"}

    monkeypatch.setattr(at.analyst, "ask", fake_ask)
    monkeypatch.setattr(at.wt, "record_analyst_query", fake_audit)

    out = at.dispatch("ask_analyst",
                      {"question": "avg sharpe by family", "output_modes": ["sql", "data", "text"]},
                      agent_id="unit-agent", task_id="task-1")
    assert out["available"] is True
    assert out["error"] is None
    assert out["sql"].startswith("SELECT")
    assert out["row_count"] == 2
    assert out["rows"] == [["momentum", 0.42], ["low_vol", 0.55]]
    assert out["text"].startswith("low_vol")
    # audited with the host-injected identity (not spoofable by the model)
    assert captured.get("agent_id") == "unit-agent"
    assert captured.get("task_id") == "task-1"
    assert captured.get("status") == "ok"
    assert "research_experiments" in (captured.get("generated_sql") or "")


def test_ask_analyst_upstream_exception_is_caught_and_audited(monkeypatch):
    monkeypatch.setattr(at.analyst, "available", lambda: True)

    def boom(*a, **k):
        raise RuntimeError("aura upstream exploded")

    audited = {}
    monkeypatch.setattr(at.analyst, "ask", boom)
    monkeypatch.setattr(at.wt, "record_analyst_query",
                        lambda **kw: audited.update(kw) or {"ok": True})

    out = at.dispatch("ask_analyst", {"question": "q"}, agent_id="unit")
    # never raises: returns a readable error result
    assert isinstance(out, dict)
    assert out.get("available") is True
    assert "aura upstream exploded" in out.get("error", "")
    assert audited.get("status") == "error"


def test_ask_analyst_caps_returned_rows(monkeypatch):
    monkeypatch.setattr(at.analyst, "available", lambda: True)
    big = [[i, i * 1.0] for i in range(at.MAX_ANALYST_ROWS + 25)]

    def fake_ask(question, output_modes=None, agent_id="", **k):
        return {"sql": "SELECT 1", "rows": big, "row_count": len(big),
                "text": None, "error": None, "latency_ms": 1.0}

    monkeypatch.setattr(at.analyst, "ask", fake_ask)
    monkeypatch.setattr(at.wt, "record_analyst_query", lambda **k: {"ok": True})

    out = at.dispatch("ask_analyst", {"question": "q"}, agent_id="unit")
    assert len(out["rows"]) == at.MAX_ANALYST_ROWS
    assert out["rows_truncated"] is True
    # the true row_count is still reported honestly
    assert out["row_count"] == len(big)


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
