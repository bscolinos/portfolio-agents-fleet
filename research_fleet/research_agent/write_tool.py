"""Templated SingleStore write tool — the ONE sanctioned way research agents
persist results, so every row is uniform regardless of which agent/node writes it.

Every write goes through a validating template that:
  * enforces required fields + controlled vocabularies (enums) — a bad value is
    rejected with a clear error, never silently written,
  * coerces numerics safely (None-tolerant; strings -> float),
  * auto-generates the primary-key id, the created/started timestamps, and the
    Qwen VECTOR(1024) embedding for findings,
  * clamps free-text lengths and normalizes JSON columns,
  * returns a small, uniform receipt `{ok, id, table, ...}`.

Exposed three ways, all backed by the same validators:
  1. Python:      `from research_agent.write_tool import TOOLS; TOOLS["write_finding"](**kw)`
  2. Dispatch:    `call_tool(name, payload_dict)` — used by the HTTP tool server
  3. CLI:         `python -m research_agent.write_tool <tool> '<json>'`

The HTTP tool server (`serve_http`) lets the OpenClaw-in-OpenShell sandbox agent
write via `POST http://host.openshell.internal:11510/tool/<name>` (the same
trusted host bridge used for inference) — no raw SingleStore :3306 egress from
the sandbox, and the schema is enforced host-side so the model literally cannot
write a nonconforming row.
"""

from __future__ import annotations

import json
import re
import sys
import uuid
from typing import Any, Callable

from . import research_db as rdb


# ---------------------------------------------------------------------------
# Controlled vocabularies — the uniformity backbone
# ---------------------------------------------------------------------------

STRATEGY_FAMILIES = {
    "equal_weight", "momentum", "mean_reversion", "vol_target",
    "low_vol", "factor", "risk_parity", "regime",
}
FINDING_KINDS = {"finding", "insight", "caveat", "next_step"}
HYPOTHESIS_STATUS = {"open", "testing", "supported", "rejected", "inconclusive"}
EXPERIMENT_METHODS = {"python-backtest", "aura-analyst", "llm-analysis"}
EXPERIMENT_STATUS = {"ok", "failed"}
ACTIVITY_PHASES = {
    "START", "RECALL", "HYPOTHESIS", "EXPERIMENT", "FINDING",
    "ANALYST", "HEARTBEAT", "END", "ERROR", "NOTE",
}
# canonical metric keys an experiment may carry (anything else is dropped)
METRIC_KEYS = {
    "ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown", "turnover",
    "cvar_95", "win_rate", "n_rebalances", "benchmark_sharpe", "beats_benchmark",
    "total_return", "universe_size",
}


class ToolError(ValueError):
    """Raised when a write is rejected for failing validation."""


# ---------------------------------------------------------------------------
# Coercion / validation helpers
# ---------------------------------------------------------------------------

def _req(payload: dict, key: str) -> Any:
    v = payload.get(key)
    if v is None or (isinstance(v, str) and not v.strip()):
        raise ToolError(f"missing required field: '{key}'")
    return v


def _enum(value: Any, allowed: set[str], field: str, default: str | None = None) -> str:
    if value is None and default is not None:
        return default
    s = str(value).strip().lower().replace("-", "_") if value is not None else ""
    # map a few common aliases
    s = {"meanreversion": "mean_reversion", "riskparity": "risk_parity",
         "voltarget": "vol_target", "lowvol": "low_vol", "equalweight": "equal_weight"}.get(s, s)
    if s not in allowed:
        if default is not None:
            return default
        raise ToolError(f"invalid {field}='{value}'. allowed: {sorted(allowed)}")
    return s


def _num(v: Any):
    if v is None or v == "":
        return None
    try:
        f = float(v)
        return f if f == f else None  # drop NaN
    except (TypeError, ValueError):
        return None


def _text(v: Any, limit: int) -> str:
    if v is None:
        return ""
    s = v if isinstance(v, str) else json.dumps(v)
    return s[:limit]


def _obj(v: Any) -> dict:
    if v is None:
        return {}
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except Exception:
            return {"_raw": v[:2000]}
    return v if isinstance(v, dict) else {"_value": v}


def _list(v: Any) -> list:
    if v is None:
        return []
    if isinstance(v, str):
        try:
            j = json.loads(v)
            return j if isinstance(j, list) else [v]
        except Exception:
            return [t.strip() for t in re.split(r"[,\s]+", v) if t.strip()]
    return list(v) if isinstance(v, (list, tuple)) else [v]


def _clean_metrics(m: Any) -> dict:
    m = _obj(m)
    out: dict = {}
    for k in METRIC_KEYS:
        if k not in m:
            continue
        if k == "beats_benchmark":
            out[k] = bool(m[k])
        elif k in ("n_rebalances", "universe_size"):
            n = _num(m[k]); out[k] = int(n) if n is not None else None
        else:
            out[k] = _num(m[k])
    return out


def _uid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# The templated write functions (one per record type)
# ---------------------------------------------------------------------------

def write_hypothesis(*, agent_id: str, statement: str, strategy_family: str,
                     rationale: str = "", params: dict | None = None,
                     confidence: float | None = 0.5, task_id: str | None = None,
                     **_ignore) -> dict:
    agent_id = _req({"agent_id": agent_id}, "agent_id")
    statement = _text(_req({"statement": statement}, "statement"), 4000)
    fam = _enum(strategy_family, STRATEGY_FAMILIES, "strategy_family")
    conf = _num(confidence)
    conf = 0.5 if conf is None else max(0.0, min(1.0, conf))
    hid = _uid("hyp")
    rdb.execute(
        """INSERT INTO research_hypotheses
           (hypothesis_id, agent_id, task_id, statement, rationale, strategy_family,
            params, status, confidence, created_at)
           VALUES (%s,%s,%s,%s,%s,%s,%s,'open',%s,NOW(6))""",
        (hid, agent_id, task_id, statement, _text(rationale, 4000), fam,
         json.dumps(_obj(params)), conf))
    return {"ok": True, "table": "research_hypotheses", "id": hid,
            "strategy_family": fam, "confidence": conf}


def write_experiment(*, agent_id: str, strategy_family: str, params: dict,
                     metrics: dict | None = None, hypothesis_id: str | None = None,
                     task_id: str | None = None, universe: str = "",
                     method: str = "python-backtest", engine: str = "cpu",
                     lookback_start: str | None = None, lookback_end: str | None = None,
                     status: str = "ok", error: str | None = None, **_ignore) -> dict:
    agent_id = _req({"agent_id": agent_id}, "agent_id")
    fam = _enum(strategy_family, STRATEGY_FAMILIES, "strategy_family")
    method = _enum(method, EXPERIMENT_METHODS, "method", default="python-backtest")
    status = _enum(status, EXPERIMENT_STATUS, "status", default="ok")
    m = _clean_metrics(metrics)
    eid = _uid("exp")
    rdb.execute(
        """INSERT INTO research_experiments
           (experiment_id, agent_id, hypothesis_id, task_id, strategy_family, universe,
            lookback_start, lookback_end, params, ann_return, ann_vol, sharpe, sortino,
            max_drawdown, turnover, cvar_95, win_rate, n_rebalances, benchmark_sharpe,
            beats_benchmark, method, engine, status, error, started_at, finished_at)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s, %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, %s,%s,%s,%s, NOW(6),NOW(6))""",
        (eid, agent_id, hypothesis_id, task_id, fam, _text(universe, 64),
         lookback_start, lookback_end, json.dumps(_obj(params)),
         m.get("ann_return"), m.get("ann_vol"), m.get("sharpe"), m.get("sortino"),
         m.get("max_drawdown"), m.get("turnover"), m.get("cvar_95"), m.get("win_rate"),
         m.get("n_rebalances"), m.get("benchmark_sharpe"),
         1 if m.get("beats_benchmark") else 0, method, _text(engine, 24)[:24] or "cpu",
         status, _text(error, 2000) if error else None))
    return {"ok": True, "table": "research_experiments", "id": eid,
            "strategy_family": fam, "sharpe": m.get("sharpe"),
            "beats_benchmark": bool(m.get("beats_benchmark"))}


def write_finding(*, agent_id: str, content: str, strategy_family: str,
                  title: str = "", kind: str = "finding", metrics: dict | None = None,
                  experiment_id: str | None = None, hypothesis_id: str | None = None,
                  task_id: str | None = None, importance: float | None = None,
                  tags: list | None = None, **_ignore) -> dict:
    agent_id = _req({"agent_id": agent_id}, "agent_id")
    content = _text(_req({"content": content}, "content"), 12000)
    fam = _enum(strategy_family, STRATEGY_FAMILIES, "strategy_family")
    kind = _enum(kind, FINDING_KINDS, "kind", default="finding")
    m = _clean_metrics(metrics)
    # importance: explicit, else derived from whether it beat the benchmark
    imp = _num(importance)
    if imp is None:
        imp = 0.8 if m.get("beats_benchmark") else 0.55
    imp = max(0.0, min(1.0, imp))
    fid = _uid("fnd")
    emb = rdb.embed(content)  # auto Qwen embedding — uniform across the fleet
    rdb.execute(
        """INSERT INTO research_findings
           (finding_id, agent_id, task_id, experiment_id, hypothesis_id, kind, title,
            content, embedding, strategy_family, metrics, importance, tags, created_at)
           VALUES (%s,%s,%s,%s,%s,%s,%s, %s, %s, %s,%s,%s,%s, NOW(6))""",
        (fid, agent_id, task_id, experiment_id, hypothesis_id, kind, _text(title, 256),
         content, rdb.vec_literal(emb), fam, json.dumps(m), imp,
         json.dumps([_text(t, 48) for t in _list(tags)][:12])))
    return {"ok": True, "table": "research_findings", "id": fid,
            "strategy_family": fam, "kind": kind, "importance": imp}


def write_activity(*, agent_id: str, phase: str, detail: dict | None = None,
                   task_id: str | None = None, tokens_in: int = 0,
                   tokens_out: int = 0, **_ignore) -> dict:
    agent_id = _req({"agent_id": agent_id}, "agent_id")
    phase = _enum_upper(phase)  # ACTIVITY_PHASES are upper-case; normalize + validate
    ti = _num(tokens_in) or 0; to = _num(tokens_out) or 0
    aid = _uid("act")
    rdb.execute(
        """INSERT INTO research_activity
           (activity_id, ts, agent_id, task_id, phase, detail, tokens_in, tokens_out)
           VALUES (%s, NOW(6), %s,%s,%s,%s,%s,%s)""",
        (aid, agent_id, task_id, phase, json.dumps(_obj(detail)), int(ti), int(to)))
    return {"ok": True, "table": "research_activity", "id": aid, "phase": phase}


def _enum_upper(value: Any) -> str:
    s = str(value).strip().upper()
    return s if s in ACTIVITY_PHASES else "NOTE"


def set_hypothesis_status(*, hypothesis_id: str, status: str, **_ignore) -> dict:
    st = _enum(status, HYPOTHESIS_STATUS, "status")
    n = rdb.execute("UPDATE research_hypotheses SET status=%s WHERE hypothesis_id=%s",
                    (st, _req({"hypothesis_id": hypothesis_id}, "hypothesis_id")))
    return {"ok": True, "table": "research_hypotheses", "id": hypothesis_id,
            "status": st, "updated": n}


def record_analyst_query(*, agent_id: str, question: str, generated_sql: str = "",
                         row_count: int = 0, answer: str = "", latency_ms: float = 0.0,
                         status: str = "ok", task_id: str | None = None, **_ignore) -> dict:
    agent_id = _req({"agent_id": agent_id}, "agent_id")
    qid = _uid("aq")
    rdb.execute(
        """INSERT INTO research_analyst_queries
           (query_id, agent_id, task_id, question, generated_sql, row_count, answer,
            latency_ms, status, created_at)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s, NOW(6))""",
        (qid, agent_id, task_id, _text(_req({"question": question}, "question"), 4000),
         _text(generated_sql, 8000), int(_num(row_count) or 0), _text(answer, 8000),
         _num(latency_ms) or 0.0, _text(status, 24) or "ok"))
    return {"ok": True, "table": "research_analyst_queries", "id": qid}


# ---------------------------------------------------------------------------
# Dispatch + machine-readable tool schema (for the OpenClaw skill / HTTP server)
# ---------------------------------------------------------------------------

TOOLS: dict[str, Callable[..., dict]] = {
    "write_hypothesis": write_hypothesis,
    "write_experiment": write_experiment,
    "write_finding": write_finding,
    "write_activity": write_activity,
    "set_hypothesis_status": set_hypothesis_status,
    "record_analyst_query": record_analyst_query,
}

TOOL_SCHEMA = {
    "write_hypothesis": {
        "required": ["agent_id", "statement", "strategy_family"],
        "optional": ["rationale", "params", "confidence", "task_id"],
        "enums": {"strategy_family": sorted(STRATEGY_FAMILIES)},
        "desc": "Record a testable hypothesis before experimenting.",
    },
    "write_experiment": {
        "required": ["agent_id", "strategy_family", "params"],
        "optional": ["metrics", "hypothesis_id", "task_id", "universe", "method",
                     "engine", "lookback_start", "lookback_end", "status", "error"],
        "enums": {"strategy_family": sorted(STRATEGY_FAMILIES),
                  "method": sorted(EXPERIMENT_METHODS), "status": sorted(EXPERIMENT_STATUS)},
        "metrics_keys": sorted(METRIC_KEYS),
        "desc": "Record a backtest/experiment run with its metrics (vs 1/N benchmark).",
    },
    "write_finding": {
        "required": ["agent_id", "content", "strategy_family"],
        "optional": ["title", "kind", "metrics", "experiment_id", "hypothesis_id",
                     "task_id", "importance", "tags"],
        "enums": {"strategy_family": sorted(STRATEGY_FAMILIES), "kind": sorted(FINDING_KINDS)},
        "desc": "Record a durable, quantitative finding (auto-embedded for fleet recall).",
    },
    "write_activity": {
        "required": ["agent_id", "phase"],
        "optional": ["detail", "task_id", "tokens_in", "tokens_out"],
        "enums": {"phase": sorted(ACTIVITY_PHASES)},
        "desc": "Append a step to the activity log.",
    },
    "set_hypothesis_status": {
        "required": ["hypothesis_id", "status"],
        "optional": [], "enums": {"status": sorted(HYPOTHESIS_STATUS)},
        "desc": "Update a hypothesis's status after evaluating it.",
    },
    "record_analyst_query": {
        "required": ["agent_id", "question"],
        "optional": ["generated_sql", "row_count", "answer", "latency_ms", "status", "task_id"],
        "enums": {}, "desc": "Record an Aura Analyst NL query + result over SingleStore.",
    },
}


def call_tool(name: str, payload: dict) -> dict:
    if name not in TOOLS:
        raise ToolError(f"unknown tool '{name}'. tools: {sorted(TOOLS)}")
    if not isinstance(payload, dict):
        raise ToolError("payload must be a JSON object")
    try:
        return TOOLS[name](**payload)
    except TypeError as e:
        # missing/unexpected kwargs -> uniform ToolError with the schema hint
        spec = TOOL_SCHEMA.get(name, {})
        raise ToolError(f"{e}. required={spec.get('required')} optional={spec.get('optional')}") from None


# ---------------------------------------------------------------------------
# CLI: python -m research_agent.write_tool <tool> '<json payload>'
# ---------------------------------------------------------------------------

def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help", "schema"):
        print(json.dumps(TOOL_SCHEMA, indent=2)); return
    name = argv[0]
    payload = json.loads(argv[1]) if len(argv) > 1 else json.loads(sys.stdin.read() or "{}")
    try:
        print(json.dumps(call_tool(name, payload)))
    except ToolError as e:
        print(json.dumps({"ok": False, "error": str(e)})); sys.exit(1)


if __name__ == "__main__":
    main()
