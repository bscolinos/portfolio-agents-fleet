"""Portfolio Agents — dashboard read API.

All endpoints are mounted under ``/api`` on the FastAPI ``app`` in ``main.py``.
Everything here is read-only (SELECTs) over the SingleStore tables in
``schema.sql``. Tables may be EMPTY before the agent fleet has run — every
endpoint degrades to empty arrays / zeroed tiles rather than 500-ing.

The one live-compute endpoint is ``/api/memory/recall``: it embeds the query
via the SingleStore-hosted Qwen endpoint and ranks stored memories with the
native ``<*>`` (DOT_PRODUCT) vector operator — the demo's headline feature.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException, Query

import embeddings
import singlestore

router = APIRouter(prefix="/api")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _rows(sql: str, params: list[Any] | None = None) -> list[dict]:
    """Run a query, returning [] on any error so the UI never 500s on empty
    or not-yet-populated tables."""
    try:
        return singlestore.query(sql, params)
    except Exception:  # noqa: BLE001
        return []


def _jparse(val: Any) -> Any:
    """Parse a JSON column that may arrive as str / bytes / already-parsed / None."""
    if val is None:
        return None
    if isinstance(val, (dict, list)):
        return val
    if isinstance(val, (bytes, bytearray)):
        val = val.decode("utf-8", "replace")
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (ValueError, TypeError):
            return val
    return val


def _f(val: Any) -> float | None:
    """Coerce to float, tolerating None / Decimal / str."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _iso(val: Any) -> Any:
    """Render datetimes/dates as ISO strings; pass through otherwise."""
    if val is None:
        return None
    isof = getattr(val, "isoformat", None)
    return isof() if callable(isof) else val


def _vec_literal(vec: list[float]) -> str:
    """Render a float list as a SingleStore VECTOR JSON literal (matches
    pa_agents/db.vec_literal)."""
    return "[" + ",".join(f"{x:.8g}" for x in vec) + "]"


# --------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------

@router.get("/health")
def api_health() -> dict:
    import os
    db = os.environ.get("SINGLESTORE_DATABASE", "")
    try:
        ping = singlestore.ping()
        return {"ok": bool(ping.get("ok")), "db_version": ping.get("version"), "db": db}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "db_version": None, "db": db, "error": str(e)}


# --------------------------------------------------------------------------
# Global money-moment tiles
# --------------------------------------------------------------------------

@router.get("/stats")
def api_stats() -> dict:
    out = {
        "n_agents": 0,
        "n_trades": 0,
        "total_notional": 0.0,
        "gpu_solves": 0,
        "cpu_solves": 0,
        "avg_solve_ms": 0.0,
        "total_memories": 0,
        "total_audit_events": 0,
        "universe_size": 0,
        "as_of": None,
    }

    r = _rows("SELECT COUNT(*) AS n FROM agents WHERE status = 'active'")
    if r:
        out["n_agents"] = int(r[0]["n"] or 0)

    r = _rows("SELECT COUNT(*) AS n, COALESCE(SUM(ABS(notional)), 0) AS notional FROM executions")
    if r:
        out["n_trades"] = int(r[0]["n"] or 0)
        out["total_notional"] = _f(r[0]["notional"]) or 0.0

    r = _rows(
        """
        SELECT
          SUM(CASE WHEN engine = 'gpu' THEN 1 ELSE 0 END) AS gpu,
          SUM(CASE WHEN engine = 'cpu' THEN 1 ELSE 0 END) AS cpu,
          AVG(solve_ms) AS avg_ms
        FROM agent_runs
        WHERE status = 'ok'
        """
    )
    if r:
        out["gpu_solves"] = int(r[0]["gpu"] or 0)
        out["cpu_solves"] = int(r[0]["cpu"] or 0)
        out["avg_solve_ms"] = _f(r[0]["avg_ms"]) or 0.0

    r = _rows("SELECT COUNT(*) AS n FROM agent_memory")
    if r:
        out["total_memories"] = int(r[0]["n"] or 0)

    r = _rows("SELECT COUNT(*) AS n FROM trade_audit")
    if r:
        out["total_audit_events"] = int(r[0]["n"] or 0)

    r = _rows("SELECT COUNT(*) AS n FROM securities WHERE is_active = 1")
    if r:
        out["universe_size"] = int(r[0]["n"] or 0)

    r = _rows("SELECT MAX(as_of_date) AS d FROM nav_history")
    if r:
        out["as_of"] = _iso(r[0]["d"])

    return out


# --------------------------------------------------------------------------
# Agent roster
# --------------------------------------------------------------------------

_AGENT_SQL = """
SELECT
  a.agent_id, a.display_name, a.strategy_type, a.objective, a.engine, a.color,
  nav.nav          AS latest_nav,
  nav.cum_return   AS cum_return,
  nav.daily_return AS daily_return,
  nav.as_of_date   AS nav_as_of,
  rk.sharpe        AS sharpe,
  pos.n_positions  AS n_positions,
  run.finished_at  AS last_run_at,
  run.engine       AS last_engine,
  run.gpu_name     AS last_gpu_name,
  run.avg_solve_ms AS avg_solve_ms
FROM agents a
LEFT JOIN (
    SELECT n1.agent_id, n1.nav, n1.cum_return, n1.daily_return, n1.as_of_date
    FROM nav_history n1
    JOIN (SELECT agent_id, MAX(as_of_date) AS d FROM nav_history GROUP BY agent_id) m
      ON n1.agent_id = m.agent_id AND n1.as_of_date = m.d
) nav ON nav.agent_id = a.agent_id
LEFT JOIN (
    SELECT agent_id, COUNT(*) AS n_positions
    FROM positions WHERE ABS(qty) > 1e-9 GROUP BY agent_id
) pos ON pos.agent_id = a.agent_id
LEFT JOIN (
    SELECT r1.agent_id, r1.sharpe
    FROM risk_metrics r1
    JOIN (SELECT agent_id, MAX(as_of_date) AS d FROM risk_metrics GROUP BY agent_id) rm
      ON r1.agent_id = rm.agent_id AND r1.as_of_date = rm.d
) rk ON rk.agent_id = a.agent_id
LEFT JOIN (
    SELECT ar.agent_id,
           MAX(ar.finished_at) AS finished_at,
           AVG(ar.solve_ms) AS avg_solve_ms,
           ANY_VALUE(ar.engine) AS engine,
           ANY_VALUE(ar.gpu_name) AS gpu_name
    FROM agent_runs ar
    WHERE ar.status = 'ok'
    GROUP BY ar.agent_id
) run ON run.agent_id = a.agent_id
"""


def _agent_row(r: dict) -> dict:
    return {
        "agent_id": r["agent_id"],
        "display_name": r["display_name"],
        "strategy_type": r["strategy_type"],
        "objective": r["objective"],
        "engine": r["engine"],
        "color": r["color"],
        "latest_nav": _f(r.get("latest_nav")),
        "cum_return": _f(r.get("cum_return")),
        "daily_return": _f(r.get("daily_return")),
        "sharpe": _f(r.get("sharpe")),
        "n_positions": int(r.get("n_positions") or 0),
        "last_run_at": _iso(r.get("last_run_at")),
        "last_engine": r.get("last_engine"),
        "last_gpu_name": r.get("last_gpu_name"),
        "avg_solve_ms": _f(r.get("avg_solve_ms")),
    }


@router.get("/agents")
def api_agents() -> list[dict]:
    rows = _rows(_AGENT_SQL + " ORDER BY a.display_name")
    return [_agent_row(r) for r in rows]


# --------------------------------------------------------------------------
# Leaderboard
# --------------------------------------------------------------------------

@router.get("/leaderboard")
def api_leaderboard() -> list[dict]:
    rows = _rows(_AGENT_SQL)
    # extra risk fields off the latest risk_metrics row per agent
    extra = {}
    for r in _rows(
        """
        SELECT r1.agent_id, r1.turnover, r1.max_drawdown, r1.volatility
        FROM risk_metrics r1
        JOIN (SELECT agent_id, MAX(as_of_date) AS d FROM risk_metrics GROUP BY agent_id) rm
          ON r1.agent_id = rm.agent_id AND r1.as_of_date = rm.d
        """
    ):
        extra[r["agent_id"]] = r

    out = []
    for r in rows:
        base = _agent_row(r)
        ex = extra.get(r["agent_id"], {})
        base["turnover"] = _f(ex.get("turnover"))
        base["max_drawdown"] = _f(ex.get("max_drawdown"))
        base["vol"] = _f(ex.get("volatility"))
        out.append(base)

    out.sort(key=lambda x: (x["cum_return"] is not None, x["cum_return"] or 0.0), reverse=True)
    for i, row in enumerate(out, start=1):
        row["rank"] = i
    return out


# --------------------------------------------------------------------------
# NAV / equity curves
# --------------------------------------------------------------------------

@router.get("/nav")
def api_nav(
    agent: str = Query("all"),
    start: str | None = Query(None),
    end: str | None = Query(None),
) -> dict:
    where = []
    params: list[Any] = []
    if agent and agent != "all":
        where.append("n.agent_id = %s")
        params.append(agent)
    if start:
        where.append("n.as_of_date >= %s")
        params.append(start)
    if end:
        where.append("n.as_of_date <= %s")
        params.append(end)
    clause = (" WHERE " + " AND ".join(where)) if where else ""

    rows = _rows(
        f"""
        SELECT n.agent_id, a.display_name, a.color,
               n.as_of_date, n.nav, n.cum_return, n.daily_return
        FROM nav_history n
        JOIN agents a ON a.agent_id = n.agent_id
        {clause}
        ORDER BY n.agent_id, n.as_of_date
        """,
        params,
    )

    series: dict[str, dict] = {}
    for r in rows:
        aid = r["agent_id"]
        s = series.get(aid)
        if s is None:
            s = {
                "agent_id": aid,
                "display_name": r["display_name"],
                "color": r["color"],
                "points": [],
            }
            series[aid] = s
        s["points"].append({
            "date": _iso(r["as_of_date"]),
            "nav": _f(r["nav"]),
            "cum_return": _f(r["cum_return"]),
            "daily_return": _f(r["daily_return"]),
        })
    return {"series": list(series.values())}


# --------------------------------------------------------------------------
# Positions
# --------------------------------------------------------------------------

@router.get("/positions")
def api_positions(agent: str = Query(...)) -> dict:
    rows = _rows(
        """
        SELECT ticker, qty, avg_cost, last_price, market_value, weight,
               (market_value - qty * avg_cost) AS unrealized_pnl, as_of_date
        FROM positions
        WHERE agent_id = %s AND ABS(qty) > 1e-9
        ORDER BY weight DESC
        """,
        [agent],
    )
    positions = [{
        "ticker": r["ticker"],
        "qty": _f(r["qty"]),
        "avg_cost": _f(r["avg_cost"]),
        "last_price": _f(r["last_price"]),
        "market_value": _f(r["market_value"]),
        "weight": _f(r["weight"]),
        "unrealized_pnl": _f(r["unrealized_pnl"]),
    } for r in rows]

    as_of = _iso(rows[0]["as_of_date"]) if rows else None

    nav = cash = None
    nrow = _rows(
        """
        SELECT nav, cash, as_of_date FROM nav_history
        WHERE agent_id = %s ORDER BY as_of_date DESC LIMIT 1
        """,
        [agent],
    )
    if nrow:
        nav = _f(nrow[0]["nav"])
        cash = _f(nrow[0]["cash"])
        if as_of is None:
            as_of = _iso(nrow[0]["as_of_date"])

    return {
        "agent_id": agent,
        "as_of": as_of,
        "cash": cash,
        "nav": nav,
        "positions": positions,
    }


# --------------------------------------------------------------------------
# Trade blotter (fills)
# --------------------------------------------------------------------------

@router.get("/blotter")
def api_blotter(agent: str = Query("all"), limit: int = Query(100, ge=1, le=1000)) -> list[dict]:
    where = ""
    params: list[Any] = []
    if agent and agent != "all":
        where = "WHERE agent_id = %s"
        params.append(agent)
    params.append(int(limit))
    rows = _rows(
        f"""
        SELECT executed_at, agent_id, ticker, side, fill_qty, fill_price,
               notional, commission, slippage_bps, venue, run_id
        FROM executions
        {where}
        ORDER BY executed_at DESC
        LIMIT %s
        """,
        params,
    )
    return [{
        "executed_at": _iso(r["executed_at"]),
        "agent_id": r["agent_id"],
        "ticker": r["ticker"],
        "side": r["side"],
        "fill_qty": _f(r["fill_qty"]),
        "fill_price": _f(r["fill_price"]),
        "notional": _f(r["notional"]),
        "commission": _f(r["commission"]),
        "slippage_bps": _f(r["slippage_bps"]),
        "venue": r["venue"],
        "run_id": r["run_id"],
    } for r in rows]


# --------------------------------------------------------------------------
# Optimization runs
# --------------------------------------------------------------------------

@router.get("/runs")
def api_runs(agent: str = Query("all"), limit: int = Query(50, ge=1, le=500)) -> list[dict]:
    where = ""
    params: list[Any] = []
    if agent and agent != "all":
        where = "WHERE agent_id = %s"
        params.append(agent)
    params.append(int(limit))
    rows = _rows(
        f"""
        SELECT run_id, agent_id, as_of_date, engine, gpu_name, num_scenarios,
               solve_ms, scenario_ms, universe_size, status, started_at, finished_at
        FROM agent_runs
        {where}
        ORDER BY started_at DESC
        LIMIT %s
        """,
        params,
    )
    return [{
        "run_id": r["run_id"],
        "agent_id": r["agent_id"],
        "as_of_date": _iso(r["as_of_date"]),
        "engine": r["engine"],
        "gpu_name": r["gpu_name"],
        "num_scenarios": int(r["num_scenarios"]) if r["num_scenarios"] is not None else None,
        "solve_ms": _f(r["solve_ms"]),
        "scenario_ms": _f(r["scenario_ms"]),
        "universe_size": int(r["universe_size"]) if r["universe_size"] is not None else None,
        "status": r["status"],
        "started_at": _iso(r["started_at"]),
        "finished_at": _iso(r["finished_at"]),
    } for r in rows]


# --------------------------------------------------------------------------
# Persisted memory feed
# --------------------------------------------------------------------------

@router.get("/memory")
def api_memory(
    agent: str = Query(...),
    kind: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
) -> list[dict]:
    where = ["agent_id = %s"]
    params: list[Any] = [agent]
    if kind:
        where.append("kind = %s")
        params.append(kind)
    params.append(int(limit))
    rows = _rows(
        f"""
        SELECT memory_id, kind, as_of_date, content, importance,
               metrics, tags, created_at
        FROM agent_memory
        WHERE {' AND '.join(where)}
        ORDER BY created_at DESC
        LIMIT %s
        """,
        params,
    )
    return [{
        "memory_id": r["memory_id"],
        "kind": r["kind"],
        "as_of_date": _iso(r["as_of_date"]),
        "content": r["content"],
        "importance": _f(r["importance"]),
        "metrics": _jparse(r["metrics"]),
        "tags": _jparse(r["tags"]),
        "created_at": _iso(r["created_at"]),
    } for r in rows]


# --------------------------------------------------------------------------
# LIVE semantic recall — embeds q via Qwen, ranks by embedding <*> qvec
# (the demo's headline feature; mirrors pa_agents/db.recall_memory)
# --------------------------------------------------------------------------

@router.get("/memory/recall")
def api_memory_recall(
    agent: str = Query(...),
    q: str = Query(..., min_length=1),
    k: int = Query(5, ge=1, le=50),
) -> dict:
    try:
        qvec = _vec_literal(embeddings.embed(q)[0])
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Embedding error: {e}")

    # embedding <*> qvec is DOT_PRODUCT similarity; blend a mild importance boost
    # exactly like pa_agents/db.recall_memory so the ranking matches the fleet's.
    rows = _rows(
        """
        SELECT content, kind, created_at, importance,
               (embedding <*> %s) AS score
        FROM agent_memory
        WHERE agent_id = %s
        ORDER BY (embedding <*> %s) * (0.7 + 0.3 * importance) DESC
        LIMIT %s
        """,
        [qvec, agent, qvec, int(k)],
    )
    results = [{
        "content": r["content"],
        "kind": r["kind"],
        "score": _f(r["score"]),
        "created_at": _iso(r["created_at"]),
        "importance": _f(r["importance"]),
    } for r in rows]
    return {"query": q, "agent_id": agent, "results": results}


# --------------------------------------------------------------------------
# Risk metric series
# --------------------------------------------------------------------------

@router.get("/risk")
def api_risk(agent: str = Query(...), limit: int = Query(100, ge=1, le=1000)) -> list[dict]:
    rows = _rows(
        """
        SELECT as_of_date, exp_return, volatility, sharpe, cvar,
               turnover, n_positions
        FROM risk_metrics
        WHERE agent_id = %s
        ORDER BY as_of_date DESC
        LIMIT %s
        """,
        [agent, int(limit)],
    )
    return [{
        "as_of_date": _iso(r["as_of_date"]),
        "exp_return": _f(r["exp_return"]),
        "volatility": _f(r["volatility"]),
        "sharpe": _f(r["sharpe"]),
        "cvar": _f(r["cvar"]),
        "turnover": _f(r["turnover"]),
        "n_positions": int(r["n_positions"]) if r["n_positions"] is not None else None,
    } for r in rows]


# --------------------------------------------------------------------------
# Compliance audit trail
# --------------------------------------------------------------------------

@router.get("/audit")
def api_audit(
    run_id: str | None = Query(None),
    agent: str | None = Query(None),
    limit: int = Query(100, ge=1, le=1000),
) -> list[dict]:
    where = []
    params: list[Any] = []
    if run_id:
        where.append("run_id = %s")
        params.append(run_id)
    if agent and agent != "all":
        where.append("agent_id = %s")
        params.append(agent)
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    params.append(int(limit))
    rows = _rows(
        f"""
        SELECT ts, agent_id, run_id, event_type, entity_ref, ticker, detail, actor
        FROM trade_audit
        {clause}
        ORDER BY ts DESC
        LIMIT %s
        """,
        params,
    )
    return [{
        "ts": _iso(r["ts"]),
        "agent_id": r["agent_id"],
        "run_id": r["run_id"],
        "event_type": r["event_type"],
        "entity_ref": r["entity_ref"],
        "ticker": r["ticker"],
        "detail": _jparse(r["detail"]),
        "actor": r["actor"],
    } for r in rows]
