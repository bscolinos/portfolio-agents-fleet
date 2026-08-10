"""SingleStore write/read API for the auto-research agents.

Every research agent (OpenClaw-through-NemoClaw, one per tiny EC2) uses this to:
  * register itself + heartbeat,
  * atomically CLAIM a task from the shared queue (so N agents don't collide),
  * log activity, hypotheses, experiments, and findings OVER TIME,
  * recall prior findings semantically (Qwen VECTOR(1024) + `<*>` DOT_PRODUCT)
    so it builds on what the fleet already learned.

Config comes from the demo's own ``.env`` (SingleStore + Qwen). Uses the
``singlestoredb`` driver, matching the rest of the demo stack.
"""

from __future__ import annotations

import json
import os
import socket
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import singlestoredb as s2
from openai import OpenAI


# --------------------------------------------------------------------------
# Config (.env loader; works on the EC2 box and locally)
# --------------------------------------------------------------------------

def _load_env() -> None:
    here = Path(__file__).resolve()
    for cand in [here.parent / ".env", here.parent.parent / ".env",
                 Path.cwd() / ".env", Path("/opt/research-agent/.env")]:
        if cand.is_file():
            for line in cand.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            break


_load_env()

HOST = os.environ.get("SINGLESTORE_HOST", "")
PORT = int(os.environ.get("SINGLESTORE_PORT", "3306"))
USER = os.environ.get("SINGLESTORE_USER", "admin")
PASSWORD = os.environ.get("SINGLESTORE_PASSWORD", "")
DATABASE = os.environ.get("SINGLESTORE_DATABASE", "portfolio_agents")
QWEN_KEY = os.environ.get("QWEN_KEY", "")
LLM_ENDPOINT = os.environ.get("LLM_ENDPOINT", "")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "")


def connect(results_type: str = "dicts"):
    return s2.connect(host=HOST, port=PORT, user=USER, password=PASSWORD,
                      database=DATABASE, results_type=results_type, autocommit=True)


def query(sql: str, params: Sequence[Any] | None = None) -> list[dict]:
    conn = connect("dicts")
    try:
        cur = conn.cursor(); cur.execute(sql, tuple(params) if params else None)
        return list(cur.fetchall())
    finally:
        conn.close()


def execute(sql: str, params: Sequence[Any] | None = None) -> int:
    conn = connect("tuples")
    try:
        cur = conn.cursor(); return cur.execute(sql, tuple(params) if params else None) or 0
    finally:
        conn.close()


def executemany(sql: str, rows: Sequence[Sequence[Any]]) -> int:
    if not rows:
        return 0
    conn = connect("tuples")
    try:
        cur = conn.cursor(); cur.executemany(sql, [tuple(r) for r in rows]); return len(rows)
    finally:
        conn.close()


# --------------------------------------------------------------------------
# Embeddings + semantic recall (Qwen VECTOR(1024))
# --------------------------------------------------------------------------

_client: OpenAI | None = None


def _embed_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=QWEN_KEY, base_url=LLM_ENDPOINT)
    return _client


def embed(text: str) -> list[float]:
    r = _embed_client().embeddings.create(model=EMBEDDING_MODEL, input=[text])
    return r.data[0].embedding


def vec_literal(vec: Sequence[float]) -> str:
    return "[" + ",".join(f"{x:.8g}" for x in vec) + "]"


def _uid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


# --------------------------------------------------------------------------
# Agent registration + heartbeat
# --------------------------------------------------------------------------

def register_agent(agent_id: str, display_name: str, focus_area: str, *,
                   persona: str = "", model: str = "", instance_id: str = "",
                   private_ip: str = "", az: str = "") -> None:
    execute(
        """INSERT INTO research_agents
           (agent_id, display_name, focus_area, persona, runner, model,
            instance_id, private_ip, az, status, heartbeat_at, created_at)
           VALUES (%s,%s,%s,%s,'openclaw-nemoclaw',%s,%s,%s,%s,'active',NOW(6),NOW(6))
           ON DUPLICATE KEY UPDATE display_name=VALUES(display_name),
             focus_area=VALUES(focus_area), persona=VALUES(persona),
             model=VALUES(model), instance_id=VALUES(instance_id),
             private_ip=VALUES(private_ip), az=VALUES(az), status='active',
             heartbeat_at=NOW(6)""",
        (agent_id, display_name, focus_area, persona, model, instance_id, private_ip, az),
    )


def heartbeat(agent_id: str, status: str = "active") -> None:
    execute("UPDATE research_agents SET heartbeat_at=NOW(6), status=%s WHERE agent_id=%s",
            (status, agent_id))


# --------------------------------------------------------------------------
# Atomic task claim from the shared queue
# --------------------------------------------------------------------------

def claim_task(agent_id: str, focus_area: str | None = None) -> dict | None:
    """Atomically claim one pending task (optionally matching focus_area first).

    Uses a guarded UPDATE ... WHERE status='pending' so two agents can't grab the
    same row: the row whose claimed_by we successfully set is ours.
    """
    for fa_clause, fa_params in ([("AND focus_area=%s", [focus_area])] if focus_area else []) + [("", [])]:
        cand = query(
            f"SELECT task_id FROM research_tasks WHERE status='pending' {fa_clause} "
            f"ORDER BY priority ASC, created_at ASC LIMIT 5", fa_params)
        for row in cand:
            tid = row["task_id"]
            n = execute(
                "UPDATE research_tasks SET status='claimed', claimed_by=%s, claimed_at=NOW(6) "
                "WHERE task_id=%s AND status='pending'", (agent_id, tid))
            if n == 1:
                got = query("SELECT * FROM research_tasks WHERE task_id=%s", (tid,))
                return got[0] if got else None
    return None


def set_task_status(task_id: str, status: str, result_summary: str | None = None) -> None:
    if status in ("done", "failed"):
        execute("UPDATE research_tasks SET status=%s, finished_at=NOW(6), result_summary=%s WHERE task_id=%s",
                (status, (result_summary or "")[:4000], task_id))
    else:
        execute("UPDATE research_tasks SET status=%s WHERE task_id=%s", (status, task_id))


# --------------------------------------------------------------------------
# Activity / hypotheses / experiments / findings
# --------------------------------------------------------------------------

def log_activity(agent_id: str, phase: str, *, task_id: str | None = None,
                 detail: dict | None = None, tokens_in: int = 0, tokens_out: int = 0) -> None:
    execute(
        """INSERT INTO research_activity
           (activity_id, ts, agent_id, task_id, phase, detail, tokens_in, tokens_out)
           VALUES (%s, NOW(6), %s,%s,%s,%s,%s,%s)""",
        (_uid("act"), agent_id, task_id, phase, json.dumps(detail or {}), tokens_in, tokens_out),
    )


def add_hypothesis(agent_id: str, statement: str, *, task_id: str | None = None,
                   rationale: str = "", strategy_family: str = "", params: dict | None = None,
                   confidence: float = 0.5) -> str:
    hid = _uid("hyp")
    execute(
        """INSERT INTO research_hypotheses
           (hypothesis_id, agent_id, task_id, statement, rationale, strategy_family,
            params, status, confidence, created_at)
           VALUES (%s,%s,%s,%s,%s,%s,%s,'open',%s,NOW(6))""",
        (hid, agent_id, task_id, statement, rationale, strategy_family,
         json.dumps(params or {}), float(confidence)),
    )
    return hid


def record_experiment(agent_id: str, params: dict, metrics: dict, *,
                      hypothesis_id: str | None = None, task_id: str | None = None,
                      strategy_family: str = "", universe: str = "", method: str = "python-backtest",
                      engine: str = "cpu", lookback_start: str | None = None,
                      lookback_end: str | None = None, status: str = "ok",
                      error: str | None = None) -> str:
    eid = _uid("exp")
    execute(
        """INSERT INTO research_experiments
           (experiment_id, agent_id, hypothesis_id, task_id, strategy_family, universe,
            lookback_start, lookback_end, params, ann_return, ann_vol, sharpe, sortino,
            max_drawdown, turnover, cvar_95, win_rate, n_rebalances, benchmark_sharpe,
            beats_benchmark, method, engine, status, error, started_at, finished_at)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s, %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, %s,%s,%s,%s, NOW(6),NOW(6))""",
        (eid, agent_id, hypothesis_id, task_id, strategy_family, universe,
         lookback_start, lookback_end, json.dumps(params),
         metrics.get("ann_return"), metrics.get("ann_vol"), metrics.get("sharpe"),
         metrics.get("sortino"), metrics.get("max_drawdown"), metrics.get("turnover"),
         metrics.get("cvar_95"), metrics.get("win_rate"), metrics.get("n_rebalances"),
         metrics.get("benchmark_sharpe"),
         1 if metrics.get("beats_benchmark") else 0, method, engine, status, error),
    )
    return eid


def write_finding(agent_id: str, content: str, *, title: str = "", kind: str = "finding",
                  task_id: str | None = None, experiment_id: str | None = None,
                  hypothesis_id: str | None = None, strategy_family: str = "",
                  metrics: dict | None = None, importance: float = 0.6,
                  tags: list | None = None) -> str:
    fid = _uid("fnd")
    emb = embed(content)
    execute(
        """INSERT INTO research_findings
           (finding_id, agent_id, task_id, experiment_id, hypothesis_id, kind, title,
            content, embedding, strategy_family, metrics, importance, tags, created_at)
           VALUES (%s,%s,%s,%s,%s,%s,%s, %s, %s, %s,%s,%s,%s, NOW(6))""",
        (fid, agent_id, task_id, experiment_id, hypothesis_id, kind, title[:256],
         content, vec_literal(emb), strategy_family, json.dumps(metrics or {}),
         float(importance), json.dumps(tags or [])),
    )
    return fid


def recall_findings(query_text: str, *, k: int = 5, strategy_family: str | None = None,
                    agent_id: str | None = None) -> list[dict]:
    """Semantic recall of prior findings across the WHOLE fleet (shared knowledge)."""
    qvec = vec_literal(embed(query_text))
    where = []
    params: list[Any] = [qvec]
    if strategy_family:
        where.append("strategy_family=%s"); params.append(strategy_family)
    if agent_id:
        where.append("agent_id=%s"); params.append(agent_id)
    wc = ("WHERE " + " AND ".join(where)) if where else ""
    params.append(qvec); params.append(int(k))
    return query(
        f"""SELECT finding_id, agent_id, kind, title, content, strategy_family,
                   metrics, importance, created_at, (embedding <*> %s) AS score
            FROM research_findings {wc}
            ORDER BY (embedding <*> %s) * (0.7 + 0.3*importance) DESC LIMIT %s""",
        params,
    )


def record_analyst_query(agent_id: str, question: str, *, task_id: str | None = None,
                         generated_sql: str = "", row_count: int = 0, answer: str = "",
                         latency_ms: float = 0.0, status: str = "ok") -> str:
    qid = _uid("aq")
    execute(
        """INSERT INTO research_analyst_queries
           (query_id, agent_id, task_id, question, generated_sql, row_count, answer,
            latency_ms, status, created_at)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s, NOW(6))""",
        (qid, agent_id, task_id, question, generated_sql[:8000], row_count,
         answer[:8000], latency_ms, status),
    )
    return qid
