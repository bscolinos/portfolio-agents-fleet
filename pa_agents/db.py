"""SingleStore access layer for the portfolio agents.

Everything the agents persist — runs, memory (with Qwen vector embeddings),
orders, executions, positions, NAV, risk, and the compliance audit trail —
flows through here. Connection config is read from the demo's own ``.env``
(loaded by :mod:`pa_agents.config`).

Uses the ``singlestoredb`` driver (same as the demo backend) so there is one
consistent stack on both the GPU fleet host and the API server.

The two headline capabilities:

* :func:`embed` / :func:`embed_batch` — turn natural-language memory into a
  Qwen VECTOR(1024) via the SingleStore-hosted embedding endpoint.
* :func:`recall_memory` — semantic top-k recall of an agent's own past
  experience using SingleStore's native ``<*>`` DOT_PRODUCT operator, so an
  agent literally re-reads what it learned last time before deciding again.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Iterable, Sequence

import singlestoredb as s2
from openai import OpenAI

from .config import settings


# --------------------------------------------------------------------------
# Connections
# --------------------------------------------------------------------------

def connect(results_type: str = "tuples"):
    """Open a fresh SingleStore connection to the demo database."""
    return s2.connect(
        host=settings.host,
        port=settings.port,
        user=settings.user,
        password=settings.password,
        database=settings.database,
        results_type=results_type,
        autocommit=True,
    )


def query(sql: str, params: Sequence[Any] | None = None) -> list[dict]:
    """Run a SELECT and return rows as dicts."""
    conn = connect(results_type="dicts")
    try:
        cur = conn.cursor()
        cur.execute(sql, tuple(params) if params else None)
        return list(cur.fetchall())
    finally:
        conn.close()


def execute(sql: str, params: Sequence[Any] | None = None) -> int:
    conn = connect()
    try:
        cur = conn.cursor()
        n = cur.execute(sql, tuple(params) if params else None)
        return n or 0
    finally:
        conn.close()


def executemany(sql: str, rows: Sequence[Sequence[Any]]) -> int:
    if not rows:
        return 0
    conn = connect()
    try:
        cur = conn.cursor()
        cur.executemany(sql, [tuple(r) for r in rows])
        return len(rows)
    finally:
        conn.close()


# --------------------------------------------------------------------------
# Embeddings (Qwen via the SingleStore-hosted OpenAI-compatible endpoint)
# --------------------------------------------------------------------------

_client: OpenAI | None = None


def _embed_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.qwen_key, base_url=settings.llm_endpoint)
    return _client


def embed(text: str) -> list[float]:
    """Embed one string to a 1024-dim vector."""
    return embed_batch([text])[0]


def embed_batch(texts: Sequence[str]) -> list[list[float]]:
    """Embed a batch of strings. Returns a list of 1024-dim vectors."""
    if not texts:
        return []
    resp = _embed_client().embeddings.create(
        model=settings.embedding_model,
        input=list(texts),
    )
    return [d.embedding for d in resp.data]


def vec_literal(vec: Sequence[float]) -> str:
    """Render a Python float list as a SingleStore VECTOR JSON string literal."""
    return "[" + ",".join(f"{x:.8g}" for x in vec) + "]"


# --------------------------------------------------------------------------
# Persisted agent memory
# --------------------------------------------------------------------------

def write_memory(
    memory_id: str,
    agent_id: str,
    kind: str,
    content: str,
    *,
    run_id: str | None = None,
    as_of_date: str | None = None,
    importance: float = 0.5,
    metrics: dict | None = None,
    tags: list | None = None,
    embedding: Sequence[float] | None = None,
) -> None:
    """Persist one memory row, embedding ``content`` if no vector supplied."""
    if embedding is None:
        embedding = embed(content)
    execute(
        """
        INSERT INTO agent_memory
            (memory_id, agent_id, run_id, kind, as_of_date, content,
             embedding, importance, metrics, tags, created_at)
        VALUES (%s,%s,%s,%s,%s,%s, %s, %s,%s,%s, NOW(6))
        """,
        (
            memory_id, agent_id, run_id, kind, as_of_date, content,
            vec_literal(embedding), float(importance),
            json.dumps(metrics or {}), json.dumps(tags or []),
        ),
    )


def recall_memory(
    agent_id: str,
    query_text: str,
    *,
    k: int = 5,
    kinds: Iterable[str] | None = None,
) -> list[dict]:
    """Semantic top-k recall of an agent's own memory by similarity to ``query_text``.

    Uses SingleStore's native ``<*>`` (DOT_PRODUCT) operator. Returns rows with a
    ``score`` column, most-relevant first — this is what the agent re-reads
    before it decides, blended with a mild importance boost.
    """
    qvec = vec_literal(embed(query_text))
    kind_clause = ""
    params: list[Any] = [qvec, agent_id]
    if kinds:
        kinds = list(kinds)
        kind_clause = " AND kind IN (%s)" % ",".join(["%s"] * len(kinds))
        params.extend(kinds)
    # ORDER BY references the query vector again; append it, then k.
    params.append(qvec)
    params.append(int(k))
    sql = f"""
        SELECT memory_id, agent_id, run_id, kind, as_of_date, content,
               importance, metrics, tags, created_at,
               (embedding <*> %s) AS score
        FROM agent_memory
        WHERE agent_id = %s{kind_clause}
        ORDER BY (embedding <*> %s) * (0.7 + 0.3 * importance) DESC
        LIMIT %s
    """
    return query(sql, params)


# --------------------------------------------------------------------------
# Compliance audit trail
# --------------------------------------------------------------------------

def audit(
    audit_id: str,
    agent_id: str,
    event_type: str,
    *,
    run_id: str | None = None,
    entity_ref: str | None = None,
    ticker: str | None = None,
    detail: dict | None = None,
    actor: str = "agent",
) -> None:
    """Append one immutable event to the trade audit trail."""
    execute(
        """
        INSERT INTO trade_audit
            (audit_id, ts, agent_id, run_id, event_type, entity_ref, ticker, detail, actor)
        VALUES (%s, NOW(6), %s,%s,%s,%s,%s,%s,%s)
        """,
        (audit_id, agent_id, run_id, event_type, entity_ref, ticker,
         json.dumps(detail or {}), actor),
    )


def now_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")
