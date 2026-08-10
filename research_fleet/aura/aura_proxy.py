"""Hardened Aura Analyst proxy — hosted gateway the research agents hit.

Fronts the REAL SingleStore Aura Analyst domain (per the hard rule: this proxies
a genuine Portal domain; it is NOT a local NL->SQL substitute). It exists so that:

  * the Aura API key lives in ONE place (this service), never on the agents;
  * every NL query is audited to SingleStore (aura_query_log) — a governance
    trail you want before any of this touches real money;
  * the hot path is resilient: bounded timeout, bounded retries with backoff, a
    circuit breaker that fails fast when Aura is down, a TTL response cache, and
    per-agent rate limiting;
  * agents get one stable internal URL with a simple, uniform response shape.

Endpoints (agents authenticate with a shared internal PROXY_TOKEN, NOT the Aura key):
  GET  /health                    -> liveness + circuit state + cache stats
  GET  /metrics                   -> counters (json)
  POST /ask     {question, agent_id?, output_modes?, no_cache?}  -> flattened JSON result
  POST /analyst/query  {message, ...}   -> passthrough-shaped (Aura /query compatible)
  POST /analyst/chat   {message, ...}   -> SSE passthrough stream (Aura /chat)

Env (from /opt/aura-proxy/.env):
  ANALYST_CHAT_URL   the real Aura /analyst/chat URL (…/analyst/chat)
  ANALYST_API_KEY    the Aura JWT (kept server-side only)
  PROXY_TOKEN        shared bearer the agents present to THIS proxy
  SINGLESTORE_*      for audit/cache logging
  plus tunables: PROXY_TIMEOUT_S, PROXY_MAX_RETRIES, PROXY_CACHE_TTL_S,
  PROXY_RATE_PER_MIN, CB_FAIL_THRESHOLD, CB_RESET_S
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import requests
import singlestoredb as s2
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

def _load_env() -> None:
    for cand in [Path("/opt/aura-proxy/.env"), Path(__file__).resolve().parent / ".env"]:
        if cand.is_file():
            for line in cand.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())
            break


_load_env()

CHAT_URL = os.environ["ANALYST_CHAT_URL"].rstrip("/")
QUERY_URL = CHAT_URL[:-len("/chat")] + "/query" if CHAT_URL.endswith("/chat") else CHAT_URL
AURA_KEY = os.environ["ANALYST_API_KEY"]
PROXY_TOKEN = os.environ.get("PROXY_TOKEN", "")

TIMEOUT_S = float(os.environ.get("PROXY_TIMEOUT_S", "90"))
MAX_RETRIES = int(os.environ.get("PROXY_MAX_RETRIES", "2"))
CACHE_TTL_S = int(os.environ.get("PROXY_CACHE_TTL_S", "900"))       # 15 min
RATE_PER_MIN = int(os.environ.get("PROXY_RATE_PER_MIN", "30"))      # per agent
CB_FAIL_THRESHOLD = int(os.environ.get("CB_FAIL_THRESHOLD", "5"))
CB_RESET_S = int(os.environ.get("CB_RESET_S", "30"))

DB = dict(
    host=os.environ["SINGLESTORE_HOST"], port=int(os.environ.get("SINGLESTORE_PORT", "3306")),
    user=os.environ["SINGLESTORE_USER"], password=os.environ["SINGLESTORE_PASSWORD"],
    database=os.environ.get("SINGLESTORE_DATABASE", "portfolio_agents"),
)

app = FastAPI(title="Aura Analyst Proxy", version="1.0.0")

METRICS: dict[str, int] = defaultdict(int)
_rate: dict[str, deque] = defaultdict(deque)
_rate_lock = threading.Lock()


# --------------------------------------------------------------------------
# Circuit breaker (fail fast when Aura is unhealthy)
# --------------------------------------------------------------------------

class Circuit:
    def __init__(self, threshold: int, reset_s: int):
        self.threshold, self.reset_s = threshold, reset_s
        self.fails = 0
        self.opened_at = 0.0
        self.lock = threading.Lock()

    def state(self, now: float) -> str:
        with self.lock:
            if self.fails < self.threshold:
                return "closed"
            if now - self.opened_at >= self.reset_s:
                return "half_open"
            return "open"

    def record(self, ok: bool, now: float) -> None:
        with self.lock:
            if ok:
                self.fails = 0
                self.opened_at = 0.0
            else:
                self.fails += 1
                if self.fails == self.threshold:
                    self.opened_at = now


CB = Circuit(CB_FAIL_THRESHOLD, CB_RESET_S)


# --------------------------------------------------------------------------
# DB helpers (best-effort: audit/cache must never break the hot path)
# --------------------------------------------------------------------------

def _conn(dict_rows: bool = False):
    return s2.connect(connect_timeout=10, results_type="dicts" if dict_rows else "tuples", **DB)


def _qhash(question: str, modes: str) -> str:
    norm = " ".join(question.lower().split())
    return hashlib.sha256(f"{norm}|{modes}".encode()).hexdigest()


def cache_get(qhash: str, modes: str) -> dict | None:
    try:
        c = _conn()
        try:
            cur = c.cursor()
            cur.execute("SELECT response_json FROM aura_cache WHERE question_hash=%s AND output_modes=%s "
                        "AND expires_at > NOW(6)", (qhash, modes))
            row = cur.fetchone()
            if row:
                cur.execute("UPDATE aura_cache SET hits=hits+1 WHERE question_hash=%s AND output_modes=%s",
                            (qhash, modes))
                val = row[0] if isinstance(row, (list, tuple)) else row["response_json"]
                return json.loads(val)
        finally:
            c.close()
    except Exception:
        return None
    return None


def cache_put(qhash: str, question: str, modes: str, payload: dict) -> None:
    try:
        c = _conn()
        try:
            cur = c.cursor()
            cur.execute(
                """INSERT INTO aura_cache
                   (question_hash, question, output_modes, response_json, confidence, row_count, created_at, expires_at, hits)
                   VALUES (%s,%s,%s,%s,%s,%s, NOW(6), DATE_ADD(NOW(6), INTERVAL %s SECOND), 0)
                   ON DUPLICATE KEY UPDATE response_json=VALUES(response_json), confidence=VALUES(confidence),
                     row_count=VALUES(row_count), created_at=NOW(6), expires_at=VALUES(expires_at)""",
                (qhash, question[:8000], modes, json.dumps(payload)[:1_000_000],
                 payload.get("confidence"), payload.get("row_count"), CACHE_TTL_S))
        finally:
            c.close()
    except Exception:
        pass


def audit(rec: dict) -> None:
    try:
        c = _conn()
        try:
            cur = c.cursor()
            cur.execute(
                """INSERT INTO aura_query_log
                   (query_id, ts, agent_id, endpoint, question, question_hash, generated_sql, confidence,
                    tables_used, row_count, answer_text, error, status, http_status, trace_id,
                    latency_ms, upstream_ms, cache_hit, attempts, session_id, client_ip)
                   VALUES (%s, NOW(6), %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (rec["query_id"], rec.get("agent_id"), rec.get("endpoint", "query"),
                 rec.get("question", "")[:8000], rec.get("question_hash", ""),
                 (rec.get("generated_sql") or "")[:8000], rec.get("confidence"),
                 json.dumps(rec.get("tables_used") or []), rec.get("row_count"),
                 (rec.get("answer_text") or "")[:8000], (rec.get("error") or "")[:2000] or None,
                 rec.get("status", "ok"), rec.get("http_status"), rec.get("trace_id"),
                 rec.get("latency_ms"), rec.get("upstream_ms"), 1 if rec.get("cache_hit") else 0,
                 rec.get("attempts", 1), rec.get("session_id"), rec.get("client_ip")))
        finally:
            c.close()
    except Exception:
        pass


# --------------------------------------------------------------------------
# Auth + rate limit
# --------------------------------------------------------------------------

def _check_auth(authorization: str | None) -> None:
    if not PROXY_TOKEN:
        return  # unset => open within the private VPC (still SG-scoped)
    if authorization != f"Bearer {PROXY_TOKEN}":
        METRICS["auth_reject"] += 1
        raise HTTPException(status_code=401, detail="invalid proxy token")


def _rate_ok(agent_id: str) -> bool:
    now = time.time()
    with _rate_lock:
        dq = _rate[agent_id]
        while dq and now - dq[0] > 60:
            dq.popleft()
        if len(dq) >= RATE_PER_MIN:
            return False
        dq.append(now)
        return True


# --------------------------------------------------------------------------
# Core upstream call: timeout + retry/backoff + circuit breaker
# --------------------------------------------------------------------------

def _flatten(data: dict) -> dict:
    results = data.get("results") or []
    first = results[0] if results else {}
    sql = first.get("sql") or {}
    dat = first.get("data") or {}
    return {
        "sql": sql.get("command"), "confidence": sql.get("confidence_score"),
        "tables_used": sql.get("tables_used"), "columns": dat.get("columns"),
        "rows": dat.get("rows"), "row_count": dat.get("row_count"),
        "text": first.get("text"), "error": first.get("error"),
        "raw": data,
    }


def call_aura(question: str, modes: list[str], session_id: str | None) -> tuple[dict, dict]:
    """Returns (flattened_payload, meta). Raises HTTPException on hard failure."""
    now = time.time()
    st = CB.state(now)
    if st == "open":
        METRICS["circuit_open"] += 1
        raise HTTPException(status_code=503, detail="aura upstream circuit open (failing fast)",
                            headers={"x-proxy-status": "circuit_open"})

    payload: dict[str, Any] = {"message": question}
    if modes:
        payload["output_modes"] = modes
    if session_id:
        payload["session_id"] = session_id

    last_err = None
    attempts = 0
    up_ms = 0.0
    for attempt in range(1, MAX_RETRIES + 2):  # 1 try + MAX_RETRIES
        attempts = attempt
        t0 = time.perf_counter()
        try:
            r = requests.post(QUERY_URL,
                              headers={"Authorization": f"Bearer {AURA_KEY}", "Content-Type": "application/json"},
                              json=payload, timeout=TIMEOUT_S)
            up_ms = (time.perf_counter() - t0) * 1e3
            trace = r.headers.get("singlestore-trace-id")
            if r.status_code == 200:
                CB.record(True, time.time())
                flat = _flatten(r.json())
                return flat, {"attempts": attempts, "upstream_ms": up_ms, "trace_id": trace,
                              "http_status": 200}
            # 4xx (bad request / auth) => don't retry, don't trip breaker on client errors
            if 400 <= r.status_code < 500:
                CB.record(True, time.time())
                raise HTTPException(status_code=r.status_code,
                                    detail=f"aura client error: {r.text[:300]}",
                                    headers={"x-proxy-status": "upstream_client_error",
                                             "x-trace-id": trace or ""})
            last_err = f"HTTP {r.status_code}: {r.text[:200]}"
        except requests.Timeout:
            up_ms = (time.perf_counter() - t0) * 1e3
            last_err = f"timeout after {TIMEOUT_S}s"
        except HTTPException:
            raise
        except Exception as e:
            up_ms = (time.perf_counter() - t0) * 1e3
            last_err = str(e)[:200]
        # backoff before retry
        if attempt < MAX_RETRIES + 1:
            time.sleep(min(2 ** (attempt - 1), 5))

    CB.record(False, time.time())
    METRICS["upstream_fail"] += 1
    raise HTTPException(status_code=502, detail=f"aura upstream failed after {attempts} attempts: {last_err}",
                        headers={"x-proxy-status": "upstream_error", "x-attempts": str(attempts)})


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=8000)
    agent_id: str | None = None
    output_modes: list[str] | None = None
    session_id: str | None = None
    no_cache: bool = False


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.get("/health")
def health():
    now = time.time()
    return {"ok": True, "circuit": CB.state(now), "circuit_fails": CB.fails,
            "aura_query_url": QUERY_URL, "cache_ttl_s": CACHE_TTL_S,
            "rate_per_min": RATE_PER_MIN, "timeout_s": TIMEOUT_S, "max_retries": MAX_RETRIES}


@app.get("/metrics")
def metrics():
    return dict(METRICS)


@app.post("/ask")
def ask(req: AskRequest, request: Request, authorization: str | None = Header(default=None)):
    _check_auth(authorization)
    agent_id = req.agent_id or "unknown"
    if not _rate_ok(agent_id):
        METRICS["rate_limited"] += 1
        rec = {"query_id": f"aq-{uuid.uuid4().hex[:16]}", "agent_id": agent_id,
               "question": req.question, "question_hash": "", "status": "rate_limited",
               "latency_ms": 0.0, "client_ip": request.client.host if request.client else None}
        audit(rec)
        raise HTTPException(status_code=429, detail=f"rate limit {RATE_PER_MIN}/min for agent {agent_id}")

    modes = req.output_modes or ["sql", "data"]
    modes_key = ",".join(sorted(modes))
    qhash = _qhash(req.question, modes_key)
    qid = f"aq-{uuid.uuid4().hex[:16]}"
    t_start = time.perf_counter()
    METRICS["requests"] += 1

    # cache
    if not req.no_cache:
        hit = cache_get(qhash, modes_key)
        if hit is not None:
            METRICS["cache_hit"] += 1
            lat = (time.perf_counter() - t_start) * 1e3
            audit({"query_id": qid, "agent_id": agent_id, "question": req.question,
                   "question_hash": qhash, "generated_sql": hit.get("sql"),
                   "confidence": hit.get("confidence"), "tables_used": hit.get("tables_used"),
                   "row_count": hit.get("row_count"), "answer_text": hit.get("text"),
                   "status": "cache", "latency_ms": lat, "cache_hit": True,
                   "client_ip": request.client.host if request.client else None})
            return {"ok": True, "cached": True, "query_id": qid, **{k: hit.get(k) for k in
                    ("sql", "confidence", "tables_used", "columns", "rows", "row_count", "text", "error")}}

    # upstream
    try:
        flat, meta = call_aura(req.question, modes, req.session_id)
    except HTTPException as e:
        lat = (time.perf_counter() - t_start) * 1e3
        status = e.headers.get("x-proxy-status", "error") if e.headers else "error"
        audit({"query_id": qid, "agent_id": agent_id, "question": req.question, "question_hash": qhash,
               "status": status, "http_status": e.status_code, "error": str(e.detail),
               "latency_ms": lat, "client_ip": request.client.host if request.client else None})
        raise

    lat = (time.perf_counter() - t_start) * 1e3
    is_err = bool(flat.get("error"))
    METRICS["success" if not is_err else "aura_result_error"] += 1
    audit({"query_id": qid, "agent_id": agent_id, "question": req.question, "question_hash": qhash,
           "endpoint": "query", "generated_sql": flat.get("sql"), "confidence": flat.get("confidence"),
           "tables_used": flat.get("tables_used"), "row_count": flat.get("row_count"),
           "answer_text": flat.get("text"), "error": flat.get("error"),
           "status": "error" if is_err else "ok", "http_status": meta["http_status"],
           "trace_id": meta["trace_id"], "latency_ms": lat, "upstream_ms": meta["upstream_ms"],
           "attempts": meta["attempts"], "session_id": req.session_id,
           "client_ip": request.client.host if request.client else None})

    # cache only clean, non-empty answers
    if not req.no_cache and not is_err and (flat.get("rows") or flat.get("text")):
        cache_put(qhash, req.question, modes_key, flat)

    return {"ok": not is_err, "cached": False, "query_id": qid, "trace_id": meta["trace_id"],
            **{k: flat.get(k) for k in ("sql", "confidence", "tables_used", "columns",
                                        "rows", "row_count", "text", "error")}}


@app.post("/analyst/query")
def analyst_query(body: dict, request: Request, authorization: str | None = Header(default=None)):
    """Aura /query-compatible passthrough (message/output_modes/session_id)."""
    _check_auth(authorization)
    req = AskRequest(question=body.get("message", ""), agent_id=body.get("agent_id"),
                     output_modes=body.get("output_modes"), session_id=body.get("session_id"),
                     no_cache=bool(body.get("no_cache")))
    return ask(req, request, authorization)


@app.post("/analyst/chat")
def analyst_chat(body: dict, request: Request, authorization: str | None = Header(default=None)):
    """SSE passthrough to Aura /chat (for streaming callers). Audited at open."""
    _check_auth(authorization)
    agent_id = body.get("agent_id") or "unknown"
    if not _rate_ok(agent_id):
        raise HTTPException(status_code=429, detail=f"rate limit {RATE_PER_MIN}/min")
    if CB.state(time.time()) == "open":
        raise HTTPException(status_code=503, detail="aura upstream circuit open")
    payload = {k: body[k] for k in ("message", "session_id", "included_events") if k in body}
    qid = f"aq-{uuid.uuid4().hex[:16]}"
    audit({"query_id": qid, "agent_id": agent_id, "endpoint": "chat",
           "question": body.get("message", ""), "question_hash": _qhash(body.get("message", ""), "chat"),
           "status": "ok", "client_ip": request.client.host if request.client else None})

    def gen():
        try:
            with requests.post(CHAT_URL,
                               headers={"Authorization": f"Bearer {AURA_KEY}",
                                        "Content-Type": "application/json", "Accept": "text/event-stream"},
                               json=payload, stream=True, timeout=TIMEOUT_S) as r:
                CB.record(r.status_code < 500, time.time())
                for line in r.iter_lines():
                    if line:
                        yield line + b"\n"
                    else:
                        yield b"\n"
        except Exception as e:
            CB.record(False, time.time())
            yield f"event: error\ndata: {json.dumps({'error': str(e)[:200]})}\n\n".encode()

    return StreamingResponse(gen(), media_type="text/event-stream")
