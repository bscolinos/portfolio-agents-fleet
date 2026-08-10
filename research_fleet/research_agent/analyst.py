"""Aura Analyst client for the research agents.

Agents no longer hold the raw Aura API key. They call the HOSTED, HARDENED
**Aura proxy** (dedicated EC2 in the VPC), which fronts the REAL SingleStore
Aura Analyst Portal domain, centralizes the credential, and adds caching,
retries, a circuit breaker, rate limiting, and full audit logging. This still
honors the HARD RULE — the proxy proxies a genuine Portal domain; it is NEVER a
local NL->SQL substitute.

Config (from the demo/agent ``.env``):
  ANALYST_PROXY_URL   e.g. http://172.31.12.154:8799   (private IP of the proxy)
  ANALYST_PROXY_TOKEN the shared bearer the proxy expects (NOT the Aura key)

Legacy direct mode (only if the proxy vars are unset AND a raw key is present):
  ANALYST_API_URL + ANALYST_API_KEY  -> call Aura directly. Prefer the proxy.

If neither is configured, :func:`available` is False and the agent SKIPS the
analysis phase (it does not fabricate SQL locally).
"""

from __future__ import annotations

import os
import time
from typing import Any

import requests

# Ensure the demo/agent .env is loaded before we read config (independent of
# import order — research_db has the same loader but may import after us).
try:
    from . import research_db as _rdb  # triggers _load_env() as a side effect
except Exception:
    pass


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


PROXY_URL = _env("ANALYST_PROXY_URL").rstrip("/")
PROXY_TOKEN = _env("ANALYST_PROXY_TOKEN")

# legacy direct-to-Aura (discouraged; proxy is preferred)
ANALYST_API_URL = _env("ANALYST_API_URL")
ANALYST_API_KEY = _env("ANALYST_API_KEY")


def _cfg() -> dict:
    """Read config dynamically so a late-loaded .env is always respected."""
    return {
        "proxy_url": _env("ANALYST_PROXY_URL").rstrip("/"),
        "proxy_token": _env("ANALYST_PROXY_TOKEN"),
        "api_url": _env("ANALYST_API_URL"),
        "api_key": _env("ANALYST_API_KEY"),
    }


def _mode() -> str:
    c = _cfg()
    if c["proxy_url"] and c["proxy_token"]:
        return "proxy"
    if c["api_url"] and c["api_key"]:
        return "direct"
    return "none"


def available() -> bool:
    """True only when the hosted proxy (preferred) or a direct Aura endpoint is configured."""
    return _mode() != "none"


def ask(message: str, output_modes: list[str] | None = None, *,
        agent_id: str = "", session_id: str | None = None, timeout: int = 120) -> dict:
    """Ask Aura an English question via the proxy (preferred) or directly.

    Returns a flattened result: {sql, confidence, tables_used, columns, rows,
    row_count, text, error, latency_ms, trace_id, cached}.
    Raises RuntimeError if nothing is configured (caller must check available()
    and skip — never substitute a local NL->SQL).
    """
    mode = _mode()
    if mode == "none":
        raise RuntimeError("Aura not configured (set ANALYST_PROXY_URL+ANALYST_PROXY_TOKEN, "
                           "or legacy ANALYST_API_URL+ANALYST_API_KEY). Do NOT build a local NL->SQL substitute.")
    c = _cfg()
    t0 = time.perf_counter()
    modes = output_modes or ["sql", "data"]

    if mode == "proxy":
        r = requests.post(
            f"{c['proxy_url']}/ask",
            headers={"Authorization": f"Bearer {c['proxy_token']}", "Content-Type": "application/json"},
            json={"question": message, "agent_id": agent_id, "output_modes": modes,
                  "session_id": session_id},
            timeout=timeout,
        )
        latency_ms = (time.perf_counter() - t0) * 1e3
        # The proxy already returns a flattened shape; surface it directly.
        if r.status_code == 200:
            d = r.json()
            return {"sql": d.get("sql"), "confidence": d.get("confidence"),
                    "tables_used": d.get("tables_used"), "columns": d.get("columns"),
                    "rows": d.get("rows"), "row_count": d.get("row_count"),
                    "text": d.get("text"), "error": d.get("error"),
                    "cached": d.get("cached"), "trace_id": d.get("trace_id"),
                    "latency_ms": latency_ms}
        # proxy-level error (429/502/503/401) — surface as an error result, do not fall back
        return {"error": f"proxy HTTP {r.status_code}: {r.text[:200]}", "sql": None,
                "rows": None, "row_count": 0, "latency_ms": latency_ms}

    # direct-to-Aura (legacy)
    url = c["api_url"]
    if url.rstrip("/").endswith("/analyst/chat"):
        url = url.rstrip("/")[: -len("/chat")] + "/query"
    payload: dict[str, Any] = {"message": message, "output_modes": modes}
    if session_id:
        payload["session_id"] = session_id
    resp = requests.post(url, headers={"Authorization": f"Bearer {c['api_key']}",
                                       "Content-Type": "application/json"},
                         json=payload, timeout=timeout)
    latency_ms = (time.perf_counter() - t0) * 1e3
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results") or []
    first = results[0] if results else {}
    sql = first.get("sql") or {}
    dat = first.get("data") or {}
    return {"sql": sql.get("command"), "confidence": sql.get("confidence_score"),
            "tables_used": sql.get("tables_used"), "columns": dat.get("columns"),
            "rows": dat.get("rows"), "row_count": dat.get("row_count"),
            "text": first.get("text"), "error": first.get("error"),
            "cached": False, "trace_id": resp.headers.get("singlestore-trace-id"),
            "latency_ms": latency_ms}
