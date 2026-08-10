from __future__ import annotations
import os
from contextlib import contextmanager
from typing import Any, Sequence
import singlestoredb as s2


def _conn_kwargs() -> dict:
    return {
        "host": os.environ["SINGLESTORE_HOST"],
        "port": int(os.environ.get("SINGLESTORE_PORT", "3306")),
        "user": os.environ.get("SINGLESTORE_USER", "admin"),
        "password": os.environ.get("SINGLESTORE_PASSWORD", ""),
        "database": os.environ.get("SINGLESTORE_DATABASE") or None,
    }


def connect(results_type: str = "tuples"):
    kw = _conn_kwargs()
    if kw["database"] is None:
        kw.pop("database")
    kw["results_type"] = results_type
    kw["autocommit"] = True  # each read sees the fleet's latest committed rows
    return s2.connect(**kw)


def query(sql: str, params: Sequence[Any] | None = None) -> list[dict]:
    """Run a read-only SELECT and return rows as dicts (empty list if none)."""
    conn = connect(results_type="dicts")
    try:
        cur = conn.cursor()
        cur.execute(sql, tuple(params) if params else None)
        return list(cur.fetchall() or [])
    finally:
        conn.close()


@contextmanager
def cursor():
    conn = connect()
    try:
        cur = conn.cursor()
        try:
            yield cur
        finally:
            cur.close()
    finally:
        conn.close()


def ping() -> dict:
    """Return {'ok': True, 'version': '...'} or raise."""
    with cursor() as cur:
        cur.execute("SELECT @@version")
        row = cur.fetchone()
        return {"ok": True, "version": row[0] if row else None}
