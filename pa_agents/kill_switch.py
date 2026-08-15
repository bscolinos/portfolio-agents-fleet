"""Persisted, fleet-wide + per-agent KILL SWITCH — the operator's big red button.

This is the first thing the pre-trade risk gate checks and the last line of
defence before real capital moves. State lives in SingleStore (``kill_switches``
table), NOT in process memory, so:

  * an engaged switch SURVIVES a fleet restart, and
  * an operator tripping it is visible to every agent process the instant the
    next gate call runs a ``SELECT`` — no message bus, no redeploy.

Two scopes:
  * ``global``    — trips the ENTIRE fleet. Every agent is hard-rejected.
  * ``<agent_id>``— trips just that one agent.

``is_engaged(agent_id)`` returns True if EITHER the global switch OR that agent's
own switch is engaged (fail-loud OR semantics — any engaged switch stops trading).

Every engage / release writes an immutable ``trade_audit`` row (event_type
``KILL_SWITCH``) so the compliance trail shows exactly who tripped what, when,
and why.

CLI (what an operator hits in an emergency — dead simple, loud banner)::

    python -m pa_agents.kill_switch engage --scope global --reason "vol spike" --by ops
    python -m pa_agents.kill_switch engage --scope max-return --reason "runaway" --by ops
    python -m pa_agents.kill_switch release --scope global --by ops
    python -m pa_agents.kill_switch status
"""

from __future__ import annotations

import argparse
import sys
import uuid

from . import db

GLOBAL = "global"


def _uid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:16]}"


# --------------------------------------------------------------------------
# Core state transitions (persisted, audited)
# --------------------------------------------------------------------------

def engage(scope: str = GLOBAL, *, reason: str, by: str) -> dict:
    """Engage the kill switch for ``scope`` ('global' or an agent_id).

    Idempotent: engaging an already-engaged scope refreshes the reason/who/when.
    Writes a KILL_SWITCH audit row.
    """
    if not scope:
        raise ValueError("scope is required ('global' or an agent_id)")
    if not reason:
        raise ValueError("reason is required — an engaged kill switch must say why")
    db.execute(
        """INSERT INTO kill_switches
               (scope, engaged, reason, engaged_by, engaged_at, updated_at)
           VALUES (%s, 1, %s, %s, NOW(6), NOW(6))
           ON DUPLICATE KEY UPDATE
               engaged=1, reason=VALUES(reason), engaged_by=VALUES(engaged_by),
               engaged_at=NOW(6), released_by=NULL, released_at=NULL,
               updated_at=NOW(6)""",
        (scope, reason, by),
    )
    audit_agent = scope if scope != GLOBAL else "*"
    db.audit(_uid("aud"), audit_agent, "KILL_SWITCH",
             detail={"action": "engage", "scope": scope, "reason": reason, "by": by},
             actor=by)
    return {"scope": scope, "engaged": True, "reason": reason, "by": by}


def release(scope: str = GLOBAL, *, by: str) -> dict:
    """Release (disengage) the kill switch for ``scope``. Writes a KILL_SWITCH audit row."""
    if not scope:
        raise ValueError("scope is required ('global' or an agent_id)")
    db.execute(
        """INSERT INTO kill_switches
               (scope, engaged, released_by, released_at, updated_at)
           VALUES (%s, 0, %s, NOW(6), NOW(6))
           ON DUPLICATE KEY UPDATE
               engaged=0, released_by=VALUES(released_by),
               released_at=NOW(6), updated_at=NOW(6)""",
        (scope, by),
    )
    audit_agent = scope if scope != GLOBAL else "*"
    db.audit(_uid("aud"), audit_agent, "KILL_SWITCH",
             detail={"action": "release", "scope": scope, "by": by},
             actor=by)
    return {"scope": scope, "engaged": False, "by": by}


def is_engaged(agent_id: str | None = None) -> dict:
    """Return the effective kill-switch state for ``agent_id``.

    Global OR the agent's own switch trips it. Returns::

        {"engaged": bool, "scope": "global"|"<agent_id>"|None, "reason": str|None}

    Global takes precedence in the reported scope/reason when both are engaged.
    """
    scopes = [GLOBAL] + ([agent_id] if agent_id else [])
    placeholders = ",".join(["%s"] * len(scopes))
    rows = db.query(
        f"""SELECT scope, engaged, reason FROM kill_switches
            WHERE scope IN ({placeholders}) AND engaged=1""",
        scopes,
    )
    if not rows:
        return {"engaged": False, "scope": None, "reason": None}
    # Prefer the global row when present so the reported reason is the fleet-wide one.
    by_scope = {r["scope"]: r for r in rows}
    hit = by_scope.get(GLOBAL) or rows[0]
    return {"engaged": True, "scope": hit["scope"], "reason": hit["reason"]}


def status() -> list[dict]:
    """Return every kill-switch row (engaged or not), newest activity first."""
    return db.query(
        """SELECT scope, engaged, reason, engaged_by, engaged_at,
                  released_by, released_at, updated_at
           FROM kill_switches
           ORDER BY engaged DESC, updated_at DESC""",
    )


# --------------------------------------------------------------------------
# CLI — loud, unmistakable
# --------------------------------------------------------------------------

_RED = "\033[91m"
_GREEN = "\033[92m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _banner(text: str, color: str) -> None:
    bar = "=" * 72
    print(f"{color}{_BOLD}{bar}")
    for line in text.splitlines():
        print(line)
    print(f"{bar}{_RESET}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("pa_agents.kill_switch",
                                 description="Fleet-wide / per-agent trading kill switch.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("engage", help="Engage the kill switch (STOPS trading).")
    e.add_argument("--scope", default=GLOBAL, help="'global' (default) or an agent_id.")
    e.add_argument("--reason", required=True, help="Why the switch is being tripped.")
    e.add_argument("--by", required=True, help="Operator engaging the switch.")

    r = sub.add_parser("release", help="Release the kill switch (RESUMES trading).")
    r.add_argument("--scope", default=GLOBAL, help="'global' (default) or an agent_id.")
    r.add_argument("--by", required=True, help="Operator releasing the switch.")

    sub.add_parser("status", help="Show all kill-switch state.")

    args = ap.parse_args(argv)

    if args.cmd == "engage":
        engage(args.scope, reason=args.reason, by=args.by)
        _banner(f"KILL SWITCH ENGAGED  —  scope='{args.scope}'\n"
                f"reason: {args.reason}\nby: {args.by}\n"
                f"ALL GATED TRADING FOR THIS SCOPE IS NOW HARD-REJECTED.", _RED)
    elif args.cmd == "release":
        release(args.scope, by=args.by)
        _banner(f"KILL SWITCH RELEASED  —  scope='{args.scope}'\nby: {args.by}\n"
                f"Trading may resume for this scope (subject to the risk gate).", _GREEN)
    elif args.cmd == "status":
        rows = status()
        if not rows:
            print("No kill-switch rows. All scopes implicitly DISENGAGED.")
            return 0
        any_engaged = any(r["engaged"] for r in rows)
        header = "KILL-SWITCH STATUS  —  " + (
            "ONE OR MORE SWITCHES ENGAGED" if any_engaged else "all clear")
        _banner(header, _RED if any_engaged else _GREEN)
        for r in rows:
            state = "ENGAGED" if r["engaged"] else "released"
            print(f"  [{state:8s}] scope={r['scope']:<16s} "
                  f"reason={r['reason'] or '-'} "
                  f"by={r['engaged_by'] or r['released_by'] or '-'} "
                  f"updated={r['updated_at']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
