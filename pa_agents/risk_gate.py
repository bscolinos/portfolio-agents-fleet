"""Pre-trade RISK GATE — the hard safety layer between a research finding and any order.

Nothing today sits between an optimizer's target weights and the real trade
lifecycle. This module is that gate. It takes a PROPOSED rebalance
(agent_id, as_of_date, target_weights, prices, plus the agent's current book +
NAV read from SingleStore) and returns a :class:`Decision`:

    {approved, reason, violations, adjusted_weights, checks}

Design posture is DEFAULT-DENY / FAIL-CLOSED: if a required input is missing or
ambiguous (no prices, no NAV, unreadable history), the gate REJECTS. It never
silently approves.

Every evaluation is fully auditable — it writes BOTH:
  * an immutable ``trade_audit`` row (event_type ``RISK_CHECK``), and
  * a first-class ``risk_decisions`` row (queryable analytics of the decision).

Enforced checks (all configurable via :class:`RiskLimits`, institutional defaults):
  * KILL SWITCH (checked FIRST) — global or per-agent → hard reject.
  * Max single-name weight — reject, or (configurable) clip + renormalize.
  * Sector concentration — only if ``securities.sector`` is populated; skipped
    gracefully otherwise (documented).
  * Gross / net exposure caps — long-only, no leverage, no shorts by default.
  * Max turnover per rebalance — reject churny rebalances.
  * Max # names / min position size — avoid over-diversification + dust.
  * Drawdown circuit-breaker — trailing peak-to-current from nav_history.
  * Daily-loss limit — today's realized+unrealized P&L floor.
  * Price sanity — missing / zero / non-positive target price → reject.
  * Notional guard — per-name absolute notional cap (ADV proxy; see limitation).

--------------------------------------------------------------------------------
INSERTION POINT (how runner.py activates this for live trading)
--------------------------------------------------------------------------------
In ``pa_agents/runner.py``, immediately AFTER the SOLVE step produces
``res.weights`` and BEFORE the ``rebalance(...)`` call (step "4) TRADE"), insert::

    from .risk_gate import evaluate
    decision = evaluate(agent_id=agent_id, as_of_date=as_of_date,
                        target_weights=res.weights, prices=prices, run_id=run_id)
    if not decision.approved:
        db.execute("UPDATE agent_runs SET status='blocked', error=%s, "
                   "finished_at=NOW(6) WHERE run_id=%s",
                   (decision.reason[:500], run_id))
        return {"run_id": run_id, "agent_id": agent_id, "blocked": True,
                "reason": decision.reason, "violations": decision.violations}
    weights = decision.adjusted_weights or res.weights   # honor any clip+renormalize
    reb = rebalance(run_id=run_id, agent_id=agent_id, as_of_date=as_of_date,
                    target_weights=weights, prices=prices, ...)

For paper trading, prefer :func:`pa_agents.paper_trader.gated_rebalance` which
wraps this gate and routes approved orders to the shadow book.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any

from . import db
from . import kill_switch


def _uid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:16]}"


# --------------------------------------------------------------------------
# Limits (institutional defaults). Every threshold lives here and is tunable.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RiskLimits:
    # --- concentration ---
    max_position_weight: float = 0.10      # ≤10% per single name
    clip_over_weight: bool = False         # False = reject; True = clip + renormalize
    max_sector_weight: float = 0.30        # ≤30% per sector (only if sector data present)
    # --- exposure ---
    max_gross_exposure: float = 1.00       # Σ|w| ≤ 100% (no leverage)
    max_net_exposure: float = 1.00         # Σ w  ≤ 100%
    allow_shorts: bool = False             # long-only by default
    min_gross_exposure: float = 0.0        # allow all-cash (defensive) proposals
    # --- churn ---
    max_turnover: float = 0.40             # ≤40% of NAV traded per rebalance
    # --- breadth ---
    max_names: int = 100                   # cap over-diversification
    min_position_weight: float = 0.005     # ≥0.5% or it's dust (0 => disabled)
    # --- drawdown / loss circuit-breakers ---
    max_drawdown: float = 0.15             # trailing peak-to-current DD ≥15% => de-risk only
    daily_loss_limit: float = 0.05         # today's P&L ≤ -5% of prior NAV => reject
    # --- price sanity ---
    require_prices: bool = True            # every target name must have a positive price
    # --- notional guard (ADV proxy — see RISK_CONTROLS.md limitation) ---
    max_name_notional: float = 25_000_000.0  # absolute $ cap on any single-name order
    # weight tolerance for gross/net comparisons (float slop)
    weight_tol: float = 1e-6


# --------------------------------------------------------------------------
# Decision object
# --------------------------------------------------------------------------

@dataclass
class Decision:
    approved: bool
    reason: str
    violations: list[dict] = field(default_factory=list)
    adjusted_weights: dict[str, float] | None = None
    checks: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "approved": self.approved,
            "reason": self.reason,
            "violations": self.violations,
            "adjusted_weights": self.adjusted_weights,
            "checks": self.checks,
        }


def _violation(code: str, detail: str, **extra) -> dict:
    v = {"code": code, "detail": detail}
    v.update(extra)
    return v


# --------------------------------------------------------------------------
# Book / history readers (fail-closed if the DB can't be read)
# --------------------------------------------------------------------------

def _current_weights(agent_id: str, prices: dict[str, float]) -> tuple[dict[str, float], float] | None:
    """Return (current_weights_by_ticker, nav) from the live book, or None on failure.

    NAV is cash + Σ market_value marked at ``prices`` (falling back to avg_cost).
    Returns ({}, seed_nav) for an agent with no book yet (inception) — that is a
    legitimate, non-ambiguous empty book, not a failure.
    """
    try:
        pos = db.query(
            "SELECT ticker, qty, avg_cost FROM positions WHERE agent_id=%s", (agent_id,))
        nav_rows = db.query(
            "SELECT nav, cash FROM nav_history WHERE agent_id=%s "
            "ORDER BY as_of_date DESC LIMIT 1", (agent_id,))
    except Exception:
        return None
    if nav_rows:
        cash = float(nav_rows[0]["cash"])
        mv = sum(float(r["qty"]) * prices.get(r["ticker"], float(r["avg_cost"])) for r in pos)
        nav = cash + mv
    else:
        nav = None  # no history — caller must supply nav
    weights: dict[str, float] = {}
    if nav and nav > 0:
        for r in pos:
            px = prices.get(r["ticker"], float(r["avg_cost"]))
            weights[r["ticker"]] = (float(r["qty"]) * px) / nav
    return weights, (nav if nav is not None else 0.0)


def _drawdown_and_daily(agent_id: str) -> dict:
    """Trailing peak-to-current drawdown + last daily P&L fraction from nav_history.

    Returns {"drawdown": float>=0, "daily_pnl_frac": float, "have_history": bool}.
    Tolerates an EMPTY history (new agent) by reporting have_history=False and
    zeroed metrics — those two circuit-breakers then no-op for a fresh agent
    (documented). A DB read FAILURE (vs empty) is surfaced as have_history=None
    so the caller can fail closed.
    """
    try:
        rows = db.query(
            "SELECT nav, daily_return FROM nav_history WHERE agent_id=%s "
            "ORDER BY as_of_date ASC", (agent_id,))
    except Exception:
        return {"drawdown": 0.0, "daily_pnl_frac": 0.0, "have_history": None}
    if not rows:
        return {"drawdown": 0.0, "daily_pnl_frac": 0.0, "have_history": False}
    peak = 0.0
    cur = 0.0
    for r in rows:
        nav = float(r["nav"])
        cur = nav
        peak = max(peak, nav)
    dd = (peak - cur) / peak if peak > 0 else 0.0
    daily = rows[-1]["daily_return"]
    daily = float(daily) if daily is not None else 0.0
    return {"drawdown": dd, "daily_pnl_frac": daily, "have_history": True}


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------

def evaluate(
    *,
    agent_id: str,
    as_of_date: str,
    target_weights: dict[str, float],
    prices: dict[str, float],
    nav: float | None = None,
    current_weights: dict[str, float] | None = None,
    limits: RiskLimits | None = None,
    run_id: str | None = None,
    mode: str = "paper",
    persist: bool = True,
) -> Decision:
    """Evaluate a PROPOSED rebalance against every risk limit. Fail-closed.

    Reads the current book + NAV + drawdown/daily-loss from SingleStore unless
    ``current_weights`` / ``nav`` are supplied (used by tests to run in-memory).
    Writes a trade_audit RISK_CHECK row and a risk_decisions row when
    ``persist`` is True. Returns a :class:`Decision`.
    """
    lim = limits or RiskLimits()
    violations: list[dict] = []
    checks: dict[str, Any] = {}
    adjusted: dict[str, float] | None = None

    # ---- 0) KILL SWITCH — checked FIRST, hard reject ----------------------
    try:
        ks = kill_switch.is_engaged(agent_id)
    except Exception as exc:
        # Can't read the kill switch => cannot prove it's safe => fail closed.
        dec = Decision(False, f"kill-switch state unreadable ({exc}); failing closed",
                       [_violation("KILL_SWITCH_UNREADABLE", str(exc))], None,
                       {"kill_switch": "unreadable"})
        _persist(dec, agent_id, as_of_date, run_id, mode, target_weights, prices, nav, persist)
        return dec
    checks["kill_switch"] = ks
    if ks["engaged"]:
        dec = Decision(
            False,
            f"KILL SWITCH ENGAGED (scope={ks['scope']}): {ks['reason']}",
            [_violation("KILL_SWITCH", ks["reason"] or "engaged", scope=ks["scope"])],
            None, checks)
        _persist(dec, agent_id, as_of_date, run_id, mode, target_weights, prices, nav, persist)
        return dec

    # ---- resolve current book + NAV (fail-closed) -------------------------
    if current_weights is None or nav is None:
        loaded = _current_weights(agent_id, prices)
        if loaded is None:
            dec = Decision(False, "could not read current book/NAV from DB; failing closed",
                           [_violation("BOOK_UNREADABLE", "positions/nav_history read failed")],
                           None, checks)
            _persist(dec, agent_id, as_of_date, run_id, mode, target_weights, prices, nav, persist)
            return dec
        cur_w, loaded_nav = loaded
        if current_weights is None:
            current_weights = cur_w
        if nav is None:
            nav = loaded_nav
    if not nav or nav <= 0:
        dec = Decision(False, "NAV missing or non-positive; failing closed",
                       [_violation("NAV_MISSING", f"nav={nav}")], None, checks)
        _persist(dec, agent_id, as_of_date, run_id, mode, target_weights, prices, nav, persist)
        return dec
    checks["nav"] = float(nav)
    current_weights = current_weights or {}

    # ---- 1) PRICE SANITY --------------------------------------------------
    bad_prices = []
    for t, w in target_weights.items():
        if abs(w) <= lim.weight_tol:
            continue
        px = prices.get(t)
        if px is None or px <= 0:
            bad_prices.append(t)
    checks["bad_prices"] = bad_prices
    if lim.require_prices and bad_prices:
        violations.append(_violation(
            "PRICE_SANITY",
            f"{len(bad_prices)} target name(s) with missing/zero price: {bad_prices[:10]}",
            tickers=bad_prices))

    # ---- 2) SHORTS / SIGN -------------------------------------------------
    shorts = [t for t, w in target_weights.items() if w < -lim.weight_tol]
    checks["shorts"] = shorts
    if shorts and not lim.allow_shorts:
        violations.append(_violation(
            "SHORTS_DISALLOWED",
            f"{len(shorts)} short target(s) but shorts are disallowed: {shorts[:10]}",
            tickers=shorts))

    # ---- 3) EXPOSURE (gross / net) ---------------------------------------
    gross = sum(abs(w) for w in target_weights.values())
    net = sum(target_weights.values())
    checks["gross_exposure"] = gross
    checks["net_exposure"] = net
    if gross > lim.max_gross_exposure + lim.weight_tol:
        violations.append(_violation(
            "GROSS_EXPOSURE", f"gross {gross:.4f} > cap {lim.max_gross_exposure:.4f}",
            gross=gross, cap=lim.max_gross_exposure))
    if net > lim.max_net_exposure + lim.weight_tol:
        violations.append(_violation(
            "NET_EXPOSURE", f"net {net:.4f} > cap {lim.max_net_exposure:.4f}",
            net=net, cap=lim.max_net_exposure))
    if gross + lim.weight_tol < lim.min_gross_exposure:
        violations.append(_violation(
            "MIN_GROSS", f"gross {gross:.4f} < floor {lim.min_gross_exposure:.4f}",
            gross=gross, floor=lim.min_gross_exposure))

    # ---- 4) MAX SINGLE-NAME WEIGHT (reject or clip+renormalize) ----------
    over = {t: w for t, w in target_weights.items() if w > lim.max_position_weight + lim.weight_tol}
    checks["max_weight"] = max(target_weights.values()) if target_weights else 0.0
    checks["over_weight_names"] = list(over)
    if over:
        if lim.clip_over_weight:
            adjusted = _clip_and_renormalize(target_weights, lim.max_position_weight, lim.weight_tol)
            checks["adjusted_max_weight"] = max(adjusted.values()) if adjusted else 0.0
        else:
            violations.append(_violation(
                "MAX_POSITION_WEIGHT",
                f"{len(over)} name(s) exceed {lim.max_position_weight:.2%}: "
                f"{ {t: round(w,4) for t,w in list(over.items())[:10]} }",
                names=over, cap=lim.max_position_weight))

    # weights used for the remaining size-sensitive checks (post-clip if any)
    eff = adjusted if adjusted is not None else target_weights

    # ---- 5) BREADTH: max names + min position (dust) ----------------------
    active = {t: w for t, w in eff.items() if abs(w) > lim.weight_tol}
    checks["n_names"] = len(active)
    if len(active) > lim.max_names:
        violations.append(_violation(
            "MAX_NAMES", f"{len(active)} names > cap {lim.max_names}",
            n=len(active), cap=lim.max_names))
    if lim.min_position_weight > 0:
        dust = {t: w for t, w in active.items() if abs(w) < lim.min_position_weight}
        checks["dust_names"] = list(dust)
        if dust:
            violations.append(_violation(
                "MIN_POSITION_WEIGHT",
                f"{len(dust)} sub-{lim.min_position_weight:.2%} dust position(s): "
                f"{list(dust)[:10]}", names=dust, floor=lim.min_position_weight))

    # ---- 6) TURNOVER ------------------------------------------------------
    all_names = set(eff) | set(current_weights)
    turnover = sum(abs(eff.get(t, 0.0) - current_weights.get(t, 0.0)) for t in all_names)
    checks["turnover"] = turnover
    if turnover > lim.max_turnover + lim.weight_tol:
        violations.append(_violation(
            "MAX_TURNOVER", f"turnover {turnover:.4f} > cap {lim.max_turnover:.4f}",
            turnover=turnover, cap=lim.max_turnover))

    # ---- 7) NOTIONAL / ADV GUARD (per-name absolute cap) -----------------
    big = {}
    for t in all_names:
        dw = eff.get(t, 0.0) - current_weights.get(t, 0.0)
        notional = abs(dw) * nav
        if notional > lim.max_name_notional:
            big[t] = notional
    checks["over_notional_names"] = {t: round(v, 2) for t, v in big.items()}
    if big:
        violations.append(_violation(
            "MAX_NAME_NOTIONAL",
            f"{len(big)} name(s) exceed ${lim.max_name_notional:,.0f} single-order notional",
            names={t: round(v, 2) for t, v in big.items()}, cap=lim.max_name_notional))

    # ---- 8) SECTOR CONCENTRATION (only if sector data present) -----------
    sector_check = _sector_concentration(eff, lim)
    checks["sector"] = sector_check["summary"]
    if sector_check["violation"]:
        violations.append(sector_check["violation"])

    # ---- 9) DRAWDOWN + DAILY-LOSS CIRCUIT-BREAKERS ------------------------
    dd = _drawdown_and_daily(agent_id)
    checks["drawdown"] = dd
    if dd["have_history"] is None:
        # DB read failed (not merely empty) -> fail closed.
        violations.append(_violation(
            "HISTORY_UNREADABLE", "nav_history unreadable; cannot verify drawdown/daily-loss"))
    elif dd["have_history"]:
        # Drawdown breach blocks RISK-INCREASING trades (gross going up); de-risking is allowed.
        if dd["drawdown"] >= lim.max_drawdown:
            cur_gross = sum(abs(w) for w in current_weights.values())
            risk_increasing = gross > cur_gross + lim.weight_tol
            checks["drawdown_breach"] = {"dd": dd["drawdown"], "risk_increasing": risk_increasing,
                                          "cur_gross": cur_gross, "tgt_gross": gross}
            if risk_increasing:
                violations.append(_violation(
                    "DRAWDOWN_BREAKER",
                    f"trailing drawdown {dd['drawdown']:.2%} ≥ {lim.max_drawdown:.2%}; "
                    f"risk-increasing trade blocked — de-risk only "
                    f"(gross {cur_gross:.3f}->{gross:.3f})",
                    drawdown=dd["drawdown"], cap=lim.max_drawdown))
        if dd["daily_pnl_frac"] <= -abs(lim.daily_loss_limit):
            violations.append(_violation(
                "DAILY_LOSS_LIMIT",
                f"last daily P&L {dd['daily_pnl_frac']:.2%} ≤ -{abs(lim.daily_loss_limit):.2%} floor",
                daily_pnl_frac=dd["daily_pnl_frac"], floor=-abs(lim.daily_loss_limit)))

    # ---- verdict ----------------------------------------------------------
    approved = len(violations) == 0
    if approved and adjusted is not None:
        reason = f"approved with adjustment: {len(over)} weight(s) clipped to {lim.max_position_weight:.2%} and renormalized"
    elif approved:
        reason = "approved: all pre-trade risk checks passed"
    else:
        codes = ", ".join(sorted({v["code"] for v in violations}))
        reason = f"REJECTED: {len(violations)} violation(s) [{codes}]"

    # Only hand back adjusted weights when they're actually usable (approved).
    dec = Decision(approved, reason, violations,
                   adjusted if approved else None, checks)
    _persist(dec, agent_id, as_of_date, run_id, mode, target_weights, prices, nav, persist)
    return dec


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _clip_and_renormalize(weights: dict[str, float], cap: float, tol: float) -> dict[str, float]:
    """Clip any weight above ``cap`` to ``cap``, then renormalize the REMAINING
    headroom across the names still below cap, iterating until stable.

    Long-only assumption for renormalization (negative weights are left as-is;
    the shorts check handles them separately). Preserves Σw ≈ original Σw.
    """
    w = dict(weights)
    target_sum = sum(w.values())
    for _ in range(100):
        over = {t: v for t, v in w.items() if v > cap + tol}
        if not over:
            break
        excess = sum(v - cap for t, v in over.items())
        for t in over:
            w[t] = cap
        under = {t: v for t, v in w.items() if v < cap - tol and v > tol}
        pool = sum(under.values())
        if pool <= tol:
            break  # nowhere to put the excess without breaching cap
        for t in under:
            w[t] += excess * (w[t] / pool)
    # numerical touch-up to preserve the original gross/net
    cur = sum(w.values())
    if cur > tol and abs(cur - target_sum) > tol:
        scale = target_sum / cur
        # only scale names that won't breach cap after scaling
        w = {t: min(v * scale, cap) if v > tol else v for t, v in w.items()}
    return {t: v for t, v in w.items() if abs(v) > tol}


def _sector_concentration(weights: dict[str, float], lim: RiskLimits) -> dict:
    """Sum weights by sector using ``securities.sector``. Skips gracefully when
    sector data is absent/all-NULL (the case in this demo today)."""
    tickers = [t for t, w in weights.items() if abs(w) > lim.weight_tol]
    if not tickers:
        return {"summary": {"applied": False, "reason": "no target names"}, "violation": None}
    try:
        placeholders = ",".join(["%s"] * len(tickers))
        rows = db.query(
            f"SELECT ticker, sector FROM securities WHERE ticker IN ({placeholders})",
            tickers)
    except Exception:
        return {"summary": {"applied": False, "reason": "securities unreadable"}, "violation": None}
    sect = {r["ticker"]: r["sector"] for r in rows if r.get("sector")}
    if not sect:
        return {"summary": {"applied": False, "reason": "no sector data (all NULL)"}, "violation": None}
    by_sector: dict[str, float] = {}
    for t, w in weights.items():
        s = sect.get(t)
        if s:
            by_sector[s] = by_sector.get(s, 0.0) + w
    over = {s: v for s, v in by_sector.items() if v > lim.max_sector_weight + lim.weight_tol}
    summary = {"applied": True, "by_sector": {s: round(v, 4) for s, v in by_sector.items()},
               "coverage": round(len(sect) / len(tickers), 3)}
    if over:
        return {"summary": summary, "violation": _violation(
            "SECTOR_CONCENTRATION",
            f"{len(over)} sector(s) exceed {lim.max_sector_weight:.0%}: "
            f"{ {s: round(v,3) for s,v in over.items()} }",
            sectors={s: round(v, 4) for s, v in over.items()}, cap=lim.max_sector_weight)}
    return {"summary": summary, "violation": None}


def _persist(dec: Decision, agent_id: str, as_of_date: str, run_id: str | None,
             mode: str, target_weights: dict[str, float], prices: dict[str, float],
             nav: float | None, persist: bool) -> None:
    """Write the decision to trade_audit (RISK_CHECK) and risk_decisions.

    Best-effort: an audit-write failure must not crash the gate, but it IS logged
    into the returned checks so callers can see persistence failed.
    """
    if not persist:
        return
    gross = sum(abs(w) for w in target_weights.values())
    net = sum(target_weights.values())
    max_w = max(target_weights.values()) if target_weights else 0.0
    n_names = sum(1 for w in target_weights.values() if abs(w) > 1e-6)
    turnover = dec.checks.get("turnover")
    decision_id = _uid("rsk")
    try:
        db.audit(_uid("aud"), agent_id, "RISK_CHECK", run_id=run_id,
                 detail={"decision_id": decision_id, "approved": dec.approved,
                         "reason": dec.reason, "mode": mode,
                         "n_violations": len(dec.violations),
                         "violations": dec.violations, "checks": dec.checks},
                 actor="risk_gate")
    except Exception as exc:  # pragma: no cover - defensive
        dec.checks["_audit_error"] = str(exc)[:200]
    try:
        db.execute(
            """INSERT INTO risk_decisions
                   (decision_id, ts, agent_id, run_id, as_of_date, mode, approved,
                    reason, n_violations, violations, checks, adjusted, nav,
                    gross_exposure, net_exposure, turnover, max_weight, n_names, actor)
               VALUES (%s, NOW(6), %s,%s,%s,%s,%s, %s,%s,%s,%s,%s,%s, %s,%s,%s,%s,%s, 'risk_gate')""",
            (decision_id, agent_id, run_id, as_of_date, mode, 1 if dec.approved else 0,
             dec.reason[:512], len(dec.violations),
             _json(dec.violations), _json(dec.checks),
             1 if dec.adjusted_weights is not None else 0,
             float(nav) if nav else None, gross, net,
             float(turnover) if turnover is not None else None, max_w, n_names),
        )
    except Exception as exc:  # pragma: no cover - defensive
        dec.checks["_risk_decisions_error"] = str(exc)[:200]


def _json(obj) -> str:
    import json
    return json.dumps(obj, default=str)
