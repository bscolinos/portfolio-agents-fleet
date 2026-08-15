"""Paper-trading harness — the "paper-trade first" gate before any live decision.

This is the execution side of the safety layer. Every proposed rebalance runs
through :func:`pa_agents.risk_gate.evaluate` FIRST; only if approved does it
touch a book — and by default that book is a SHADOW (paper) book, never the real
one.

    gated_rebalance(agent_id, as_of_date, target_weights, prices,
                    nav=None, mode='paper', limits=None) -> dict

Modes:
  * ``paper`` (default) — routes approved orders to ``paper_orders`` /
    ``paper_positions`` / ``paper_nav_history``. NEVER writes the real
    orders/positions/nav_history tables. An operator can watch the paper equity
    curve build before authorizing anything live.
  * ``live`` — requires an EXPLICIT ``allow_live=True`` AND a disengaged kill
    switch AND gate approval, then calls the existing internal simulated book via
    :func:`pa_agents.trading.rebalance`. There is deliberately NO real broker:
    a broker adapter is the ONE remaining step and is intentionally not
    implemented here (documented in RISK_CONTROLS.md). The GATE is identical for
    paper and live — only the execution sink differs.

The paper fill math MIRRORS ``trading.py`` EXACTLY: it imports and uses the same
:class:`pa_agents.trading.CostModel` (commission_per_share, min_commission,
slippage_bps) and the same MOC-against-adjusted-close, cost-basis, realized-P&L
and NAV-marking conventions. It is a faithful shadow of the real lifecycle,
writing to the paper_* tables instead.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from . import db
from . import kill_switch
from . import risk_gate
from .risk_gate import RiskLimits
from .trading import CostModel


def _uid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:16]}"


# --------------------------------------------------------------------------
# Gated rebalance — the single entry point
# --------------------------------------------------------------------------

def gated_rebalance(
    *,
    agent_id: str,
    as_of_date: str,
    target_weights: dict[str, float],
    prices: dict[str, float],
    nav: float | None = None,
    mode: str = "paper",
    limits: RiskLimits | None = None,
    cost_model: CostModel | None = None,
    run_id: str | None = None,
    allow_live: bool = False,
    starting_nav: float | None = None,
) -> dict:
    """Gate a proposed rebalance and, if approved, route it to the right book.

    Returns a dict describing what happened::

        {"approved": bool, "mode": str, "reason": str, "violations": [...],
         "adjusted": bool, "result": {...}|None}

    ``result`` carries the paper fill summary (paper mode) or the RebalanceResult
    fields (live mode); it is None when the trade was rejected.
    """
    run_id = run_id or _uid(f"gate-{agent_id}")
    decision = risk_gate.evaluate(
        agent_id=agent_id, as_of_date=as_of_date, target_weights=target_weights,
        prices=prices, nav=nav, limits=limits, run_id=run_id, mode=mode)

    out = {
        "approved": decision.approved,
        "mode": mode,
        "reason": decision.reason,
        "violations": decision.violations,
        "adjusted": decision.adjusted_weights is not None,
        "run_id": run_id,
        "result": None,
    }
    if not decision.approved:
        # Rejection already persisted by the gate (trade_audit + risk_decisions).
        return out

    weights = decision.adjusted_weights or target_weights

    if mode == "paper":
        out["result"] = _paper_rebalance(
            run_id=run_id, agent_id=agent_id, as_of_date=as_of_date,
            target_weights=weights, prices=prices,
            cost_model=cost_model or CostModel(), starting_nav=starting_nav)
        return out

    if mode == "live":
        # Belt-and-suspenders: re-check the kill switch immediately before any
        # live write (the gate checked it too, but time may have passed).
        ks = kill_switch.is_engaged(agent_id)
        if ks["engaged"]:
            out["approved"] = False
            out["reason"] = f"KILL SWITCH ENGAGED at live-execution time: {ks['reason']}"
            return out
        if not allow_live:
            out["approved"] = False
            out["reason"] = ("live mode requires explicit allow_live=True — refusing to "
                             "trade the real book without it (fail-closed)")
            db.audit(_uid("aud"), agent_id, "RISK_CHECK", run_id=run_id,
                     detail={"blocked": "allow_live_false", "mode": "live"},
                     actor="paper_trader")
            return out
        # Approved + allowed + kill switch clear -> internal SIMULATED book.
        # NOTE: this is trading.rebalance, the same in-DB simulated lifecycle the
        # backtest uses. A REAL broker adapter (FIX/REST to an execution venue)
        # is the ONLY remaining step to move real capital and is INTENTIONALLY
        # NOT implemented here. See RISK_CONTROLS.md "what is still NOT covered".
        from .trading import rebalance as _live_rebalance
        reb = _live_rebalance(
            run_id=run_id, agent_id=agent_id, as_of_date=as_of_date,
            target_weights=weights, prices=prices,
            cost_model=cost_model or CostModel(), starting_nav=starting_nav)
        out["result"] = {
            "run_id": reb.run_id, "nav_before": reb.nav_before, "nav_after": reb.nav_after,
            "cash": reb.cash, "invested": reb.invested, "turnover": reb.turnover,
            "tcost": reb.tcost, "n_orders": reb.n_orders, "n_positions": reb.n_positions,
            "realized_pnl": reb.realized_pnl, "unrealized_pnl": reb.unrealized_pnl,
        }
        return out

    raise ValueError(f"unknown mode {mode!r} (expected 'paper' or 'live')")


# --------------------------------------------------------------------------
# Paper (shadow) rebalance — mirrors trading.rebalance into the paper_* tables
# --------------------------------------------------------------------------

def _paper_rebalance(
    *,
    run_id: str,
    agent_id: str,
    as_of_date: str,
    target_weights: dict[str, float],
    prices: dict[str, float],
    cost_model: CostModel,
    starting_nav: float | None = None,
) -> dict:
    """A faithful shadow of :func:`pa_agents.trading.rebalance`, writing to the
    paper_* tables. Same CostModel, same MOC/cost-basis/realized-P&L/NAV math."""
    cm = cost_model
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")

    # ---- current PAPER book -------------------------------------------------
    pos_rows = db.query(
        "SELECT ticker, qty, avg_cost, realized_pnl FROM paper_positions WHERE agent_id=%s",
        (agent_id,))
    book = {r["ticker"]: dict(r) for r in pos_rows}

    nav_rows = db.query(
        "SELECT nav, cash FROM paper_nav_history WHERE agent_id=%s "
        "ORDER BY as_of_date DESC LIMIT 1", (agent_id,))
    if nav_rows:
        cash = float(nav_rows[0]["cash"])
        mv = sum(float(r["qty"]) * prices.get(t, float(r["avg_cost"]))
                 for t, r in book.items())
        nav_before = cash + mv
    else:
        nav_before = float(starting_nav if starting_nav is not None else 100_000_000.0)
        cash = nav_before

    # ---- desired shares -----------------------------------------------------
    target_qty: dict[str, float] = {}
    for t, w in target_weights.items():
        px = prices.get(t)
        if not px or px <= 0 or w <= 0:
            continue
        target_qty[t] = (w * nav_before) / px

    all_tickers = set(book) | set(target_qty)
    paper_orders: list[tuple] = []
    turnover_dollars = 0.0
    total_tcost = 0.0
    realized_pnl_delta = 0.0

    for t in sorted(all_tickers):
        px = prices.get(t)
        cur = book.get(t)
        cur_qty = float(cur["qty"]) if cur else 0.0
        tgt = target_qty.get(t, 0.0)
        if px is None or px <= 0:
            continue
        delta = tgt - cur_qty
        if abs(delta * px) < 1.0:   # skip sub-$1 dust trades (same as trading.py)
            continue

        side = "BUY" if delta > 0 else "SELL"
        ref_price = px
        fill_price = cm.fill_price(ref_price, side)
        notional = abs(delta) * fill_price
        commission = cm.commission(delta)
        prev_w = (cur_qty * px) / nav_before if nav_before else 0.0
        tgt_w = target_weights.get(t, 0.0)

        if side == "BUY":
            cash -= (notional + commission)
            new_qty = cur_qty + delta
            prev_cost = (cur["avg_cost"] * cur_qty) if cur else 0.0
            avg_cost = (prev_cost + notional) / new_qty if new_qty else 0.0
        else:  # SELL
            cash += (notional - commission)
            sold = -delta
            basis = (cur["avg_cost"] if cur else fill_price)
            realized = (fill_price - basis) * sold - commission
            realized_pnl_delta += realized
            new_qty = cur_qty + delta
            avg_cost = cur["avg_cost"] if cur and new_qty > 0 else 0.0

        turnover_dollars += notional
        total_tcost += commission + abs(delta) * (fill_price - ref_price) * (1 if side == "BUY" else -1)

        book[t] = {
            "ticker": t, "qty": new_qty, "avg_cost": avg_cost,
            "realized_pnl": (float(cur["realized_pnl"]) if cur else 0.0)
                            + (realized if side == "SELL" else 0.0),
        }
        paper_orders.append((
            _uid("pord"), run_id, agent_id, as_of_date, t, side, tgt_w, prev_w,
            delta, ref_price, abs(delta) * ref_price, fill_price, notional,
            commission, cm.slippage_bps, "PAPER", "FILLED", ts,
        ))

    if paper_orders:
        db.executemany(
            """INSERT INTO paper_orders
               (order_id, run_id, agent_id, as_of_date, ticker, side, target_weight,
                prev_weight, order_qty, ref_price, order_notional, fill_price, notional,
                commission, slippage_bps, venue, status, created_at)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            paper_orders)

    # ---- rebuild PAPER positions + NAV -------------------------------------
    invested = 0.0
    unreal = 0.0
    marked = {}
    for t, r in book.items():
        qty = float(r["qty"])
        if abs(qty) < 1e-9:
            continue
        px = prices.get(t, float(r["avg_cost"]))
        mv = qty * px
        marked[t] = (qty, px, mv, float(r["avg_cost"]), float(r.get("realized_pnl", 0.0)))
        invested += mv
    nav_after = cash + invested

    pos_upserts = []
    for t, (qty, px, mv, avg_cost, rpnl) in marked.items():
        weight = mv / nav_after if nav_after else 0.0
        unreal += (px - avg_cost) * qty
        pos_upserts.append((agent_id, t, qty, avg_cost, px, mv, weight, rpnl, as_of_date, ts))

    db.execute("DELETE FROM paper_positions WHERE agent_id=%s", (agent_id,))
    if pos_upserts:
        db.executemany(
            """INSERT INTO paper_positions
               (agent_id, ticker, qty, avg_cost, last_price, market_value, weight,
                realized_pnl, as_of_date, updated_at)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            pos_upserts)

    prior = db.query(
        "SELECT nav FROM paper_nav_history WHERE agent_id=%s ORDER BY as_of_date DESC LIMIT 1",
        (agent_id,))
    inception = db.query(
        "SELECT nav FROM paper_nav_history WHERE agent_id=%s ORDER BY as_of_date ASC LIMIT 1",
        (agent_id,))
    daily_return = (nav_after / float(prior[0]["nav"]) - 1.0) if prior else 0.0
    base_nav = float(inception[0]["nav"]) if inception else nav_before
    cum_return = (nav_after / base_nav - 1.0) if base_nav else 0.0
    turnover_frac = turnover_dollars / nav_before if nav_before else 0.0

    db.execute(
        """INSERT INTO paper_nav_history
           (agent_id, as_of_date, nav, cash, invested, daily_return, cum_return,
            realized_pnl, unrealized_pnl, turnover, tcost)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
        (agent_id, as_of_date, nav_after, cash, invested, daily_return, cum_return,
         realized_pnl_delta, unreal, turnover_frac, total_tcost))

    db.audit(_uid("aud"), agent_id, "PAPER_FILL", run_id=run_id,
             detail={"nav_before": nav_before, "nav_after": nav_after, "cash": cash,
                     "invested": invested, "turnover": turnover_frac, "tcost": total_tcost,
                     "n_orders": len(paper_orders), "n_positions": len(marked)},
             actor="paper_trader")

    return {
        "run_id": run_id, "agent_id": agent_id, "as_of_date": as_of_date,
        "book": "paper", "nav_before": nav_before, "nav_after": nav_after,
        "cash": cash, "invested": invested, "turnover": turnover_frac,
        "tcost": total_tcost, "n_orders": len(paper_orders), "n_positions": len(marked),
        "realized_pnl": realized_pnl_delta, "unrealized_pnl": unreal,
    }


# --------------------------------------------------------------------------
# Promote a re-scored candidate through the gate in paper mode
# --------------------------------------------------------------------------

def promote_candidate(
    *,
    agent_id: str,
    as_of_date: str,
    target_weights: dict[str, float],
    prices: dict[str, float],
    nav: float | None = None,
    limits: RiskLimits | None = None,
    run_id: str | None = None,
) -> dict:
    """Run the re-scored best strategy through the gate in PAPER mode so an
    operator can see the paper equity curve before any live decision.

    Thin wrapper over :func:`gated_rebalance` (mode='paper'); the point is to give
    the promotion flow a clearly-named, safe entry point that can NEVER touch the
    real book.
    """
    return gated_rebalance(
        agent_id=agent_id, as_of_date=as_of_date, target_weights=target_weights,
        prices=prices, nav=nav, mode="paper", limits=limits, run_id=run_id)
