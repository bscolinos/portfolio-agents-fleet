"""Goldman-level trade lifecycle: weights -> orders -> fills -> positions -> NAV.

The NVIDIA optimizers speak in *weight fractions* on a notional. A trading desk
speaks in *shares*, *fills*, *commission*, *slippage*, *positions* and *P&L*.
This module is the bridge: it takes a target weight vector for a rebalance date
and produces the full front-to-back record a compliance desk would expect,
persisting every step to SingleStore with an immutable audit event.

Conventions (backtest-realistic, deliberately simple and auditable):
  * Orders are Market-On-Close (MOC) against the decision-date adjusted close.
  * Slippage is modeled in bps on the traded notional (buys pay up, sells give up).
  * Commission is a per-share rate with a per-order minimum.
  * Positions carry cost basis; realized P&L books on sells.
  * NAV = cash + Σ market_value, marked at the as-of close.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence

from . import db


# --------------------------------------------------------------------------
# Cost model
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CostModel:
    commission_per_share: float = 0.005   # $0.005/share (institutional)
    min_commission: float = 1.0           # $1 per order floor
    slippage_bps: float = 2.0             # 2 bps modeled market impact
    venue: str = "SIM"

    def commission(self, qty: float) -> float:
        return max(self.min_commission, abs(qty) * self.commission_per_share)

    def fill_price(self, ref_price: float, side: str) -> float:
        # Buys cross the spread up, sells down, by slippage_bps.
        adj = ref_price * (self.slippage_bps / 1e4)
        return ref_price + adj if side == "BUY" else ref_price - adj


def _uid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:16]}"


# --------------------------------------------------------------------------
# Rebalance: translate target weights into the full trade lifecycle
# --------------------------------------------------------------------------

@dataclass
class RebalanceResult:
    run_id: str
    agent_id: str
    as_of_date: str
    nav_before: float
    nav_after: float
    cash: float
    invested: float
    turnover: float
    tcost: float
    n_orders: int
    n_positions: int
    realized_pnl: float
    unrealized_pnl: float


def rebalance(
    *,
    run_id: str,
    agent_id: str,
    as_of_date: str,
    target_weights: dict[str, float],
    prices: dict[str, float],
    cost_model: CostModel | None = None,
    starting_nav: float | None = None,
) -> RebalanceResult:
    """Execute one rebalance for ``agent_id`` on ``as_of_date``.

    ``target_weights``  ticker -> desired fraction of NAV (from the optimizer).
    ``prices``          ticker -> as-of adjusted close (decision + mark price).
    ``starting_nav``    only used to seed the very first rebalance (inception).

    Reads the agent's current positions/cash from SingleStore, computes the
    share deltas to reach ``target_weights``, books orders + fills (with cost),
    updates positions, marks NAV, and appends audit + risk rows. Idempotent per
    ``run_id`` in the sense that each call writes its own run-scoped rows.
    """
    cm = cost_model or CostModel()
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")

    # ---- current book -------------------------------------------------------
    pos_rows = db.query(
        "SELECT ticker, qty, avg_cost, realized_pnl FROM positions WHERE agent_id=%s",
        (agent_id,),
    )
    book = {r["ticker"]: r for r in pos_rows}

    nav_rows = db.query(
        "SELECT nav, cash FROM nav_history WHERE agent_id=%s ORDER BY as_of_date DESC LIMIT 1",
        (agent_id,),
    )
    if nav_rows:
        cash = float(nav_rows[0]["cash"])
        # mark current positions at as-of prices for a fair pre-trade NAV
        mv = sum(float(r["qty"]) * prices.get(t, float(r["avg_cost"]))
                 for t, r in book.items())
        nav_before = cash + mv
    else:
        nav_before = float(starting_nav if starting_nav is not None else 100_000_000.0)
        cash = nav_before

    db.audit(_uid("aud"), agent_id, "RUN_START", run_id=run_id,
             detail={"as_of_date": as_of_date, "nav_before": nav_before,
                     "n_targets": len(target_weights)})

    # ---- desired shares -----------------------------------------------------
    # Target dollars per name, then shares at the ref price.
    target_qty: dict[str, float] = {}
    for t, w in target_weights.items():
        px = prices.get(t)
        if not px or px <= 0 or w <= 0:
            continue
        target_qty[t] = (w * nav_before) / px

    all_tickers = set(book) | set(target_qty)

    orders: list[tuple] = []
    fills: list[tuple] = []
    audit_rows: list[tuple] = []
    turnover_dollars = 0.0
    total_tcost = 0.0
    realized_pnl_delta = 0.0

    for t in sorted(all_tickers):
        px = prices.get(t)
        cur = book.get(t)
        cur_qty = float(cur["qty"]) if cur else 0.0
        tgt = target_qty.get(t, 0.0)
        if px is None or px <= 0:
            # can't trade or mark without a price; hold existing qty
            continue
        delta = tgt - cur_qty
        if abs(delta * px) < 1.0:   # skip sub-$1 dust trades
            continue

        side = "BUY" if delta > 0 else "SELL"
        ref_price = px
        fill_price = cm.fill_price(ref_price, side)
        notional = abs(delta) * fill_price
        commission = cm.commission(delta)
        prev_w = (cur_qty * px) / nav_before if nav_before else 0.0
        tgt_w = target_weights.get(t, 0.0)

        order_id = _uid("ord")
        exec_id = _uid("exc")
        orders.append((
            order_id, run_id, agent_id, as_of_date, t, side, tgt_w, prev_w,
            delta, ref_price, abs(delta) * ref_price, "MOC", "DAY", "FILLED", ts,
        ))
        fills.append((
            exec_id, order_id, run_id, agent_id, as_of_date, t, side,
            delta, fill_price, notional, commission, cm.slippage_bps, cm.venue, ts,
        ))

        # ---- cash + position bookkeeping ----------------------------------
        if side == "BUY":
            cash -= (notional + commission)
            new_qty = cur_qty + delta
            prev_cost = (cur["avg_cost"] * cur_qty) if cur else 0.0
            avg_cost = (prev_cost + notional) / new_qty if new_qty else 0.0
        else:  # SELL
            cash += (notional - commission)
            sold = -delta  # positive number of shares sold
            basis = (cur["avg_cost"] if cur else fill_price)
            realized = (fill_price - basis) * sold - commission
            realized_pnl_delta += realized
            new_qty = cur_qty + delta
            avg_cost = cur["avg_cost"] if cur and new_qty > 0 else 0.0
            if cur:
                cur["realized_pnl"] = float(cur["realized_pnl"]) + realized

        turnover_dollars += notional
        total_tcost += commission + abs(delta) * (fill_price - ref_price) * (1 if side == "BUY" else -1)

        book[t] = {
            "ticker": t, "qty": new_qty, "avg_cost": avg_cost,
            "realized_pnl": (float(cur["realized_pnl"]) if cur else 0.0),
        }
        audit_rows.append((
            _uid("aud"), ts, agent_id, run_id, "FILL", exec_id, t,
            db_json({"side": side, "qty": delta, "fill_price": fill_price,
                     "commission": commission}), "agent",
        ))

    # ---- persist orders + fills + audit ------------------------------------
    if orders:
        db.executemany(
            """INSERT INTO orders
               (order_id, run_id, agent_id, as_of_date, ticker, side, target_weight,
                prev_weight, order_qty, ref_price, order_notional, order_type, tif,
                status, created_at)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            orders,
        )
    if fills:
        db.executemany(
            """INSERT INTO executions
               (exec_id, order_id, run_id, agent_id, as_of_date, ticker, side,
                fill_qty, fill_price, notional, commission, slippage_bps, venue, executed_at)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            fills,
        )
    if audit_rows:
        db.executemany(
            """INSERT INTO trade_audit
               (audit_id, ts, agent_id, run_id, event_type, entity_ref, ticker, detail, actor)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            audit_rows,
        )

    # ---- rebuild positions + snapshots -------------------------------------
    invested = 0.0
    unreal = 0.0
    pos_upserts: list[tuple] = []
    snap_rows: list[tuple] = []
    # first pass to compute NAV for weights
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

    for t, (qty, px, mv, avg_cost, rpnl) in marked.items():
        weight = mv / nav_after if nav_after else 0.0
        unreal += (px - avg_cost) * qty
        pos_upserts.append((agent_id, t, qty, avg_cost, px, mv, weight, rpnl, as_of_date, ts))
        snap_rows.append((agent_id, as_of_date, t, qty, px, mv, weight))

    # clear positions that went to zero
    db.execute("DELETE FROM positions WHERE agent_id=%s", (agent_id,))
    if pos_upserts:
        db.executemany(
            """INSERT INTO positions
               (agent_id, ticker, qty, avg_cost, last_price, market_value, weight,
                realized_pnl, as_of_date, updated_at)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            pos_upserts,
        )
    if snap_rows:
        db.executemany(
            """INSERT INTO position_snapshots
               (agent_id, as_of_date, ticker, qty, last_price, market_value, weight)
               VALUES (%s,%s,%s,%s,%s,%s,%s)""",
            snap_rows,
        )

    # ---- NAV history --------------------------------------------------------
    prior = db.query(
        "SELECT nav FROM nav_history WHERE agent_id=%s ORDER BY as_of_date DESC LIMIT 1",
        (agent_id,),
    )
    inception = db.query(
        "SELECT nav FROM nav_history WHERE agent_id=%s ORDER BY as_of_date ASC LIMIT 1",
        (agent_id,),
    )
    daily_return = (nav_after / float(prior[0]["nav"]) - 1.0) if prior else 0.0
    base_nav = float(inception[0]["nav"]) if inception else nav_before
    cum_return = (nav_after / base_nav - 1.0) if base_nav else 0.0
    turnover_frac = turnover_dollars / nav_before if nav_before else 0.0

    db.execute(
        """INSERT INTO nav_history
           (agent_id, as_of_date, nav, cash, invested, daily_return, cum_return,
            realized_pnl, unrealized_pnl, turnover, tcost)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
        (agent_id, as_of_date, nav_after, cash, invested, daily_return, cum_return,
         realized_pnl_delta, unreal, turnover_frac, total_tcost),
    )

    db.audit(_uid("aud"), agent_id, "NAV", run_id=run_id,
             detail={"nav_after": nav_after, "cash": cash, "invested": invested,
                     "turnover": turnover_frac, "tcost": total_tcost,
                     "daily_return": daily_return, "cum_return": cum_return})
    db.audit(_uid("aud"), agent_id, "RUN_END", run_id=run_id,
             detail={"n_orders": len(orders), "n_positions": len(marked)})

    return RebalanceResult(
        run_id=run_id, agent_id=agent_id, as_of_date=as_of_date,
        nav_before=nav_before, nav_after=nav_after, cash=cash, invested=invested,
        turnover=turnover_frac, tcost=total_tcost, n_orders=len(orders),
        n_positions=len(marked), realized_pnl=realized_pnl_delta,
        unrealized_pnl=unreal,
    )


def mark_to_market(agent_id: str, as_of_date: str, prices: dict[str, float]) -> float | None:
    """Re-mark an agent's held positions at ``prices`` and append a NAV row.

    Used on non-rebalance dates so the equity curve is dense (daily) without
    trading. Returns the new NAV, or None if the agent has no book yet.
    """
    pos = db.query(
        "SELECT ticker, qty, avg_cost, realized_pnl FROM positions WHERE agent_id=%s", (agent_id,))
    if not pos:
        return None
    nav_rows = db.query(
        "SELECT cash FROM nav_history WHERE agent_id=%s ORDER BY as_of_date DESC LIMIT 1",
        (agent_id,))
    cash = float(nav_rows[0]["cash"]) if nav_rows else 0.0
    invested = 0.0
    unreal = 0.0
    snap_rows = []
    marked = []
    for r in pos:
        qty = float(r["qty"])
        px = prices.get(r["ticker"], float(r["avg_cost"]))
        mv = qty * px
        invested += mv
        unreal += (px - float(r["avg_cost"])) * qty
        marked.append((r["ticker"], qty, px, mv, float(r["avg_cost"]), float(r.get("realized_pnl", 0.0))))
    nav = cash + invested
    for ticker, qty, px, mv, _, _ in marked:
        snap_rows.append((agent_id, as_of_date, ticker, qty, px, mv, mv / nav if nav else 0.0))
    prior = db.query(
        "SELECT nav FROM nav_history WHERE agent_id=%s ORDER BY as_of_date DESC LIMIT 1",
        (agent_id,))
    inception = db.query(
        "SELECT nav FROM nav_history WHERE agent_id=%s ORDER BY as_of_date ASC LIMIT 1",
        (agent_id,))
    daily_return = (nav / float(prior[0]["nav"]) - 1.0) if prior else 0.0
    base = float(inception[0]["nav"]) if inception else nav
    cum = (nav / base - 1.0) if base else 0.0
    db.execute(
        """INSERT INTO nav_history
           (agent_id, as_of_date, nav, cash, invested, daily_return, cum_return,
            realized_pnl, unrealized_pnl, turnover, tcost)
           VALUES (%s,%s,%s,%s,%s,%s,%s,0,%s,0,0)""",
        (agent_id, as_of_date, nav, cash, invested, daily_return, cum, unreal),
    )
    if snap_rows:
        db.executemany(
            """INSERT INTO position_snapshots
               (agent_id, as_of_date, ticker, qty, last_price, market_value, weight)
               VALUES (%s,%s,%s,%s,%s,%s,%s)""",
            snap_rows,
        )
    # keep the live positions' last_price/market_value fresh too — rebuild in
    # one DELETE + batch INSERT (avoids a per-position UPDATE round-trip storm).
    db.execute("DELETE FROM positions WHERE agent_id=%s", (agent_id,))
    db.executemany(
        """INSERT INTO positions
           (agent_id, ticker, qty, avg_cost, last_price, market_value, weight,
            realized_pnl, as_of_date, updated_at)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW(6))""",
        [(agent_id, ticker, qty, avg_cost, px, mv, mv / nav if nav else 0.0,
          rpnl, as_of_date) for (ticker, qty, px, mv, avg_cost, rpnl) in marked],
    )
    return nav


def db_json(obj) -> str:
    import json
    return json.dumps(obj)
