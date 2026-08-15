"""Fleet orchestrator + CLI — drives the whole multi-agent backtest.

Run on the GPU box:
    python -m pa_agents.fleet load-prices --csv ~/portfolio-optimization/data/stock_data/sp500.csv
    python -m pa_agents.fleet backtest --start 2023-01-01 --end 2024-12-31 \
        --universe 60 --rebalance-freq 21 --lookback 252

The backtest walks trading dates forward. On each rebalance date every agent
runs a full memory->solve->trade->reflect cycle (GPU cuOpt/cuML where available);
on intervening dates every agent is marked-to-market so the equity curves are
daily. All state persists to SingleStore — kill the process mid-run and the
agents resume with intact memory + books.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from . import db
from .runner import (ensure_repo_on_path, register_agents, run_rebalance,
                     load_returns)
from .trading import mark_to_market
from .strategies import ROSTER


# --------------------------------------------------------------------------
# Price loading (from the NVIDIA blueprint's sp500.csv)
# --------------------------------------------------------------------------

def load_prices_csv(csv_path: str, *, batch: int = 20000) -> dict:
    """Load a wide (date x ticker adj-close) CSV into the prices table.

    Computes per-ticker daily simple returns. Handles both wide format
    (index=date, columns=tickers) and long format (date,ticker,adj_close).
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    # detect format
    lower = {c.lower(): c for c in df.columns}
    if {"date", "ticker"} <= set(lower) and any(k in lower for k in ("adj_close", "close", "adjclose")):
        # long format
        dcol = lower["date"]; tcol = lower["ticker"]
        pcol = lower.get("adj_close") or lower.get("adjclose") or lower["close"]
        long = df[[dcol, tcol, pcol]].rename(columns={dcol: "date", tcol: "ticker", pcol: "px"})
    else:
        # wide format: first column is the date index
        date_col = df.columns[0]
        df = df.rename(columns={date_col: "date"})
        long = df.melt(id_vars="date", var_name="ticker", value_name="px")
    long = long.dropna(subset=["px"])
    long["date"] = pd.to_datetime(long["date"]).dt.strftime("%Y-%m-%d")
    long = long.sort_values(["ticker", "date"])
    long["ret"] = long.groupby("ticker")["px"].pct_change()

    tickers = sorted(long["ticker"].unique().tolist())
    # register securities
    db.executemany(
        "INSERT INTO securities (ticker, is_active) VALUES (%s,1) "
        "ON DUPLICATE KEY UPDATE is_active=1",
        [(t,) for t in tickers],
    )
    # insert prices in batches
    rows = list(long[["ticker", "date", "px", "ret"]].itertuples(index=False, name=None))
    rows = [(t, d, float(px), (None if (r != r) else float(r))) for (t, d, px, r) in rows]
    total = 0
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        db.executemany(
            "INSERT INTO prices (ticker, trade_date, adj_close, daily_return) VALUES (%s,%s,%s,%s)",
            chunk,
        )
        total += len(chunk)
    return {"tickers": len(tickers), "rows": total,
            "date_min": long["date"].min(), "date_max": long["date"].max()}


def trading_dates(start: str, end: str) -> list[str]:
    rows = db.query(
        "SELECT DISTINCT trade_date FROM prices WHERE trade_date BETWEEN %s AND %s ORDER BY trade_date",
        (start, end))
    return [r["trade_date"].strftime("%Y-%m-%d") if hasattr(r["trade_date"], "strftime")
            else str(r["trade_date"]) for r in rows]


def top_universe(n: int, as_of: str) -> list[str]:
    """Pick the N most-liquid/complete tickers with data as of ``as_of``.

    Uses longest available history as a simple liquidity/completeness proxy.
    """
    rows = db.query(
        """SELECT ticker, COUNT(*) c FROM prices WHERE trade_date <= %s
           GROUP BY ticker HAVING c > 200 ORDER BY c DESC, ticker LIMIT %s""",
        (as_of, int(n)))
    return [r["ticker"] for r in rows]


def prices_on(date: str, tickers: list[str]) -> dict:
    rows = db.query(
        "SELECT ticker, adj_close FROM prices WHERE trade_date=%s AND ticker IN (%s)"
        % ("%s", ",".join(["%s"] * len(tickers))),
        [date, *tickers])
    return {r["ticker"]: float(r["adj_close"]) for r in rows}


# --------------------------------------------------------------------------
# Backtest driver
# --------------------------------------------------------------------------

def run_backtest(*, start: str, end: str, universe_n: int, rebalance_freq: int,
                 lookback: int, starting_nav: float = 100_000_000.0,
                 reflect: bool = True, agents: list[str] | None = None,
                 enforce_risk: bool = False) -> dict:
    ensure_repo_on_path()
    register_agents()
    agent_ids = agents or [a.agent_id for a in ROSTER]

    dates = trading_dates(start, end)
    if len(dates) < lookback // 4:
        raise SystemExit(f"Not enough trading dates in range ({len(dates)}). Load prices first?")

    universe = top_universe(universe_n, dates[-1])
    print(f"[fleet] {len(dates)} dates, universe={len(universe)}, agents={agent_ids}")

    all_dates = trading_dates("1900-01-01", end)  # for lookback windows before `start`
    date_pos = {d: i for i, d in enumerate(all_dates)}

    summary = {a: {"runs": 0, "gpu": 0, "cpu": 0, "solve_ms": 0.0} for a in agent_ids}
    n_rebal = 0
    for di, d in enumerate(dates):
        px = prices_on(d, universe)
        if not px:
            continue
        is_rebalance = (di % rebalance_freq == 0)
        if is_rebalance:
            n_rebal += 1
            gi = date_pos.get(d, 0)
            lb_start = all_dates[max(0, gi - lookback)]
            lb_end = all_dates[max(0, gi - 1)]
            for aid in agent_ids:
                t0 = time.perf_counter()
                try:
                    r = run_rebalance(
                        aid, as_of_date=d, lookback_start=lb_start, lookback_end=lb_end,
                        universe=universe, prices=px,
                        starting_nav=starting_nav, do_reflect=reflect,
                        enforce_risk=enforce_risk)
                    if r.get("blocked"):
                        print(f"  {d} {aid:14s} BLOCKED by risk gate: {r.get('reason','')[:80]}")
                        continue
                    summary[aid]["runs"] += 1
                    summary[aid]["solve_ms"] += r["solve_ms"] or 0.0
                    summary[aid]["gpu" if r["engine"] == "gpu" else "cpu"] += 1
                    print(f"  {d} {aid:14s} eng={r['engine']:3s} solve={r['solve_ms']:7.1f}ms "
                          f"scen={r['num_scenarios']:5d} orders={r['n_orders']:3d} "
                          f"nav={r['nav_after']:,.0f} sharpe={r['sharpe'] or 0:.2f} recall={r['recalled']}")
                except Exception as exc:
                    print(f"  {d} {aid:14s} FAILED: {exc}")
        else:
            for aid in agent_ids:
                mark_to_market(aid, d, px)

    print(f"[fleet] done. rebalances={n_rebal}")
    for aid, s in summary.items():
        avg = s["solve_ms"] / s["runs"] if s["runs"] else 0
        print(f"  {aid:14s} runs={s['runs']} gpu={s['gpu']} cpu={s['cpu']} avg_solve={avg:.1f}ms")
    return {"dates": len(dates), "rebalances": n_rebal, "universe": len(universe),
            "summary": summary}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser("pa_agents.fleet")
    sub = ap.add_subparsers(dest="cmd", required=True)

    lp = sub.add_parser("load-prices")
    lp.add_argument("--csv", required=True)

    bt = sub.add_parser("backtest")
    bt.add_argument("--start", required=True)
    bt.add_argument("--end", required=True)
    bt.add_argument("--universe", type=int, default=60)
    bt.add_argument("--rebalance-freq", type=int, default=21)
    bt.add_argument("--lookback", type=int, default=252)
    bt.add_argument("--starting-nav", type=float, default=100_000_000.0)
    bt.add_argument("--no-reflect", action="store_true")
    bt.add_argument("--enforce-risk", action="store_true",
                    help="insert the pre-trade risk gate (off by default so a plain "
                         "backtest replay is unaffected)")
    bt.add_argument("--agents", default="")

    sub.add_parser("register")

    args = ap.parse_args(argv)
    if args.cmd == "load-prices":
        print(json.dumps(load_prices_csv(args.csv), indent=2))
    elif args.cmd == "register":
        register_agents(); print("agents registered")
    elif args.cmd == "backtest":
        agents = [a for a in args.agents.split(",") if a] or None
        run_backtest(start=args.start, end=args.end, universe_n=args.universe,
                     rebalance_freq=args.rebalance_freq, lookback=args.lookback,
                     starting_nav=args.starting_nav, reflect=not args.no_reflect,
                     agents=agents, enforce_risk=args.enforce_risk)


if __name__ == "__main__":
    main()
