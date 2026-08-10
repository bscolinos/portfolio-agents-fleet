"""Local smoke test of the non-GPU path against the real SingleStore workspace.

Seeds a tiny synthetic price panel, then runs equal-weight + risk-parity agents
(pure NumPy, no cvxpy/GPU) through the full memory->solve->trade->NAV->audit
loop. Proves db.py / trading.py / runner.py and vector memory recall work
end-to-end before the GPU fleet runs the real backtest.
"""
import os, sys, math, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pa_agents import db
from pa_agents.runner import register_agents, run_rebalance
from pa_agents.trading import mark_to_market

random.seed(7)
TICKERS = [f"T{i:02d}" for i in range(12)]
DATES = [f"2024-01-{d:02d}" for d in range(2, 25)]  # 23 business-ish days

def seed_prices():
    db.execute("DELETE FROM prices WHERE ticker LIKE 'T%'")
    db.executemany("INSERT INTO securities (ticker,is_active) VALUES (%s,1) ON DUPLICATE KEY UPDATE is_active=1",
                   [(t,) for t in TICKERS])
    rows = []
    px = {t: 100.0 + random.random()*50 for t in TICKERS}
    prev = dict(px)
    for i, d in enumerate(DATES):
        for t in TICKERS:
            drift = 0.0004 + (hash(t) % 5) * 0.0002
            shock = random.gauss(0, 0.012)
            px[t] = max(1.0, px[t] * (1 + drift + shock))
            ret = None if i == 0 else px[t]/prev[t]-1
            rows.append((t, d, round(px[t],4), ret))
        prev = dict(px)
    db.executemany("INSERT INTO prices (ticker,trade_date,adj_close,daily_return) VALUES (%s,%s,%s,%s)", rows)
    print(f"seeded {len(rows)} price rows over {len(DATES)} dates, {len(TICKERS)} tickers")

def prices_on(d):
    return {r["ticker"]: float(r["adj_close"]) for r in
            db.query("SELECT ticker,adj_close FROM prices WHERE trade_date=%s AND ticker LIKE 'T%%'", (d,))}

def clean_agent(aid):
    for tbl in ("agent_runs","agent_memory","orders","executions","positions",
                "position_snapshots","nav_history","risk_metrics","trade_audit"):
        db.execute(f"DELETE FROM {tbl} WHERE agent_id=%s", (aid,))

def main():
    seed_prices()
    register_agents()
    agents = ["equal-weight", "risk-parity"]
    for a in agents:
        clean_agent(a)
    # rebalance on day 8 and day 15, mark-to-market in between
    rebal_dates = {DATES[7], DATES[14]}
    for i, d in enumerate(DATES[7:], start=7):
        px = prices_on(d)
        if not px:
            continue
        if d in rebal_dates:
            for a in agents:
                r = run_rebalance(a, as_of_date=d, lookback_start=DATES[0], lookback_end=DATES[i-1],
                                  universe=TICKERS, prices=px, starting_nav=100_000_000.0, do_reflect=False)
                print(f"  {d} {a:12s} eng={r['engine']} orders={r['n_orders']} nav={r['nav_after']:,.0f} "
                      f"sharpe={r['sharpe'] or 0:.2f} recall={r['recalled']}")
        else:
            for a in agents:
                mark_to_market(a, d, px)

    print("\n=== verification ===")
    for a in agents:
        nav = db.query("SELECT COUNT(*) c, MAX(nav) mx FROM nav_history WHERE agent_id=%s", (a,))[0]
        orders = db.query("SELECT COUNT(*) c FROM orders WHERE agent_id=%s", (a,))[0]["c"]
        fills = db.query("SELECT COUNT(*) c FROM executions WHERE agent_id=%s", (a,))[0]["c"]
        pos = db.query("SELECT COUNT(*) c FROM positions WHERE agent_id=%s", (a,))[0]["c"]
        mem = db.query("SELECT COUNT(*) c FROM agent_memory WHERE agent_id=%s", (a,))[0]["c"]
        aud = db.query("SELECT COUNT(*) c FROM trade_audit WHERE agent_id=%s", (a,))[0]["c"]
        print(f"  {a:12s} nav_rows={nav['c']} last_nav={nav['mx']:,.0f} orders={orders} fills={fills} pos={pos} memory={mem} audit={aud}")

    # memory recall test
    print("\n=== memory recall (equal-weight) ===")
    for m in db.recall_memory("equal-weight", "how did turnover and nav change during rebalance", k=3):
        print(f"  score={m['score']:.3f} [{m['kind']}] {m['content'][:90]}")
    print("\nSMOKE TEST PASSED")

if __name__ == "__main__":
    main()
