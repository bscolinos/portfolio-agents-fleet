"""Seed the research work queue with concrete strategy-research briefs.

Each task is a self-contained research brief a single OpenClaw agent can pull,
investigate against the sp500 prices already in SingleStore, and report on. The
fleet claims tasks atomically so multiple agents don't collide.
"""
import os, uuid, singlestoredb as s2
from pathlib import Path

env = Path(__file__).resolve().parents[2] / "demos" / "portfolio-agents" / ".env"
for line in env.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("="); os.environ.setdefault(k.strip(), v.strip())

TASKS = [
    ("Momentum: cross-sectional 12-1", "momentum",
     "Investigate cross-sectional momentum on the S&P 500 (prices table in SingleStore, "
     "portfolio_agents DB). Rank names by trailing 12-month return skipping the most recent "
     "month (12-1), long the top decile, monthly rebalance. Backtest 2018-2024, compare Sharpe "
     "and max drawdown vs an equal-weight benchmark. Log a hypothesis, run the experiment, and "
     "write a finding on whether 12-1 momentum adds value and in which regimes.", 1),
    ("Mean-reversion: short-term reversal", "mean_reversion",
     "Test a short-term reversal strategy on the S&P 500: each week, long the prior-week losers "
     "and evaluate forward 1-week returns. Does 1-week reversal exist net of turnover? Backtest "
     "2019-2024, report Sharpe, turnover, and sensitivity to the number of names. Write a finding.", 2),
    ("Volatility targeting overlay", "vol_target",
     "Study a volatility-targeting overlay: scale exposure to hit a constant 10% annualized "
     "portfolio volatility using a 20-day realized-vol estimate on an equal-weight S&P 500 book. "
     "Backtest 2018-2024. Does vol-targeting improve risk-adjusted return and reduce drawdown vs "
     "static full investment? Write a finding with the realized vol and Sharpe deltas.", 2),
    ("Low-volatility factor", "factor",
     "Examine the low-volatility anomaly: form a portfolio of the 50 lowest-trailing-60-day-vol "
     "S&P 500 names, monthly rebalance, and compare to equal-weight over 2018-2024. Report Sharpe, "
     "vol, drawdown, and whether low-vol delivered comparable returns at lower risk. Write a finding.", 3),
    ("Risk-parity vs equal-weight", "risk_parity",
     "Compare naive risk parity (inverse-vol weights, capped at 10% per name) to equal-weight on a "
     "60-name S&P 500 universe, monthly rebalance, 2018-2024. Which has the better Sharpe and "
     "smaller drawdown, and at what turnover cost? Write a finding.", 3),
    ("Regime detection: trend filter", "regime",
     "Investigate a simple regime filter: hold an equal-weight S&P 500 book only when the index "
     "(proxy: equal-weight average) is above its 200-day moving average, else hold cash. Backtest "
     "2015-2024. Does the trend filter improve Sharpe and cut drawdown vs always-invested? Write a finding.", 2),
    ("Momentum + vol-target combo", "momentum",
     "Combine 12-1 cross-sectional momentum (top decile) with a 10% vol-target overlay on the S&P 500. "
     "Backtest 2018-2024. Does the combination beat either component alone on Sharpe and drawdown? "
     "Form the hypothesis, run experiments for each component and the combo, and write a comparative finding.", 4),
    ("Concentration limits sensitivity", "factor",
     "Study how per-name concentration limits (max weight 5% vs 10% vs 20%) affect an equal-weight-ish "
     "S&P 500 book's risk/return over 2018-2024. Is there a sweet spot that improves Sharpe without "
     "over-concentrating? Write a finding with the tradeoff curve.", 4),
]

conn = s2.connect(host=os.environ["SINGLESTORE_HOST"], port=int(os.environ["SINGLESTORE_PORT"]),
                  user=os.environ["SINGLESTORE_USER"], password=os.environ["SINGLESTORE_PASSWORD"],
                  database="portfolio_agents")
cur = conn.cursor()
cur.execute("SELECT COUNT(*) FROM research_tasks")
if cur.fetchone()[0] == 0:
    rows = [(f"task-{uuid.uuid4().hex[:10]}", t, fa, p, pr) for (t, fa, p, pr) in TASKS]
    cur.executemany(
        "INSERT INTO research_tasks (task_id,title,focus_area,prompt,priority,status,created_at) "
        "VALUES (%s,%s,%s,%s,%s,'pending',NOW(6))", rows)
    print(f"seeded {len(rows)} research tasks")
else:
    print("research_tasks already seeded; skipping")
cur.execute("SELECT focus_area, COUNT(*) FROM research_tasks GROUP BY focus_area")
for r in cur.fetchall(): print("  ", r[0], r[1])
cur.close(); conn.close()
