# Strategy Sweep — testing thousands of configurations honestly

We backtested the **entire parameter grid** — every configuration across all 8
strategy families — on real S&P 500 prices, with the discipline a real-money
decision demands: an **in-sample / out-of-sample split** and a **multiple-testing
correction**. With thousands of trials, the highest raw Sharpe is almost always a
lucky fluke; this pipeline is built to find the edge that *survives* that fact.

## How it works

| File | Role |
|---|---|
| `research_fleet/research_agent/param_space.py` | Per-family parameter grids + deterministic random sampler; full grid = **2448 unique configs** |
| `research_fleet/research_agent/sweep.py` | Driver: caches price panels (so N configs → a handful of DB loads), runs each config on IS **and** OOS windows, writes results. Parity-verified identical to `backtest.run_backtest` within 1e-9. |
| `research_fleet/research_agent/sweep_schema.sql` + `apply_sweep_schema.py` | Tables `sweep_results` (columnstore), `sweep_runs`, `sweep_analysis` |
| `research_fleet/research_agent/sweep_analyze.py` | OOS-ranked, overfitting-aware analysis (deflated-Sharpe hurdle, robustness, family view) |

```bash
# preview the plan (no run):
python -m research_fleet.research_agent.sweep plan --target 2500 --seed 7
# run the full grid (each config backtested on IS + OOS):
python -m research_fleet.research_agent.sweep run  --target 2500 --seed 7
# analyze the latest sweep:
python -m research_fleet.research_agent.sweep_analyze --latest --top 25
```

Windows are CLI-overridable; default **IS = 2010-01-01…2019-12-31**,
**OOS = 2020-01-01…2024-12-31** (both inside the 2005–2024 price history).

## Results (sweep `swp-00a9e15a841f`)

**2448 configs, 0 errors.** Ranked by **out-of-sample** Sharpe (ranking by
in-sample is the wrong thing to do — every config was implicitly selected to
maximize IS).

**Honest winner — 3-month momentum:**

| Metric | Value |
|---|---|
| Family | `momentum` |
| Params | `lookback_days=63, top_n=30, rebalance_days=5, skip_days=21, turnover_cost_bps=5, universe_n=100` |
| OOS Sharpe | **1.533** |
| IS Sharpe | 1.471 |
| IS→OOS gap | **−0.062** (did *better* out-of-sample — the signature of a real edge, not overfitting) |
| OOS ann. return | 26.1% |
| OOS max drawdown | −17.0% |
| OOS turnover | 0.068 |

**The money-shot (why you can't ship the top backtest):** the config with the
**best in-sample** Sharpe (1.591) collapses to **rank #30 out-of-sample**. The
in-sample leader is not the out-of-sample leader.

**Reality check on the old "winner":** the honest OOS Sharpe (~1.5) is far below
the pre-hardening headline of **3.66** for a daily `regime` strategy — that number
was a cost-undercharging + in-sample-selection artifact, and it's gone once real
costs and walk-forward discipline are applied.

## Multiple-testing correction

With N = 2448 trials, the luckiest zero-edge strategy would post a sizeable Sharpe
by chance. Using a deflated-Sharpe / extreme-value hurdle (E[max] of N nulls +
95% Gumbel-tail cushion), a config must clear an **OOS Sharpe of ~0.145** to be
distinguishable from the best of 2448 coin-flips.

- **2436 / 2448** clear the noise hurdle; **12** are likely pure noise.
- **1022 / 2448** are *robust* (OOS positive, beats OOS benchmark, IS→OOS
  degradation bounded — the edge survived the walk-forward).
- Assumption: trials treated as independent (optimistic — real sweeps share
  signals, so the true hurdle is a touch lower). This makes the gate slightly
  **conservative**, the safe direction for real money.

## Family-level view (a stable edge across many configs > one lucky config)

| Family | n | median OOS | best OOS | median IS→OOS gap | robust |
|---|---|---|---|---|---|
| risk_parity | 162 | 0.955 | 1.215 | 0.308 | 141 |
| vol_target | 144 | 0.934 | 1.127 | 0.110 | 0 |
| equal_weight | 18 | 0.913 | 0.961 | 0.306 | 0 |
| factor | 486 | 0.909 | 1.288 | 0.413 | 222 |
| low_vol | 486 | 0.909 | 1.288 | 0.413 | 222 |
| mean_reversion | 432 | 0.883 | 1.356 | 0.108 | 76 |
| momentum | 576 | 0.878 | **1.533** | 0.268 | 264 |
| regime | 144 | 0.811 | 1.287 | 0.183 | 97 |

Worst-overfitting family: `factor` (median IS→OOS gap 0.413).

## Winner → risk gate (paper mode)

The winning momentum config was translated into target weights as of the latest
trading date (same `_weights_for` logic as the backtest) and pushed through the
[risk gate](RISK_CONTROLS.md) in **paper mode** via
`paper_trader.promote_candidate(...)`: **approved**, $100M deployed across 30
names (~$25.5K modeled cost) into the shadow book — the real `orders` / `nav_history`
stayed untouched. This is the intended path: sweep → analyze OOS → gate →
paper-trade → live. See
[`PRODUCTION_READINESS.md`](PRODUCTION_READINESS.md) for the live-capital roadmap.
