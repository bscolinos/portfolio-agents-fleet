# Production Readiness — safety layer & live-capital roadmap

**Context.** The best research agent is built to trade **real capital**. This
document is the summary: what was hardened, what the verified numbers say, and
what to wire to route live orders. It reports honestly — trust the computed
metrics and the gate, not any headline Sharpe.

**Scope of this pass: safety-first.** Everything that must precede a live order —
correct winner selection, a hard risk/execution gate, a kill switch, and a
paper-trading harness — is done and verified. Paper mode is the default execution
sink; a **real brokerage adapter** is the remaining integration to route the same
gated decisions to a live venue (see §4).

---

## 1. Backtest correctness — the winner is now selected honestly

**Bug fixed (critical): turnover cost was undercharged.** The engine read only
`params["tc_bps"]` (default 2bps) and ignored the `turnover_cost_bps` the research
agents actually emit. A daily-rebalancing strategy that declared 10bps was charged
2 — inflating its Sharpe. Cost resolution now prefers `turnover_cost_bps`, falls
back to `tc_bps`, else a 5bps institutional default, and adds a 2bps modeled
`slippage_bps` leg (matching `CostModel.slippage_bps` in the live trade module),
charged on actual per-rebalance turnover Σ|Δw|.

**Bug fixed: look-ahead + survivorship in universe construction.** Eligibility is
now decided from trailing history available *as of* each rebalance date
(`eligible_as_of`), not from full-future-window completeness; the leaky
`.ffill().dropna()` was removed (bounded gap-fill only, no fill across
delistings). *Residual limitation (documented in `data_caveats`):* there is no
point-in-time index-membership table, so true historical constituents are not
reconstructable — this removes the worst bias, not all of it.

**Re-score of all 146 experiments** (`experiment_rescores` table, originals
untouched):

| | Strategy | Sharpe (old → new) | Note |
|---|---|---|---|
| OLD #1 | regime, daily, 10bps | 3.663 | undercharged at 2bps |
| NEW #1 | **same regime experiment** | **3.663 → 3.544** (Δ −0.118) | still #1 of 146 |

The regime winner **survives** because a trend filter is binary (fully invested
vs. fully cash) — despite daily rebalancing its *real* turnover is tiny (~0.099),
so the cost fix barely touches it. But the correction is not cosmetic elsewhere:
**35 experiments flipped `beats_benchmark`** once correctly costed (27 lost it, 8
gained it). High-churn families (e.g. `mean_reversion`, avg turnover ~0.36)
re-ranked materially. **Takeaway: a large fraction of "benchmark-beating"
experiments were cost-inflated mirages; the top-of-book was robust.**

Re-run anytime: `python -m research_fleet.research_agent.rescore`.

---

## 2. Pre-trade risk/execution gate — nothing trades until it clears

`pa_agents/risk_gate.py` — `evaluate(...) -> Decision`. **Default-deny /
fail-closed**: missing prices, missing/non-positive NAV, unreadable book, history,
or kill switch all **reject**. Enforced limits (institutional defaults, all
overridable per call):

kill switch (checked first) · max single-name weight 10% · gross ≤100% (no
leverage) · net ≤100% · shorts disallowed · max turnover 40% · max 100 names ·
min position 0.5% · drawdown breaker 15% (blocks risk-increasing trades) ·
daily-loss limit 5% · price sanity · per-name notional cap $25M · sector cap 30%
(auto-skips until `securities.sector` is populated).

Every decision is written to **`trade_audit`** (`event_type='RISK_CHECK'`) and
**`risk_decisions`**. Full detail + rationale in
[`RISK_CONTROLS.md`](RISK_CONTROLS.md).

**Wired into `runner.py`** (opt-in `enforce_risk` / `--enforce-risk`, off by
default so plain backtests are unaffected). Inception funding relaxes only the
turnover cap for the first trade (cash→invested is 100% one-way by construction,
still bounded by the gross cap); all other checks apply.

---

## 3. Kill switch + paper trading

- **Kill switch** (`pa_agents/kill_switch.py`) — persisted in SingleStore
  (`kill_switches`), survives restarts, fleet-wide instantly. Global or per-agent.
  Checked first on every gate call. CLI:
  `python -m pa_agents.kill_switch engage --scope global --reason "…" --by ops`
  (`status` / `release` likewise). Every toggle writes a `KILL_SWITCH` audit row.
- **Paper trading** (`pa_agents/paper_trader.py`) — `gated_rebalance(..., mode='paper')`
  runs the gate then routes fills to a SHADOW book (`paper_orders` /
  `paper_positions` / `paper_nav_history`), using the *same* cost model as the
  live path, and **never** touches the real book. `mode='live'` requires an
  explicit `allow_live=True` + disengaged kill switch + gate approval — and even
  then hits only the internal simulated book (no real broker; see §4).

---

## 4. Connecting live capital (roadmap)

Paper mode is the default execution sink and the gate/kill-switch/audit path is
identical for paper and live — only the sink changes. To route the same gated
decisions to a live venue, wire the following:

1. **Broker adapter** — `mode='live'` currently uses the internal simulated book.
   Real order routing (FIX/REST, order-state reconciliation, partial fills,
   cancels/rejects, DMA throttles) is the single largest remaining piece.
2. **No ADV/liquidity data** — the notional guard is an absolute $ cap, not a
   %ADV participation limit.
3. **No point-in-time / survivorship-clean feed** — prices are a backtest fixture;
   live needs a corporate-action-adjusted, real-time feed + tighter staleness guard.
4. **Sector/factor limits inactive** until `securities.sector` (and ideally
   factor/beta exposures) are populated.
5. **No restricted-list / wash-trade / borrow checks**, and **no per-agent
   execution lock** (serialize live rebalances to avoid a read-decide-trade race).
6. **LLM finding prose can cite fabricated priors** — the numbers are trustworthy
   (computed by the corrected engine); the narrative is not evidence.

**Recommended path:** paper-trade the re-scored winner through the gate, watch
the `paper_nav_history` curve, then connect items 1–3 to route live capital
through the same gate. The infra Aura proxy is production-grade
([`research_fleet/aura/`](research_fleet/aura/)); adding HA/alerting for it is a
straightforward follow-on.

---

## Verification (reproduce)

```bash
source /Users/billscolinos/Documents/code_factory/.venv/bin/activate
cd demos/portfolio-agents
python -m pytest pa_agents/tests -q                                   # 16 passed — gate + kill switch
python -m pytest research_fleet/research_agent/tests/test_backtest.py -q  # 7 passed — cost + bias fixes
python -m research_fleet.research_agent.rescore                       # re-rank all experiments
python -m pa_agents.kill_switch status                                # kill-switch state
```
