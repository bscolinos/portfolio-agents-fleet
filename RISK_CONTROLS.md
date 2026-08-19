# RISK CONTROLS — the pre-trade safety layer

This is the operator runbook for the safety layer that sits between a quant
research **finding** and any real-money **order**. Nothing trades until it clears
the gate. The gate is the *same* for paper and live — only the execution sink
differs.

**Posture: DEFAULT-DENY / FAIL-CLOSED.** If a required input is missing or
unreadable (no prices, no NAV, unreadable book/history, unreadable kill switch),
the gate **REJECTS**. It never silently approves.

Components (all under `pa_agents/`):

| File | Role |
|------|------|
| `pa_agents/risk_gate.py`   | Hard pre-trade gate — `evaluate(...) -> Decision` |
| `pa_agents/kill_switch.py` | Persisted global + per-agent kill switch (+ CLI) |
| `pa_agents/paper_trader.py`| `gated_rebalance(...)` + shadow (paper) book |
| `risk_schema.sql`          | `risk_decisions`, `kill_switches`, `paper_*` tables |
| `apply_risk_schema.py`     | Idempotent applier (already run against the live DB) |

Every evaluation is written to **two** places: an immutable `trade_audit` row
(`event_type='RISK_CHECK'`) and a queryable `risk_decisions` row.

---

## 1. Enforced limits (defaults + rationale)

All limits live in the `RiskLimits` dataclass in `pa_agents/risk_gate.py` and are
overridable per call. Institutional defaults:

| Check | Code | Default | Rationale |
|-------|------|---------|-----------|
| **Kill switch (FIRST)** | `KILL_SWITCH` | — | Global or per-agent circuit breaker; hard-reject before any other check. |
| Max single-name weight | `MAX_POSITION_WEIGHT` | `max_position_weight = 0.10` (10%) | No single name dominates. `clip_over_weight=False` → reject; set `True` → clip to cap + renormalize. |
| Sector concentration | `SECTOR_CONCENTRATION` | `max_sector_weight = 0.30` (30%) | Cap sector bets. **Only applied if `securities.sector` is populated** — today it is all-NULL, so this check **skips gracefully** (reported `applied:false`). |
| Gross exposure | `GROSS_EXPOSURE` | `max_gross_exposure = 1.00` | No leverage — Σ\|w\| ≤ 100%. |
| Net exposure | `NET_EXPOSURE` | `max_net_exposure = 1.00` | Σw ≤ 100%. |
| Shorts | `SHORTS_DISALLOWED` | `allow_shorts = False` | Long-only mandate; any negative weight rejects. |
| Min gross | `MIN_GROSS` | `min_gross_exposure = 0.0` | Allows all-cash defensive proposals. |
| Max turnover | `MAX_TURNOVER` | `max_turnover = 0.40` (40% of NAV) | Reject churny rebalances (cost + market-impact control). |
| Max # names | `MAX_NAMES` | `max_names = 100` | Cap over-diversification / operational sprawl. |
| Min position size | `MIN_POSITION_WEIGHT` | `min_position_weight = 0.005` (0.5%) | No dust positions. Set `0` to disable. |
| Drawdown breaker | `DRAWDOWN_BREAKER` | `max_drawdown = 0.15` (15%) | If trailing peak→current DD ≥ 15%, **block risk-increasing trades** (gross going up); de-risking is still allowed. |
| Daily-loss limit | `DAILY_LOSS_LIMIT` | `daily_loss_limit = 0.05` (5%) | If the last daily P&L ≤ −5% of prior NAV, reject. |
| Price sanity | `PRICE_SANITY` | `require_prices = True` | Any target name with missing/zero/non-positive price rejects. |
| Per-name notional | `MAX_NAME_NOTIONAL` | `max_name_notional = 25_000_000` ($25M) | Absolute per-order notional cap. **ADV/liquidity proxy** — see limitations. |
| NAV present | `NAV_MISSING` | — | Non-positive/missing NAV → fail-closed. |
| Book readable | `BOOK_UNREADABLE` | — | Can't read positions/nav → fail-closed. |
| History readable | `HISTORY_UNREADABLE` | — | `nav_history` read *error* (not merely empty) → fail-closed. |
| Kill switch readable | `KILL_SWITCH_UNREADABLE` | — | Can't read the switch → fail-closed. |

**Empty-history behaviour (documented):** for a brand-new agent with no
`nav_history` rows, the drawdown and daily-loss circuit-breakers **no-op**
(there is no peak or prior P&L to breach) — they are *not* treated as a failure.
A genuine DB read *error* is distinct and fails closed (`HISTORY_UNREADABLE`).

The `Decision` object returned by `evaluate(...)`:

```python
Decision(approved: bool,
         reason: str,
         violations: list[dict],          # [{code, detail, ...}]
         adjusted_weights: dict | None,   # set only when approved AND clipped
         checks: dict)                    # full structured check set
```

---

## 2. Kill switch — the big red button

Persisted in SingleStore (`kill_switches`), so an engaged switch **survives
restarts** and is visible fleet-wide the instant the next gate call runs. Scope
`global` trips the entire fleet; scope `<agent_id>` trips one agent.
`is_engaged(agent)` is true if **either** is engaged. Every engage/release writes
a `trade_audit` row (`event_type='KILL_SWITCH'`).

### Exact CLI commands

```bash
source /Users/billscolinos/Documents/code_factory/.venv/bin/activate

# STOP the whole fleet (emergency):
python -m pa_agents.kill_switch engage --scope global --reason "vol spike / bad data" --by ops

# STOP just one agent:
python -m pa_agents.kill_switch engage --scope max-return --reason "runaway turnover" --by ops

# See state (prints a loud banner; red if anything engaged):
python -m pa_agents.kill_switch status

# RESUME the fleet:
python -m pa_agents.kill_switch release --scope global --by ops

# RESUME one agent:
python -m pa_agents.kill_switch release --scope max-return --by ops
```

`--reason` is **required** on `engage` (an engaged switch must say why).
Programmatic API: `kill_switch.engage(scope, reason=, by=)`,
`kill_switch.release(scope, by=)`, `kill_switch.is_engaged(agent_id)`,
`kill_switch.status()`.

---

## 3. Paper mode — "paper-trade first"

`pa_agents.paper_trader.gated_rebalance(...)` is the single entry point. It:

1. Calls `risk_gate.evaluate(...)` first. If not approved, it records the
   rejection (already persisted by the gate) and returns **without trading**.
2. If approved and `mode='paper'` (**default**): routes fills to the SHADOW book
   — `paper_orders` / `paper_positions` / `paper_nav_history`. It **never**
   touches the real `orders`/`positions`/`nav_history`. The paper fill math
   mirrors `trading.py` exactly (same `CostModel`: `commission_per_share=0.005`,
   `min_commission=1.0`, `slippage_bps=2.0`; same MOC-vs-adjusted-close,
   cost-basis, realized-P&L and NAV-marking).
3. If `mode='live'`: requires **`allow_live=True`** AND a disengaged kill switch
   (re-checked at execution time) AND gate approval, then calls the existing
   `trading.rebalance(...)` — the internal **simulated** book. There is **no real
   broker**: a broker adapter is the only remaining step and is intentionally not
   implemented (see below).

```python
from pa_agents import paper_trader
# operator watches the paper equity curve build before authorizing live:
res = paper_trader.promote_candidate(
    agent_id="max-sharpe", as_of_date="2024-12-31",
    target_weights=best_weights, prices=prices)   # mode='paper', can NEVER hit the real book
```

Inspect the paper equity curve:

```sql
SELECT as_of_date, nav, cum_return, turnover, tcost
FROM paper_nav_history WHERE agent_id='max-sharpe' ORDER BY as_of_date;
```

---

## 4. Activating the gate — ALREADY WIRED into `runner.py` (opt-in)

The gate is **wired into `pa_agents/runner.py`** between SOLVE and TRADE, behind
an opt-in flag so a plain historical backtest replay is unaffected (the
drawdown / turnover / daily-loss breakers would otherwise block legitimate
*simulated* rebalances and corrupt the equity curve).

- `run_rebalance(..., enforce_risk=False, risk_limits=None)` — default **off**.
  When `enforce_risk=True`, the gate runs; a **rejected** rebalance places no
  orders, marks the run `status='blocked'`, writes a `RUN_END` audit row with the
  violations, and returns `{"blocked": True, "reason": ..., "violations": [...]}`.
  Gate-clipped weights (over-weight names) are what get traded when approved.
- `pa_agents/fleet.py` exposes it as **`--enforce-risk`**:

```bash
# gated replay (blocks + audits rebalances that violate limits):
python -m pa_agents.fleet backtest --start 2023-01-01 --end 2024-12-31 \
    --universe 60 --rebalance-freq 21 --lookback 252 --enforce-risk
```

**Inception handling:** on an agent's first trade there is no `nav_history`, so
the runner seeds NAV from `starting_nav` and relaxes *only* the turnover cap to
the gross cap for that one rebalance — funding a book from cash is 100% one-way
turnover by construction, not churn, and its size is still bounded by the gross
limit. Every other check (weights, shorts, leverage, kill switch, price sanity)
applies unchanged at inception.

For a paper-first forward flow, call
`paper_trader.gated_rebalance(..., mode='paper')` directly — it runs the same
gate and routes fills to the shadow book, never the real one.

---

## 5. Connecting live capital (roadmap)

The gate is the safety *decision* layer and is identical for paper and live. To
route a live, real-capital order through it, wire the following:

1. **Broker adapter.** `mode='live'` calls `trading.rebalance(...)`, the
   internal *simulated* book. Sending real orders to an execution venue
   (FIX/REST, order-state reconciliation, partial fills, cancels/rejects, DMA
   throttles) is **not implemented**. This is the single largest remaining piece.
2. **No ADV / liquidity data.** The notional guard is an *absolute* per-order $
   cap (`max_name_notional`), not a % of Average Daily Volume. Without ADV we
   cannot enforce true participation-rate / market-impact limits. Load ADV and
   convert `MAX_NAME_NOTIONAL` into a `%ADV` check before trading illiquid names.
3. **No point-in-time / survivorship-clean data.** Prices come from the demo
   `prices` table (backtest fixture). Live trading needs a real-time,
   corporate-action-adjusted, survivorship-bias-free feed, plus a staleness
   guard tighter than "price present and > 0".
4. **Sector limits inactive.** `securities.sector` is all-NULL today, so
   `SECTOR_CONCENTRATION` skips. Populate sectors (and ideally factor/beta
   exposures) to activate it.
5. **Drawdown/daily-loss depend on `nav_history` fidelity.** For a fresh agent
   with no history these breakers no-op by design. They are only as good as the
   daily marks feeding `nav_history`.
6. **Pre-trade compliance/restricted-list, wash-trade, and borrow checks.**
   Add these to the gate before routing live capital.
7. **Single-writer assumption.** The gate reads the book, decides, then trades;
   there is no cross-process lock. Concurrent rebalances of the *same* agent
   could race. Serialize per-agent execution (or add an advisory lock) for live.

Paper mode is the default execution sink; connect items 1–3 to route live capital
through the same gate.
