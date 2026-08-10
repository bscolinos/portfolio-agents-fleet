# Portfolio Agents — API contract

FastAPI backend at `backend/main.py`, base path `/api`. All responses JSON.
Reads the SingleStore tables defined in `schema.sql` (DB `portfolio_agents`).
Uses `backend/singlestore.py` (`singlestoredb` driver) for DB and the demo's
`.env` for config. CORS open to the frontend dev origin.

Tables (see `schema.sql`): agents, securities, prices, agent_runs, agent_memory,
strategy_params, orders, executions, positions, position_snapshots, nav_history,
risk_metrics, trade_audit.

## Endpoints

- `GET /api/health` → `{ok, db_version, db}`.
- `GET /api/stats` → global money-moment tiles:
  `{n_agents, n_trades, total_notional, gpu_solves, cpu_solves, avg_solve_ms,
    total_memories, total_audit_events, universe_size, as_of}`.
- `GET /api/agents` → roster, each:
  `{agent_id, display_name, strategy_type, objective, engine, color,
    latest_nav, cum_return, daily_return, sharpe, n_positions, last_run_at,
    last_engine, last_gpu_name, avg_solve_ms}`.
- `GET /api/leaderboard` → agents ranked by `cum_return` desc (same fields as
  /agents plus `rank`, `turnover`, `max_drawdown`, `vol`).
- `GET /api/nav?agent=<id|all>&start=&end=` → equity curves:
  `{series: [{agent_id, display_name, color, points: [{date, nav, cum_return, daily_return}]}]}`.
- `GET /api/positions?agent=<id>` →
  `{agent_id, as_of, cash, nav, positions: [{ticker, qty, avg_cost, last_price,
    market_value, weight, unrealized_pnl}]}` (weight desc).
- `GET /api/blotter?agent=<id|all>&limit=100` → trade blotter, recent fills:
  `[{executed_at, agent_id, ticker, side, fill_qty, fill_price, notional,
     commission, slippage_bps, venue, run_id}]` (executed_at desc).
- `GET /api/runs?agent=<id|all>&limit=50` → recent optimization runs:
  `[{run_id, agent_id, as_of_date, engine, gpu_name, num_scenarios, solve_ms,
     scenario_ms, universe_size, status, started_at, finished_at}]`.
- `GET /api/memory?agent=<id>&kind=&limit=50` → persisted memory feed:
  `[{memory_id, kind, as_of_date, content, importance, metrics, tags, created_at}]`.
- `GET /api/memory/recall?agent=<id>&q=<text>&k=5` → LIVE semantic recall
  (embeds `q` via Qwen, ranks by `embedding <*> qvec`):
  `{query, agent_id, results: [{content, kind, score, created_at, importance}]}`.
- `GET /api/risk?agent=<id>&limit=100` → risk metric series:
  `[{as_of_date, exp_return, volatility, sharpe, cvar, turnover, n_positions}]`.
- `GET /api/audit?run_id=&agent=&limit=100` → compliance trail:
  `[{ts, agent_id, run_id, event_type, entity_ref, ticker, detail, actor}]` (ts desc).

## Notes
- All numeric returns are fractions (0.1234 == 12.34%); frontend formats.
- `metrics`, `tags`, `detail` columns are JSON — parse to objects in responses.
- Empty tables must return empty arrays / zeroed tiles (never 500) so the UI
  renders before the fleet has run.
- Vector recall must call the Qwen endpoint exactly like `pa_agents/db.py`
  (`embed` + `vec_literal` + `<*>`), or import and reuse those helpers.
