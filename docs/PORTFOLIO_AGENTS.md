# Portfolio Agents — NVIDIA × SingleStore

A fleet of autonomous **strategy agents** that compete on the S&P 500 using
**NVIDIA GPU-accelerated portfolio optimization** (cuOpt + cuML on an L4), with
**truly persisted agent memory** and **Goldman-level trade tracking** — all in
**SingleStore**.

Built on the [NVIDIA portfolio-optimization blueprint](https://github.com/NVIDIA-AI-Blueprints/portfolio-optimization)
(Mean-CVaR / Mean-Variance solved on GPU, cuML KDE scenario generation).

## The two theses

1. **Truly persisted agent memory.** Every agent writes episodic observations,
   decisions and distilled learnings — each embedded as a Qwen `VECTOR(1024)` —
   into SingleStore. Before each rebalance it **recalls its most relevant past
   experience by semantic similarity** (`embedding <*> query` DOT_PRODUCT) and
   re-reads what it learned last time. Kill the process and restart a week
   later: the agent resumes with its full memory and book intact, because memory
   lives in the database, not RAM.

2. **Goldman-level trade tracking.** The optimizer emits *weight fractions*; a
   trading desk lives in *shares, fills, commission, slippage, positions, P&L*
   and an *audit trail*. The trade engine translates target weights into the
   full front-to-back lifecycle — parent orders → executions (with commission +
   modeled slippage + venue) → share-level positions with cost basis → daily
   mark-to-market NAV / realized + unrealized P&L → per-run risk analytics → an
   immutable compliance audit log — every step persisted and queryable.

Both run on **one engine**: transactional book-keeping, analytical time series,
JSON, and native vector search side by side in SingleStore.

## Architecture

```
NVIDIA L4 GPU box (EC2 g6.2xlarge)                 SingleStore (S-2, us-east-1)
┌──────────────────────────────┐                  ┌───────────────────────────┐
│ pa_agents fleet              │   reads prices   │ prices (1.9M rows, 384     │
│  ├ strategies.py  ┐          │◀────────────────▶│   tickers, 2005–2024)      │
│  ├ nvidia_adapter │ cuOpt /  │                  │ agents / securities        │
│  │   (GPU solve)  │ cuML KDE │   writes trades  │ agent_runs                 │
│  ├ trading.py  (weights→     │─────────────────▶│ orders / executions        │
│  │   orders→fills→NAV)       │                  │ positions / snapshots      │
│  ├ runner.py  (recall→solve→ │   writes memory  │ nav_history / risk_metrics │
│  │   trade→reflect)          │─────────────────▶│ agent_memory  VECTOR(1024) │
│  └ fleet.py   (backtest)     │   recall <*>     │ strategy_params            │
└──────────────────────────────┘◀────────────────│ trade_audit                │
                                                   └───────────────────────────┘
      FastAPI backend (:8210)  ──/api/*──▶  Next.js dashboard (:3011)
      (agents, leaderboard, nav, blotter, positions, memory + LIVE recall, runs, audit)
```

## The agents

| Agent | Strategy | Engine | Mandate |
|---|---|---|---|
| **max-sharpe** | Mean-Variance | GPU cuOpt | Maximize risk-adjusted return |
| **min-cvar** | Mean-CVaR (95%) | GPU cuOpt + cuML KDE | Minimize tail risk over scenarios |
| **risk-parity** | Inverse-vol | CPU | Balance risk contribution |
| **max-return** | Variance-capped MV | GPU cuOpt | Return-seeking under a risk cap |
| **equal-weight** | 1/N | CPU | Benchmark every strategy must beat |

Each run records the real `engine` and `gpu_name` it used, plus `solve_ms` /
`scenario_ms` — so the dashboard reports honestly what ran on the L4.

## Layout

- `pa_agents/` — the runtime (runs on the GPU box):
  `config.py`, `db.py` (SingleStore + Qwen embeddings + `<*>` recall),
  `strategies.py` (roster + optimizer adapter), `nvidia_adapter.py` (the one
  file that calls cuOpt/cuML), `trading.py` (trade lifecycle),
  `runner.py` (recall→solve→trade→reflect), `fleet.py` (backtest CLI), `llm.py`.
- `backend/` — FastAPI, all `/api/*` endpoints (see `API_CONTRACT.md`).
- `frontend/` — Next.js trading-terminal dashboard.
- `schema.sql` / `apply_schema.py` — the 13-table schema.
- `smoke_test.py` — local non-GPU end-to-end validation.
- `../../staging/portfolio-agents-prep/deploy_fleet.sh` — ship + run the fleet
  on the GPU box.

## Run it

```bash
# 1. schema (once) — workspace + DB already provisioned
source /Users/billscolinos/Documents/code_factory/.venv/bin/activate
python apply_schema.py

# 2. fleet on the GPU box (loads prices, runs the walk-forward backtest on the L4)
bash ../../staging/portfolio-agents-prep/deploy_fleet.sh

# 3. UI locally
bash run_demo.sh          # backend :8210 + frontend :3011
open http://localhost:3011
```

## Infra (live)

- **SingleStore**: workspace `portfolio-agents` (S-2), DB `portfolio_agents`,
  us-east-1. Connection + model keys in `.env`.
- **GPU box**: EC2 `g6.2xlarge` (single NVIDIA **L4**), us-east-1c. Repo at
  `/home/ubuntu/portfolio-optimization`; fleet shipped to `pa_agents/` there.
  SSH key at `../../staging/portfolio-agents-gpu.pem` (SG scoped to operator /32).

> Cost note: the GPU box bills while running. Stop or terminate it when done
> (`aws ec2 stop-instances` / `terminate-instances`). See teardown below.

## Teardown

```bash
# stop GPU billing (keeps the instance; terminate to delete)
AWS_SHARED_CREDENTIALS_FILE=~/.aws/credentials_cf AWS_PROFILE=cf_gpu \
  aws ec2 terminate-instances --instance-ids i-0e1611e05d37dae11 --region us-east-1
# tear down the SingleStore workspace + folder
cd /Users/billscolinos/Documents/code_factory && python -m factory teardown portfolio-agents
```
