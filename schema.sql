-- ============================================================================
-- Portfolio Agents — SingleStore schema
--
-- Two pillars, one engine:
--   1. TRULY PERSISTED AGENT MEMORY — every strategy agent writes episodic
--      observations, decisions, reflections and distilled learnings into a
--      unified memory store with Qwen VECTOR(1024) embeddings, then recalls the
--      most relevant past experience by semantic similarity at the start of the
--      next run. Memory survives process death: it lives in SingleStore, not RAM.
--   2. GOLDMAN-SACHS-LEVEL TRADE TRACKING — the full front-to-back trade
--      lifecycle: target weights -> parent orders -> executions (with
--      commission + slippage + venue) -> share-level positions -> daily NAV /
--      P&L -> per-run risk analytics -> an immutable compliance audit trail.
--
-- Engine choices follow SingleStore best practice:
--   * ROWSTORE for small, hot, point-lookup / frequently-updated tables
--     (agents, securities, positions, live memory).
--   * COLUMNSTORE for high-volume, append-mostly, analytical time series
--     (prices, orders, executions, nav, risk, snapshots, audit).
--   * VECTOR(1024) matches the SingleStore-hosted Qwen embedding dimension.
--
-- DB name is injected by apply.py (defaults to the demo's database).
-- ============================================================================

-- ---------------------------------------------------------------------------
-- REFERENCE DIMENSIONS (rowstore)
-- ---------------------------------------------------------------------------

-- The strategy agents themselves. Each is a distinct optimization "trader".
CREATE ROWSTORE TABLE IF NOT EXISTS agents (
    agent_id        VARCHAR(64)   NOT NULL,        -- e.g. 'max-sharpe', 'min-cvar'
    display_name    VARCHAR(128)  NOT NULL,
    strategy_type   VARCHAR(48)   NOT NULL,        -- mean_variance | cvar | risk_parity | equal_weight | max_return
    objective       VARCHAR(256)  NOT NULL,        -- human-readable mandate
    engine          VARCHAR(24)   NOT NULL DEFAULT 'gpu',  -- gpu (cuOpt) | cpu (cvxpy)
    default_params  JSON          NOT NULL,        -- baseline hyperparameters
    color           VARCHAR(16)   NOT NULL DEFAULT '#76b900',
    status          VARCHAR(24)   NOT NULL DEFAULT 'active',
    created_at      DATETIME(6)   NOT NULL,
    PRIMARY KEY (agent_id)
);

-- The tradable universe (S&P 500 constituents used by the NVIDIA blueprint).
CREATE ROWSTORE TABLE IF NOT EXISTS securities (
    ticker          VARCHAR(16)   NOT NULL,
    name            VARCHAR(128),
    sector          VARCHAR(64),
    is_active       TINYINT       NOT NULL DEFAULT 1,
    PRIMARY KEY (ticker)
);

-- ---------------------------------------------------------------------------
-- MARKET DATA (columnstore time series)
-- ---------------------------------------------------------------------------

-- Daily adjusted prices + simple return, backing every backtest.
CREATE TABLE IF NOT EXISTS prices (
    ticker          VARCHAR(16)   NOT NULL,
    trade_date      DATE          NOT NULL,
    adj_close       DOUBLE        NOT NULL,
    daily_return    DOUBLE,                        -- pct change vs prior trading day
    SHARD KEY (ticker),
    SORT KEY (trade_date, ticker)
);

-- ---------------------------------------------------------------------------
-- AGENT RUNS + PERSISTED MEMORY
-- ---------------------------------------------------------------------------

-- One row per optimization / backtest cycle an agent performs. The spine that
-- ties memory, orders, risk and P&L together.
CREATE ROWSTORE TABLE IF NOT EXISTS agent_runs (
    run_id          VARCHAR(64)   NOT NULL,        -- e.g. 'run-<agent>-<ts>-<seq>'
    agent_id        VARCHAR(64)   NOT NULL,
    run_type        VARCHAR(32)   NOT NULL DEFAULT 'rebalance',  -- rebalance | backtest | optimize
    universe_size   INT           NOT NULL,
    lookback_start  DATE,
    lookback_end    DATE,
    as_of_date      DATE          NOT NULL,        -- the rebalance / decision date
    params          JSON          NOT NULL,        -- exact hyperparameters used this run
    engine          VARCHAR(24)   NOT NULL,        -- gpu | cpu
    gpu_name        VARCHAR(64),                   -- e.g. 'NVIDIA L4'
    num_scenarios   INT,                           -- CVaR/KDE scenarios generated
    solve_ms        DOUBLE,                        -- optimizer wall time (ms)
    scenario_ms     DOUBLE,                        -- scenario-generation wall time (ms)
    status          VARCHAR(24)   NOT NULL DEFAULT 'running',    -- running | ok | failed
    error           TEXT,
    started_at      DATETIME(6)   NOT NULL,
    finished_at     DATETIME(6),
    PRIMARY KEY (run_id),
    KEY (agent_id) USING HASH,
    KEY (as_of_date) USING HASH
);

-- The persisted memory store. `kind` discriminates episodic vs semantic entries;
-- `embedding` powers semantic recall (ORDER BY embedding <*> :q DESC). An agent
-- reads its top-k most relevant memories before deciding, and writes new
-- observations/learnings after. This is the "memory survives the process" pillar.
CREATE ROWSTORE TABLE IF NOT EXISTS agent_memory (
    memory_id       VARCHAR(64)   NOT NULL,
    agent_id        VARCHAR(64)   NOT NULL,
    run_id          VARCHAR(64),                   -- run that produced this memory (nullable for seeded priors)
    kind            VARCHAR(24)   NOT NULL,        -- observation | decision | reflection | learning
    as_of_date      DATE,
    content         TEXT          NOT NULL,        -- natural-language memory the agent can re-read
    embedding       VECTOR(1024)  NOT NULL,        -- Qwen embedding of `content`
    importance      FLOAT         NOT NULL DEFAULT 0.5,  -- 0..1, decayed recall weight
    metrics         JSON,                          -- structured snapshot (sharpe, cvar, turnover, ...)
    tags            JSON,                          -- free-form labels for filtering
    created_at      DATETIME(6)   NOT NULL,
    PRIMARY KEY (memory_id),
    KEY (agent_id, kind) USING HASH
);

-- Versioned, evolving hyperparameter sets. An agent persists "the best config I
-- have found so far" so subsequent runs start from learned settings, not defaults.
CREATE ROWSTORE TABLE IF NOT EXISTS strategy_params (
    param_id        VARCHAR(64)   NOT NULL,
    agent_id        VARCHAR(64)   NOT NULL,
    version         INT           NOT NULL,
    params          JSON          NOT NULL,
    rationale       TEXT,                          -- why the agent chose these
    source_run_id   VARCHAR(64),                   -- run that motivated the change
    is_current      TINYINT       NOT NULL DEFAULT 1,
    created_at      DATETIME(6)   NOT NULL,
    PRIMARY KEY (param_id),
    KEY (agent_id, version) USING HASH
);

-- ---------------------------------------------------------------------------
-- TRADE LIFECYCLE (Goldman-level front-to-back)
-- ---------------------------------------------------------------------------

-- Parent orders: the intent produced by translating target weights (from the
-- optimizer) into share-level buys/sells given current NAV and price.
CREATE TABLE IF NOT EXISTS orders (
    order_id        VARCHAR(80)   NOT NULL,
    run_id          VARCHAR(64)   NOT NULL,
    agent_id        VARCHAR(64)   NOT NULL,
    as_of_date      DATE          NOT NULL,
    ticker          VARCHAR(16)   NOT NULL,
    side            VARCHAR(4)    NOT NULL,        -- BUY | SELL
    target_weight   DOUBLE        NOT NULL,
    prev_weight     DOUBLE        NOT NULL,
    order_qty       DOUBLE        NOT NULL,        -- signed shares to trade
    ref_price       DOUBLE        NOT NULL,        -- decision-time price
    order_notional  DOUBLE        NOT NULL,        -- |qty| * ref_price
    order_type      VARCHAR(16)   NOT NULL DEFAULT 'MOC',   -- market-on-close (backtest convention)
    tif             VARCHAR(8)     NOT NULL DEFAULT 'DAY',
    status          VARCHAR(16)   NOT NULL DEFAULT 'FILLED',
    created_at      DATETIME(6)   NOT NULL,
    SHARD KEY (run_id),
    SORT KEY (as_of_date, agent_id, ticker),
    KEY (agent_id) USING HASH
);

-- Executions / fills. Goldman-level: every fill carries commission, modeled
-- slippage (bps) and a venue, so transaction-cost analysis is first-class.
CREATE TABLE IF NOT EXISTS executions (
    exec_id         VARCHAR(80)   NOT NULL,
    order_id        VARCHAR(80)   NOT NULL,
    run_id          VARCHAR(64)   NOT NULL,
    agent_id        VARCHAR(64)   NOT NULL,
    as_of_date      DATE          NOT NULL,
    ticker          VARCHAR(16)   NOT NULL,
    side            VARCHAR(4)    NOT NULL,
    fill_qty        DOUBLE        NOT NULL,        -- signed shares filled
    fill_price      DOUBLE        NOT NULL,        -- ref_price adjusted for slippage
    notional        DOUBLE        NOT NULL,
    commission      DOUBLE        NOT NULL DEFAULT 0,
    slippage_bps    DOUBLE        NOT NULL DEFAULT 0,
    venue           VARCHAR(24)   NOT NULL DEFAULT 'SIM',
    executed_at     DATETIME(6)   NOT NULL,
    SHARD KEY (run_id),
    SORT KEY (as_of_date, agent_id, ticker),
    KEY (agent_id) USING HASH
);

-- Live positions per agent (hot, point-lookup, updated every rebalance).
CREATE ROWSTORE TABLE IF NOT EXISTS positions (
    agent_id        VARCHAR(64)   NOT NULL,
    ticker          VARCHAR(16)   NOT NULL,
    qty             DOUBLE        NOT NULL,
    avg_cost        DOUBLE        NOT NULL,        -- cost basis / share
    last_price      DOUBLE        NOT NULL,
    market_value    DOUBLE        NOT NULL,
    weight          DOUBLE        NOT NULL,        -- fraction of NAV
    realized_pnl    DOUBLE        NOT NULL DEFAULT 0,
    as_of_date      DATE          NOT NULL,
    updated_at      DATETIME(6)   NOT NULL,
    PRIMARY KEY (agent_id, ticker)
);

-- Daily position history (append-only, for time-travel + attribution).
CREATE TABLE IF NOT EXISTS position_snapshots (
    agent_id        VARCHAR(64)   NOT NULL,
    as_of_date      DATE          NOT NULL,
    ticker          VARCHAR(16)   NOT NULL,
    qty             DOUBLE        NOT NULL,
    last_price      DOUBLE        NOT NULL,
    market_value    DOUBLE        NOT NULL,
    weight          DOUBLE        NOT NULL,
    SHARD KEY (agent_id),
    SORT KEY (as_of_date, agent_id, ticker)
);

-- Per-agent daily NAV / return / cumulative P&L series (the equity curve).
CREATE TABLE IF NOT EXISTS nav_history (
    agent_id        VARCHAR(64)   NOT NULL,
    as_of_date      DATE          NOT NULL,
    nav             DOUBLE        NOT NULL,        -- total portfolio value
    cash            DOUBLE        NOT NULL,
    invested        DOUBLE        NOT NULL,
    daily_return    DOUBLE,                        -- vs prior NAV
    cum_return      DOUBLE,                        -- vs inception NAV
    realized_pnl    DOUBLE        NOT NULL DEFAULT 0,
    unrealized_pnl  DOUBLE        NOT NULL DEFAULT 0,
    turnover        DOUBLE        NOT NULL DEFAULT 0,   -- Σ|Δw| this period
    tcost           DOUBLE        NOT NULL DEFAULT 0,   -- commission + slippage cost
    SHARD KEY (agent_id),
    SORT KEY (as_of_date, agent_id)
);

-- Per-run risk analytics (the optimizer's own view + realized backtest stats).
CREATE TABLE IF NOT EXISTS risk_metrics (
    run_id          VARCHAR(64)   NOT NULL,
    agent_id        VARCHAR(64)   NOT NULL,
    as_of_date      DATE          NOT NULL,
    exp_return      DOUBLE,                        -- annualized expected return
    volatility      DOUBLE,                        -- annualized vol
    sharpe          DOUBLE,
    cvar            DOUBLE,                        -- conditional value-at-risk (tail)
    var_95          DOUBLE,
    max_drawdown    DOUBLE,
    turnover        DOUBLE,
    n_positions     INT,
    gross_exposure  DOUBLE,
    net_exposure    DOUBLE,
    created_at      DATETIME(6)   NOT NULL,
    SHARD KEY (run_id),
    SORT KEY (as_of_date, agent_id)
);

-- Immutable compliance audit trail: every material event in the trade lifecycle
-- (run started, weights solved, order created, filled, position updated, memory
-- written). Append-only, the record a Goldman compliance desk would demand.
CREATE TABLE IF NOT EXISTS trade_audit (
    audit_id        VARCHAR(80)   NOT NULL,
    ts              DATETIME(6)   NOT NULL,
    agent_id        VARCHAR(64)   NOT NULL,
    run_id          VARCHAR(64),
    event_type      VARCHAR(40)   NOT NULL,        -- RUN_START | SCENARIOS | SOLVE | ORDER | FILL | POSITION | NAV | MEMORY | RUN_END | ERROR
    entity_ref      VARCHAR(80),                   -- order_id / exec_id / memory_id it concerns
    ticker          VARCHAR(16),
    detail          JSON,                          -- structured before/after + reason
    actor           VARCHAR(64)   NOT NULL DEFAULT 'agent',
    SHARD KEY (run_id),
    SORT KEY (ts, agent_id)
);
