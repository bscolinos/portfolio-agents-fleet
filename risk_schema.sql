-- ============================================================================
-- Portfolio Agents — RISK / SAFETY-LAYER schema
--
-- The pre-trade safety layer that must sit between a research "finding" and any
-- real-money order. Four capabilities, one engine:
--   1. RISK DECISIONS — an immutable, queryable record of every pre-trade gate
--      evaluation (approved/rejected + every violation + the full check set).
--      Complements trade_audit (which also gets a RISK_CHECK row) with a
--      first-class, analytics-friendly table.
--   2. KILL SWITCHES — persisted global + per-agent circuit breakers. Because
--      they live in SingleStore (not RAM), an engaged switch survives process
--      death and is visible fleet-wide the instant an operator trips it.
--   3. PAPER BOOK — a shadow trade lifecycle (orders / positions / nav) that
--      mirrors the real trading.py tables but is NEVER the real book. This is
--      the "paper-trade first" gate before any live decision.
--
-- Engine choices mirror schema.sql:
--   * ROWSTORE for small, hot, point-lookup / frequently-updated tables
--     (kill_switches, paper_positions).
--   * COLUMNSTORE for append-mostly analytical series
--     (risk_decisions, paper_orders, paper_nav_history).
--   * DATETIME(6) everywhere for microsecond audit ordering.
--
-- Idempotent: every statement is CREATE TABLE IF NOT EXISTS. Apply with
--   python apply_risk_schema.py
-- ============================================================================

-- ---------------------------------------------------------------------------
-- RISK DECISIONS (columnstore, append-only audit of every gate evaluation)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS risk_decisions (
    decision_id     VARCHAR(80)   NOT NULL,        -- e.g. 'rsk-<uuid>'
    ts              DATETIME(6)   NOT NULL,
    agent_id        VARCHAR(64)   NOT NULL,
    run_id          VARCHAR(64),                   -- run being gated (nullable)
    as_of_date      DATE          NOT NULL,        -- the rebalance / decision date
    mode            VARCHAR(16)   NOT NULL DEFAULT 'paper',  -- paper | live
    approved        TINYINT       NOT NULL,        -- 1 = cleared to trade, 0 = rejected
    reason          VARCHAR(512)  NOT NULL,        -- one-line human summary
    n_violations    INT           NOT NULL DEFAULT 0,
    violations      JSON,                          -- [{code, detail, ...}]
    checks          JSON,                          -- full structured check set
    adjusted        TINYINT       NOT NULL DEFAULT 0,  -- 1 = weights were clipped/renormalized
    nav             DOUBLE,                        -- NAV the decision was evaluated against
    gross_exposure  DOUBLE,                        -- Σ|w| of target
    net_exposure    DOUBLE,                        -- Σ w of target
    turnover        DOUBLE,                        -- Σ|Δw| vs current book / NAV
    max_weight      DOUBLE,                        -- largest single-name target weight
    n_names         INT,                           -- number of target names
    actor           VARCHAR(64)   NOT NULL DEFAULT 'risk_gate',
    SHARD KEY (agent_id),
    SORT KEY (ts, agent_id)
);

-- ---------------------------------------------------------------------------
-- KILL SWITCHES (rowstore, hot point-lookup, frequently toggled)
--
-- One row per scope. scope = 'global' trips the whole fleet; scope = <agent_id>
-- trips just that agent. is_engaged(agent) is true if EITHER is engaged.
-- ---------------------------------------------------------------------------
CREATE ROWSTORE TABLE IF NOT EXISTS kill_switches (
    scope           VARCHAR(64)   NOT NULL,        -- 'global' | '<agent_id>'
    engaged         TINYINT       NOT NULL DEFAULT 0,
    reason          VARCHAR(512),
    engaged_by      VARCHAR(64),
    engaged_at      DATETIME(6),
    released_by     VARCHAR(64),
    released_at     DATETIME(6),
    updated_at      DATETIME(6)   NOT NULL,
    PRIMARY KEY (scope)
);

-- ---------------------------------------------------------------------------
-- PAPER BOOK — shadow trade lifecycle (mirror of orders/positions/nav_history)
-- These tables NEVER intersect the real book. paper_trader.py mirrors the exact
-- CostModel + fill math from trading.py, writing here instead of orders/etc.
-- ---------------------------------------------------------------------------

-- Shadow parent orders (columnstore, mirrors `orders`).
CREATE TABLE IF NOT EXISTS paper_orders (
    order_id        VARCHAR(80)   NOT NULL,
    run_id          VARCHAR(64)   NOT NULL,
    agent_id        VARCHAR(64)   NOT NULL,
    as_of_date      DATE          NOT NULL,
    ticker          VARCHAR(16)   NOT NULL,
    side            VARCHAR(4)    NOT NULL,        -- BUY | SELL
    target_weight   DOUBLE        NOT NULL,
    prev_weight     DOUBLE        NOT NULL,
    order_qty       DOUBLE        NOT NULL,        -- signed shares to trade
    ref_price       DOUBLE        NOT NULL,
    order_notional  DOUBLE        NOT NULL,
    fill_price      DOUBLE        NOT NULL,        -- ref_price adjusted for slippage
    notional        DOUBLE        NOT NULL,        -- |qty| * fill_price
    commission      DOUBLE        NOT NULL DEFAULT 0,
    slippage_bps    DOUBLE        NOT NULL DEFAULT 0,
    venue           VARCHAR(24)   NOT NULL DEFAULT 'PAPER',
    status          VARCHAR(16)   NOT NULL DEFAULT 'FILLED',
    created_at      DATETIME(6)   NOT NULL,
    SHARD KEY (run_id),
    SORT KEY (as_of_date, agent_id, ticker),
    KEY (agent_id) USING HASH
);

-- Shadow live positions (rowstore, mirrors `positions`).
CREATE ROWSTORE TABLE IF NOT EXISTS paper_positions (
    agent_id        VARCHAR(64)   NOT NULL,
    ticker          VARCHAR(16)   NOT NULL,
    qty             DOUBLE        NOT NULL,
    avg_cost        DOUBLE        NOT NULL,        -- cost basis / share
    last_price      DOUBLE        NOT NULL,
    market_value    DOUBLE        NOT NULL,
    weight          DOUBLE        NOT NULL,        -- fraction of paper NAV
    realized_pnl    DOUBLE        NOT NULL DEFAULT 0,
    as_of_date      DATE          NOT NULL,
    updated_at      DATETIME(6)   NOT NULL,
    PRIMARY KEY (agent_id, ticker)
);

-- Shadow NAV / equity curve (columnstore, mirrors `nav_history`).
CREATE TABLE IF NOT EXISTS paper_nav_history (
    agent_id        VARCHAR(64)   NOT NULL,
    as_of_date      DATE          NOT NULL,
    nav             DOUBLE        NOT NULL,
    cash            DOUBLE        NOT NULL,
    invested        DOUBLE        NOT NULL,
    daily_return    DOUBLE,
    cum_return      DOUBLE,
    realized_pnl    DOUBLE        NOT NULL DEFAULT 0,
    unrealized_pnl  DOUBLE        NOT NULL DEFAULT 0,
    turnover        DOUBLE        NOT NULL DEFAULT 0,
    tcost           DOUBLE        NOT NULL DEFAULT 0,
    SHARD KEY (agent_id),
    SORT KEY (as_of_date, agent_id)
);
