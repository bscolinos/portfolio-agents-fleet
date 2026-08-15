-- ============================================================================
-- Re-score results — SingleStore schema (extends portfolio_agents DB)
--
-- The original backtester had two correctness bugs (turnover cost undercharged;
-- survivorship + look-ahead bias in universe construction) that mis-selected the
-- "winner" strategy for a REAL-MONEY decision. `rescore.py` re-runs every row in
-- research_experiments through the CORRECTED engine and records the before/after
-- here. The original research_experiments rows are LEFT UNTOUCHED — this table is
-- an append-only audit of the correction so the true ranking is defensible.
-- ============================================================================

CREATE TABLE IF NOT EXISTS experiment_rescores (
    rescore_id       VARCHAR(64)   NOT NULL,        -- e.g. 'rsc-<hex>'
    experiment_id    VARCHAR(64)   NOT NULL,        -- FK -> research_experiments
    agent_id         VARCHAR(64)   NOT NULL,
    strategy_family  VARCHAR(64),
    lookback_start   DATE,
    lookback_end     DATE,
    universe_n       INT,
    -- before/after correctness metrics
    old_sharpe       DOUBLE,                         -- as stored in research_experiments
    new_sharpe       DOUBLE,                         -- corrected engine
    delta_sharpe     DOUBLE,                         -- new - old
    old_turnover     DOUBLE,
    new_turnover     DOUBLE,
    old_beats_benchmark   TINYINT,
    beats_benchmark_new   TINYINT,
    -- corrected cost model (bps on traded notional)
    turnover_cost_bps DOUBLE,                        -- resolved commission/spread leg
    slippage_bps     DOUBLE,                         -- modeled market-impact leg
    all_in_cost_bps  DOUBLE,                         -- total charged per unit turnover
    gross_cost       DOUBLE,                         -- Σ turnover * all_in_cost_bps/1e4
    params           JSON          NOT NULL,         -- the experiment's own config
    data_caveats     TEXT,                           -- residual bias disclosure
    status           VARCHAR(24)   NOT NULL DEFAULT 'ok',  -- ok|failed
    error            TEXT,
    rescored_at      DATETIME(6)   NOT NULL,
    SHARD KEY (experiment_id),
    SORT KEY (new_sharpe, rescored_at)
);
