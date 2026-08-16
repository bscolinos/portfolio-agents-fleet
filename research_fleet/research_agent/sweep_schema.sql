-- ============================================================================
-- Strategy parameter-sweep results — SingleStore schema (extends portfolio_agents DB)
--
-- A companion engine backtests THOUSANDS of strategy configs on real S&P 500
-- prices, running each on an in-sample (IS) window and an out-of-sample (OOS)
-- window, and appends one row per config to `sweep_results`. `sweep_runs` records
-- one row per sweep (the windows, seed, families, status). `sweep_analysis`
-- stores the compact, overfitting-aware summary produced by `sweep_analyze.py`
-- (the HONEST ranking: OOS-first, with a multiple-testing / deflated-Sharpe
-- haircut so the naive "highest Sharpe" false discovery is not shipped to a
-- real-money decision).
--
-- Engine choices (matching research_schema.sql conventions):
--   * COLUMNSTORE for `sweep_results` — high-volume append-only + analytical
--     ranking scans. SHARD KEY(result_id); SORT KEY(sweep_id) so all rows for a
--     sweep are co-located/ordered for the per-sweep ranking read.
--   * ROWSTORE for `sweep_runs` and `sweep_analysis` — low-volume, point-lookup
--     by primary key / sweep_id.
-- DB: portfolio_agents (same workspace as the trading + research demos).
-- Both this analyzer and the engine may CREATE TABLE IF NOT EXISTS the shared
-- tables with IDENTICAL DDL — it is idempotent as long as the columns match.
-- ============================================================================

-- ---------------------------------------------------------------------------
-- One row per backtested strategy config (IS + OOS metrics side by side).
-- `is_oos_sharpe_gap` = is_sharpe - oos_sharpe (large positive = overfit).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sweep_results (
    result_id           VARCHAR(64)   NOT NULL,        -- e.g. 'res-<hex>'
    sweep_id            VARCHAR(64),                   -- FK -> sweep_runs.sweep_id
    family              VARCHAR(64),                   -- strategy family (one of 8)
    params              JSON,                          -- the exact config tested
    -- in-sample (IS) metrics
    is_sharpe           DOUBLE,
    is_ann_return       DOUBLE,
    is_ann_vol          DOUBLE,
    is_max_drawdown     DOUBLE,
    is_turnover         DOUBLE,
    is_beats_benchmark  TINYINT,
    -- out-of-sample (OOS) metrics
    oos_sharpe          DOUBLE,
    oos_ann_return      DOUBLE,
    oos_ann_vol         DOUBLE,
    oos_max_drawdown    DOUBLE,
    oos_turnover        DOUBLE,
    oos_beats_benchmark TINYINT,
    -- degradation + costing
    is_oos_sharpe_gap   DOUBLE,                         -- is_sharpe - oos_sharpe
    all_in_cost_bps     DOUBLE,                         -- total cost charged per unit turnover
    universe_n          INT,
    error               TEXT,                           -- NULL when the config backtested cleanly
    created_at          DATETIME(6),
    SHARD KEY (result_id),
    SORT KEY (sweep_id)
);

-- ---------------------------------------------------------------------------
-- One row per sweep: the windows, seed, families exercised, and status.
-- ---------------------------------------------------------------------------
CREATE ROWSTORE TABLE IF NOT EXISTS sweep_runs (
    sweep_id     VARCHAR(64)   NOT NULL,               -- e.g. 'swp-<hex>'
    target_n     INT,                                  -- configs requested
    actual_n     INT,                                  -- configs actually written
    seed         INT,
    is_start     DATE,
    is_end       DATE,
    oos_start    DATE,
    oos_end      DATE,
    families     TEXT,                                 -- comma/JSON list of families exercised
    status       VARCHAR(24),                          -- pending|running|done|failed
    started_at   DATETIME(6),
    finished_at  DATETIME(6),
    notes        TEXT,
    PRIMARY KEY (sweep_id)
);

-- ---------------------------------------------------------------------------
-- One row per `sweep_analyze` run: the compact HONEST-ranking summary.
-- Owned by the analyzer (the engine never writes here).
-- ---------------------------------------------------------------------------
CREATE ROWSTORE TABLE IF NOT EXISTS sweep_analysis (
    analysis_id            VARCHAR(64)  NOT NULL,       -- e.g. 'ana-<hex>'
    sweep_id               VARCHAR(64),                 -- FK -> sweep_runs.sweep_id
    best_oos_family        VARCHAR(64),                 -- family of the OOS #1 config
    best_oos_sharpe        DOUBLE,                       -- OOS Sharpe of the OOS #1 config
    best_oos_params        JSON,                         -- params of the OOS #1 config
    n_total                INT,                          -- valid (error IS NULL, metrics present) trials ranked
    n_robust               INT,                          -- passed the overfitting-robustness gate
    n_survive_mtc          INT,                          -- cleared the multiple-testing / deflated-Sharpe hurdle
    naive_is_best_params   JSON,                         -- params of the highest-IS-Sharpe config (the WRONG pick)
    naive_is_best_oos_rank INT,                          -- where that IS winner lands in the OOS ranking
    summary_json           JSON,                         -- full compact summary (thresholds, family stats, top-K)
    created_at             DATETIME(6),
    PRIMARY KEY (analysis_id),
    KEY (sweep_id) USING HASH
);
