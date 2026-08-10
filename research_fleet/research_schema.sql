-- ============================================================================
-- Research Agents — SingleStore schema (extends portfolio_agents DB)
--
-- A fleet of autonomous auto-research agents (OpenClaw-through-NemoClaw, one per
-- tiny EC2) research and test trading/portfolio strategies, then write their
-- findings to SingleStore OVER TIME. This schema is the shared substrate:
--   * a research WORK QUEUE the fleet pulls from (so N agents don't collide),
--   * a registry of the research agents + their EC2 identity,
--   * HYPOTHESES the agents form, EXPERIMENTS they run, and FINDINGS they log,
--   * an append-only activity LOG (what each agent did, each step),
--   * a persisted-memory store with Qwen VECTOR(1024) recall (same pattern as
--     the trading agents) so a research agent recalls prior findings before it
--     designs the next experiment,
--   * an Aura Analyst query log for the NL-over-SingleStore analysis phase.
--
-- Engine choices: ROWSTORE for hot/point-lookup/queue rows; COLUMNSTORE for the
-- high-volume append-only logs + experiment results.
-- DB: portfolio_agents (same workspace as the trading demo).
-- ============================================================================

-- ---------------------------------------------------------------------------
-- The research agents (one row per EC2-hosted agent)
-- ---------------------------------------------------------------------------
CREATE ROWSTORE TABLE IF NOT EXISTS research_agents (
    agent_id        VARCHAR(64)   NOT NULL,        -- e.g. 'research-01'
    display_name    VARCHAR(128)  NOT NULL,
    focus_area      VARCHAR(128)  NOT NULL,        -- e.g. 'momentum', 'mean-reversion', 'vol-targeting', 'factor', 'regime'
    persona         TEXT,                          -- the research mandate / system framing
    runner          VARCHAR(48)   NOT NULL DEFAULT 'openclaw-nemoclaw',
    model           VARCHAR(64),                   -- claude model id used
    instance_id     VARCHAR(32),                   -- EC2 instance id
    private_ip      VARCHAR(24),
    az              VARCHAR(24),
    status          VARCHAR(24)   NOT NULL DEFAULT 'provisioning', -- provisioning|active|idle|stopped|error
    heartbeat_at    DATETIME(6),                   -- last time the agent phoned home
    created_at      DATETIME(6)   NOT NULL,
    PRIMARY KEY (agent_id),
    KEY (status) USING HASH
);

-- ---------------------------------------------------------------------------
-- Research work queue — the fleet claims tasks atomically (status transitions)
-- ---------------------------------------------------------------------------
CREATE ROWSTORE TABLE IF NOT EXISTS research_tasks (
    task_id         VARCHAR(64)   NOT NULL,
    title           VARCHAR(256)  NOT NULL,
    focus_area      VARCHAR(128)  NOT NULL,
    prompt          TEXT          NOT NULL,        -- the research brief handed to the agent
    priority        INT           NOT NULL DEFAULT 5,   -- lower = sooner
    status          VARCHAR(24)   NOT NULL DEFAULT 'pending', -- pending|claimed|running|done|failed
    claimed_by      VARCHAR(64),                   -- research_agents.agent_id
    claimed_at      DATETIME(6),
    finished_at     DATETIME(6),
    result_summary  TEXT,
    created_at      DATETIME(6)   NOT NULL,
    PRIMARY KEY (task_id),
    KEY (status, priority) USING HASH,
    KEY (focus_area) USING HASH
);

-- ---------------------------------------------------------------------------
-- Hypotheses the agents form (what they think might work + why)
-- ---------------------------------------------------------------------------
CREATE ROWSTORE TABLE IF NOT EXISTS research_hypotheses (
    hypothesis_id   VARCHAR(64)   NOT NULL,
    agent_id        VARCHAR(64)   NOT NULL,
    task_id         VARCHAR(64),
    statement       TEXT          NOT NULL,        -- "Cross-sectional momentum with 12-1 lookback beats 1/N in trending regimes"
    rationale       TEXT,
    strategy_family VARCHAR(64),                   -- momentum|mean_reversion|vol_target|risk_parity|factor|...
    params          JSON,                          -- proposed parameterization to test
    status          VARCHAR(24)   NOT NULL DEFAULT 'open',   -- open|testing|supported|rejected|inconclusive
    confidence      FLOAT         NOT NULL DEFAULT 0.5,
    created_at      DATETIME(6)   NOT NULL,
    PRIMARY KEY (hypothesis_id),
    KEY (agent_id) USING HASH
);

-- ---------------------------------------------------------------------------
-- Experiments — a concrete backtest/analysis run of a hypothesis
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS research_experiments (
    experiment_id   VARCHAR(64)   NOT NULL,
    agent_id        VARCHAR(64)   NOT NULL,
    hypothesis_id   VARCHAR(64),
    task_id         VARCHAR(64),
    strategy_family VARCHAR(64),
    universe        VARCHAR(64),                   -- e.g. 'sp500-top60'
    lookback_start  DATE,
    lookback_end    DATE,
    params          JSON          NOT NULL,        -- exact config tested
    -- realized backtest metrics
    ann_return      DOUBLE,
    ann_vol         DOUBLE,
    sharpe          DOUBLE,
    sortino         DOUBLE,
    max_drawdown    DOUBLE,
    turnover        DOUBLE,
    cvar_95         DOUBLE,
    win_rate        DOUBLE,
    n_rebalances    INT,
    benchmark_sharpe DOUBLE,                        -- vs 1/N benchmark
    beats_benchmark TINYINT,
    method          VARCHAR(48),                   -- 'python-backtest' | 'aura-analyst' | 'llm-analysis'
    engine          VARCHAR(24)   NOT NULL DEFAULT 'cpu',
    status          VARCHAR(24)   NOT NULL DEFAULT 'ok',    -- ok|failed
    error           TEXT,
    started_at      DATETIME(6)   NOT NULL,
    finished_at     DATETIME(6),
    SHARD KEY (experiment_id),
    SORT KEY (started_at, agent_id)
);

-- ---------------------------------------------------------------------------
-- Findings — the durable, human-readable conclusions (the "results over time")
-- Each has a Qwen VECTOR embedding so agents recall prior findings semantically.
-- ---------------------------------------------------------------------------
CREATE ROWSTORE TABLE IF NOT EXISTS research_findings (
    finding_id      VARCHAR(64)   NOT NULL,
    agent_id        VARCHAR(64)   NOT NULL,
    task_id         VARCHAR(64),
    experiment_id   VARCHAR(64),
    hypothesis_id   VARCHAR(64),
    kind            VARCHAR(24)   NOT NULL DEFAULT 'finding',  -- finding|insight|caveat|next_step
    title           VARCHAR(256),
    content         TEXT          NOT NULL,        -- natural-language finding the agent (and peers) re-read
    embedding       VECTOR(1024)  NOT NULL,        -- Qwen embedding of `content`
    strategy_family VARCHAR(64),
    metrics         JSON,                          -- structured snapshot of the supporting numbers
    importance      FLOAT         NOT NULL DEFAULT 0.6,
    tags            JSON,
    created_at      DATETIME(6)   NOT NULL,
    PRIMARY KEY (finding_id),
    KEY (agent_id) USING HASH,
    KEY (strategy_family) USING HASH
);

-- ---------------------------------------------------------------------------
-- Activity log — append-only trace of what each agent did, step by step
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS research_activity (
    activity_id     VARCHAR(64)   NOT NULL,
    ts              DATETIME(6)   NOT NULL,
    agent_id        VARCHAR(64)   NOT NULL,
    task_id         VARCHAR(64),
    phase           VARCHAR(40)   NOT NULL,        -- START|RECALL|HYPOTHESIS|EXPERIMENT|FINDING|ANALYST|HEARTBEAT|END|ERROR
    detail          JSON,
    tokens_in       INT,
    tokens_out      INT,
    SHARD KEY (agent_id),
    SORT KEY (ts, agent_id)
);

-- ---------------------------------------------------------------------------
-- Aura Analyst query log — NL questions the agents ask over SingleStore
-- (populated once the Aura Analyst domain is crawled + key provided)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS research_analyst_queries (
    query_id        VARCHAR(64)   NOT NULL,
    agent_id        VARCHAR(64)   NOT NULL,
    task_id         VARCHAR(64),
    question        TEXT          NOT NULL,        -- English question sent to Aura Analyst
    generated_sql   TEXT,                          -- SQL Aura returned
    row_count       INT,
    answer          TEXT,                          -- Aura's narrated answer
    latency_ms      DOUBLE,
    status          VARCHAR(24)   NOT NULL DEFAULT 'ok',
    created_at      DATETIME(6)   NOT NULL,
    SHARD KEY (query_id),
    SORT KEY (created_at, agent_id)
);
