-- ============================================================================
-- Aura Analyst proxy — audit + cache schema (portfolio_agents DB)
--
-- The hosted proxy fronts the real SingleStore Aura Analyst domain. Every query
-- an agent makes is logged here (full governance/audit trail — required for a
-- system that will trade real money), and successful answers are cached to cut
-- latency + Aura load and to survive brief upstream outages.
-- ============================================================================

-- Full audit trail of every proxied Aura query (append-only, columnstore).
CREATE TABLE IF NOT EXISTS aura_query_log (
    query_id        VARCHAR(64)   NOT NULL,
    ts              DATETIME(6)   NOT NULL,
    agent_id        VARCHAR(64),                   -- caller (research-01..05, or 'proxy'/'test')
    endpoint        VARCHAR(16)   NOT NULL DEFAULT 'query',  -- query | chat
    question        TEXT          NOT NULL,        -- the NL question sent to Aura
    question_hash   VARCHAR(64)   NOT NULL,        -- sha256(normalized question) for cache key
    generated_sql   TEXT,                          -- SQL Aura produced
    confidence      DOUBLE,                        -- Aura's SQL confidence 0..1
    tables_used     JSON,
    row_count       INT,
    answer_text     TEXT,                          -- Aura's narrated answer (if any)
    error           TEXT,                          -- error message if the query failed
    status          VARCHAR(24)   NOT NULL DEFAULT 'ok',  -- ok | error | upstream_error | timeout | rate_limited | circuit_open | cache
    http_status     INT,
    trace_id        VARCHAR(80),                   -- singlestore-trace-id from Aura (for support)
    latency_ms      DOUBLE,                        -- total time the proxy spent
    upstream_ms     DOUBLE,                        -- time spent in the Aura call itself
    cache_hit       TINYINT       NOT NULL DEFAULT 0,
    attempts        INT           NOT NULL DEFAULT 1,   -- how many upstream tries (retries)
    session_id      VARCHAR(64),
    client_ip       VARCHAR(64),
    SHARD KEY (query_id),
    SORT KEY (ts, agent_id),
    KEY (question_hash) USING HASH,
    KEY (status) USING HASH
);

-- Response cache: question_hash -> the flattened Aura result, with TTL.
-- Rowstore for fast point-lookup on the hot path.
CREATE ROWSTORE TABLE IF NOT EXISTS aura_cache (
    question_hash   VARCHAR(64)   NOT NULL,
    question        TEXT          NOT NULL,
    output_modes    VARCHAR(64)   NOT NULL DEFAULT 'sql,data',
    response_json   TEXT          NOT NULL,        -- the flattened {sql,data,text,...} payload
    confidence      DOUBLE,
    row_count       INT,
    created_at      DATETIME(6)   NOT NULL,
    expires_at      DATETIME(6)   NOT NULL,
    hits            INT           NOT NULL DEFAULT 0,
    PRIMARY KEY (question_hash, output_modes)
);
