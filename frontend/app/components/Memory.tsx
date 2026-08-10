"use client";

import { useEffect, useMemo, useState } from "react";
import { api, Memory as Mem, RecallResult } from "../lib/api";
import { useApi } from "../lib/useApi";
import { Panel, Loading, ErrorState, EmptyState, Chip } from "./ui";
import { useAgents } from "../lib/AgentsContext";
import AgentSelect from "./AgentSelect";
import { agentColor, KIND_META } from "../lib/theme";
import { fmtDate, fmtDateTime } from "../lib/format";

const EXAMPLE_QUERIES = [
  "high turnover",
  "tail risk in drawdown",
  "concentration limits",
  "GPU speedup vs CPU",
  "regime shift signal",
];

function KindBadge({ kind }: { kind: string }) {
  const meta = KIND_META[kind?.toLowerCase()] ?? {
    label: (kind || "note").toUpperCase(),
    color: "#8b98a5",
    bg: "rgba(139,152,165,0.14)",
  };
  return (
    <span className="kind-badge" style={{ color: meta.color, background: meta.bg }}>
      {meta.label}
    </span>
  );
}

function ImportanceMeter({ value }: { value: number }) {
  const v = Math.max(0, Math.min(1, value ?? 0));
  return (
    <span className="imp-meter" title={`importance ${v.toFixed(2)}`}>
      <span className="imp-track">
        <span className="imp-fill" style={{ width: `${v * 100}%` }} />
      </span>
    </span>
  );
}

export default function MemoryPanel() {
  const { agents } = useAgents();
  const [agent, setAgent] = useState<string>("");

  useEffect(() => {
    if (!agent && agents.length > 0) setAgent(agents[0].agent_id);
  }, [agents, agent]);

  const color = agent ? agentColor(agent) : "#76B900";

  return (
    <Panel
      id="memory"
      title={
        <>
          Persisted memory <span className="hero-tag">+ live vector recall</span>
        </>
      }
      subtitle="Each agent's episodic memory lives in SingleStore — recalled by semantic similarity before every rebalance"
      accent={color}
      right={<AgentSelect value={agent || "all"} onChange={setAgent} />}
    >
      <div className="memory-hero">
        <RecallPane agent={agent} color={color} />
        <MemoryFeed agent={agent} color={color} />
      </div>
    </Panel>
  );
}

// ---------------------------------------------------------------------------
// Live recall (semantic vector search)
// ---------------------------------------------------------------------------

function RecallPane({ agent, color }: { agent: string; color: string }) {
  const [q, setQ] = useState("");
  const [submitted, setSubmitted] = useState<string | null>(null);
  const [results, setResults] = useState<RecallResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tookMs, setTookMs] = useState<number | null>(null);

  async function run(query: string) {
    const text = query.trim();
    if (!text || !agent) return;
    setLoading(true);
    setError(null);
    setSubmitted(text);
    const t0 = performance.now();
    try {
      const res = await api.recall(agent, text, 5);
      setResults(res.results ?? []);
      setTookMs(performance.now() - t0);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
      setResults([]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="recall-pane" style={{ ["--accent" as string]: color }}>
      <div className="recall-head">
        <span className="recall-kicker">
          <span className="recall-dot" /> LIVE SEMANTIC RECALL
        </span>
        <span className="recall-sub">
          Qwen embedding · <code>embedding &lt;*&gt; qvec</code> in SingleStore
        </span>
      </div>

      <form
        className="recall-form"
        onSubmit={(e) => {
          e.preventDefault();
          run(q);
        }}
      >
        <input
          className="recall-input mono"
          placeholder="Recall a past learning… e.g. what happened in high-turnover regimes"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          disabled={!agent}
        />
        <button className="recall-btn" type="submit" disabled={!agent || !q.trim() || loading}>
          {loading ? "Searching…" : "Recall"}
        </button>
      </form>

      <div className="recall-chips">
        <span className="recall-chips-label">try:</span>
        {EXAMPLE_QUERIES.map((ex) => (
          <Chip
            key={ex}
            onClick={() => {
              setQ(ex);
              run(ex);
            }}
            active={submitted === ex}
          >
            {ex}
          </Chip>
        ))}
      </div>

      <div className="recall-results">
        {!agent ? (
          <EmptyState title="Select an agent" icon="✦" />
        ) : loading ? (
          <Loading label="Embedding query & searching vectors" />
        ) : error ? (
          <div className="state state-error">
            <span className="mono">{error}</span>
          </div>
        ) : submitted == null ? (
          <div className="recall-hint">
            <span className="recall-hint-icon">✦</span>
            Enter a question — the agent embeds it and ranks its memories by
            vector similarity, exactly as it does before deciding a trade.
          </div>
        ) : results.length === 0 ? (
          <EmptyState
            title="No memories matched"
            hint="This agent may not have run yet — recall needs persisted memories."
            icon="✦"
          />
        ) : (
          <>
            <div className="recall-meta">
              Top {results.length} for{" "}
              <span className="recall-q">“{submitted}”</span>
              {tookMs != null && (
                <span className="recall-took"> · {tookMs.toFixed(0)} ms</span>
              )}
            </div>
            <ol className="recall-list">
              {results.map((r, i) => {
                const score = Math.max(0, Math.min(1, r.score ?? 0));
                return (
                  <li key={i} className="recall-card">
                    <div className="recall-card-head">
                      <KindBadge kind={r.kind} />
                      <span className="score-pill mono" title="cosine similarity">
                        <span
                          className="score-fill"
                          style={{ width: `${score * 100}%` }}
                        />
                        <span className="score-num">{score.toFixed(3)}</span>
                      </span>
                    </div>
                    <p className="recall-content">{r.content}</p>
                    <div className="recall-card-foot dim">
                      {fmtDateTime(r.created_at)}
                    </div>
                  </li>
                );
              })}
            </ol>
          </>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Memory feed (raw persisted memories)
// ---------------------------------------------------------------------------

const KIND_FILTERS = ["all", "observation", "decision", "reflection", "learning"];

function MemoryFeed({ agent, color }: { agent: string; color: string }) {
  const [kind, setKind] = useState("all");
  const { data, error, loading } = useApi<Mem[]>(
    (signal) => api.memory(agent, kind === "all" ? "" : kind, 50, signal),
    [agent, kind],
    agent ? 30000 : 0,
  );

  const items = useMemo(() => data ?? [], [data]);

  return (
    <div className="feed-pane" style={{ ["--accent" as string]: color }}>
      <div className="feed-head">
        <span className="feed-title">Memory feed</span>
        <div className="feed-filters">
          {KIND_FILTERS.map((k) => (
            <button
              key={k}
              className={`feed-filter ${kind === k ? "feed-filter-on" : ""}`}
              onClick={() => setKind(k)}
            >
              {k}
            </button>
          ))}
        </div>
      </div>

      <div className="feed-scroll">
        {!agent ? (
          <EmptyState title="Select an agent" icon="◷" />
        ) : loading && !data ? (
          <Loading label="Loading memories" />
        ) : error && !data ? (
          <ErrorState message={error} />
        ) : items.length === 0 ? (
          <EmptyState
            title="No memories yet"
            hint="Agents write observations, decisions & learnings as they run."
            icon="◷"
          />
        ) : (
          <ul className="feed-list">
            {items.map((m) => (
              <li key={m.memory_id} className="feed-item">
                <div className="feed-item-head">
                  <KindBadge kind={m.kind} />
                  <span className="feed-date dim">
                    {fmtDate(m.as_of_date ?? m.created_at)}
                  </span>
                  <ImportanceMeter value={m.importance} />
                </div>
                <p className="feed-content">{m.content}</p>
                {Array.isArray(m.tags) && m.tags.length > 0 && (
                  <div className="feed-tags">
                    {(m.tags as string[]).slice(0, 6).map((t, i) => (
                      <span key={i} className="feed-tag">
                        {String(t)}
                      </span>
                    ))}
                  </div>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
