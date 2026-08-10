"use client";

import { api, LeaderboardRow } from "../lib/api";
import { useApi } from "../lib/useApi";
import { Panel, Loading, ErrorState, EmptyState, EngineBadge } from "./ui";
import { agentColor } from "../lib/theme";
import {
  signedPct,
  num,
  moneyCompact,
  pnlClass,
  fmtDate,
} from "../lib/format";

export default function Leaderboard() {
  const { data, error, loading, refreshing } = useApi<LeaderboardRow[]>(
    (signal) => api.leaderboard(signal),
    [],
    15000,
  );

  return (
    <Panel
      id="leaderboard"
      title="Agent leaderboard"
      subtitle="Autonomous strategies ranked by cumulative return"
      accent="#76B900"
      right={
        refreshing ? <span className="live-pill">● live</span> : undefined
      }
    >
      {loading && !data ? (
        <Loading label="Ranking agents" />
      ) : error && !data ? (
        <ErrorState message={error} />
      ) : !data || data.length === 0 ? (
        <EmptyState
          title="No agents ranked yet"
          hint="Seed the roster and run the fleet to populate the leaderboard."
          icon="▲"
        />
      ) : (
        <div className="leader-grid">
          {data.map((a, i) => {
            const color = agentColor(a.agent_id, a.color, i);
            const cls = pnlClass(a.cum_return);
            return (
              <article
                key={a.agent_id}
                className="leader-card"
                style={{ ["--accent" as string]: color }}
              >
                <div className="leader-rank">#{a.rank ?? i + 1}</div>
                <div className="leader-top">
                  <span className="leader-swatch" style={{ background: color }} />
                  <div className="leader-id">
                    <h3>{a.display_name}</h3>
                    <span className="leader-strat">
                      {a.strategy_type?.replace(/_/g, " ")}
                    </span>
                  </div>
                </div>

                <div className={`leader-return ${cls} mono`}>
                  {signedPct(a.cum_return)}
                </div>
                <div className="leader-return-label">cumulative return</div>

                <div className="leader-stats">
                  <div>
                    <span className="ls-label">Sharpe</span>
                    <span className="ls-val mono">{num(a.sharpe, 2)}</span>
                  </div>
                  <div>
                    <span className="ls-label">NAV</span>
                    <span className="ls-val mono">
                      {moneyCompact(a.latest_nav)}
                    </span>
                  </div>
                  <div>
                    <span className="ls-label">Positions</span>
                    <span className="ls-val mono">{a.n_positions ?? 0}</span>
                  </div>
                  <div>
                    <span className="ls-label">Max DD</span>
                    <span className="ls-val mono neg">
                      {a.max_drawdown != null ? signedPct(-Math.abs(a.max_drawdown)) : "—"}
                    </span>
                  </div>
                </div>

                <div className="leader-foot">
                  <EngineBadge engine={a.last_engine ?? a.engine} gpuName={a.last_gpu_name} />
                  {a.last_run_at && (
                    <span className="leader-lastrun">
                      last run {fmtDate(a.last_run_at)}
                    </span>
                  )}
                </div>
                <p className="leader-objective" title={a.objective}>
                  {a.objective}
                </p>
              </article>
            );
          })}
        </div>
      )}
    </Panel>
  );
}
