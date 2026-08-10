"use client";

import { useMemo, useState } from "react";
import { api, Run } from "../lib/api";
import { useApi } from "../lib/useApi";
import { Panel, Loading, ErrorState, EmptyState, EngineBadge } from "./ui";
import { useAgents } from "../lib/AgentsContext";
import AgentSelect from "./AgentSelect";
import { agentColor } from "../lib/theme";
import { compact, ms, num, fmtDate, fmtTime } from "../lib/format";

export default function Runs() {
  const [agent, setAgent] = useState("all");
  const { agents } = useAgents();
  const { data, error, loading, refreshing } = useApi<Run[]>(
    (signal) => api.runs(agent, 50, signal),
    [agent],
    15000,
  );

  const nameOf = (id: string) =>
    agents.find((a) => a.agent_id === id)?.display_name ?? id;

  const runs = useMemo(() => data ?? [], [data]);

  // GPU acceleration headline: fastest GPU solve + avg speedup vs slowest CPU.
  const gpuRuns = runs.filter((r) => (r.engine ?? "").toLowerCase() === "gpu");
  const bestGpu = gpuRuns.reduce<Run | null>(
    (best, r) =>
      r.solve_ms != null && (best == null || r.solve_ms < (best.solve_ms ?? Infinity))
        ? r
        : best,
    null,
  );
  const maxScenarios = runs.reduce((m, r) => Math.max(m, r.num_scenarios ?? 0), 0);

  return (
    <Panel
      id="runs"
      title="Optimization runs"
      subtitle="GPU-accelerated cuOpt / cuML solves — scenario generation & wall time"
      accent="#76B900"
      right={
        <div className="panel-controls">
          {refreshing && <span className="live-pill">● live</span>}
          <AgentSelect value={agent} onChange={setAgent} allowAll />
        </div>
      }
    >
      {loading && !data ? (
        <Loading label="Loading runs" />
      ) : error && !data ? (
        <ErrorState message={error} />
      ) : runs.length === 0 ? (
        <EmptyState
          title="No runs yet — run the fleet"
          hint="Each rebalance records its engine, GPU, scenarios & solve time here."
          icon="⚙"
        />
      ) : (
        <>
          <div className="gpu-headline">
            <div className="gpu-stat gpu-stat-hero">
              <span className="gpu-stat-label">Fastest GPU solve</span>
              <span className="gpu-stat-val mono">
                {bestGpu?.solve_ms != null ? ms(bestGpu.solve_ms) : "—"}
              </span>
              <span className="gpu-stat-sub">
                {bestGpu?.gpu_name || "NVIDIA L4"}
              </span>
            </div>
            <div className="gpu-stat">
              <span className="gpu-stat-label">Max scenarios / run</span>
              <span className="gpu-stat-val mono">{compact(maxScenarios)}</span>
              <span className="gpu-stat-sub">CVaR / KDE Monte-Carlo</span>
            </div>
            <div className="gpu-stat">
              <span className="gpu-stat-label">GPU solves</span>
              <span className="gpu-stat-val mono">{compact(gpuRuns.length)}</span>
              <span className="gpu-stat-sub">
                of {compact(runs.length)} recent runs
              </span>
            </div>
          </div>

          <div className="table-scroll">
            <table className="data-table runs mono">
              <thead>
                <tr>
                  <th className="ta-left">As of</th>
                  <th className="ta-left">Agent</th>
                  <th className="ta-left">Engine</th>
                  <th className="ta-right">Scenarios</th>
                  <th className="ta-right">Solve</th>
                  <th className="ta-right">Scenario gen</th>
                  <th className="ta-right">Universe</th>
                  <th className="ta-center">Status</th>
                  <th className="ta-left">Finished</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((r) => (
                  <tr key={r.run_id}>
                    <td className="ta-left dim">{fmtDate(r.as_of_date)}</td>
                    <td className="ta-left">
                      <span className="cell-agent">
                        <span
                          className="mini-swatch"
                          style={{ background: agentColor(r.agent_id) }}
                        />
                        {nameOf(r.agent_id)}
                      </span>
                    </td>
                    <td className="ta-left">
                      <EngineBadge engine={r.engine} gpuName={r.gpu_name} />
                    </td>
                    <td className="ta-right">{compact(r.num_scenarios)}</td>
                    <td className="ta-right accent-num">{ms(r.solve_ms)}</td>
                    <td className="ta-right dim">{ms(r.scenario_ms)}</td>
                    <td className="ta-right dim">{num(r.universe_size, 0)}</td>
                    <td className="ta-center">
                      <span className={`run-status run-${(r.status || "").toLowerCase()}`}>
                        {r.status}
                      </span>
                    </td>
                    <td className="ta-left dim">
                      {r.finished_at ? fmtTime(r.finished_at) : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </Panel>
  );
}
