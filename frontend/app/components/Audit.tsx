"use client";

import { useMemo, useState } from "react";
import { api, AuditEvent } from "../lib/api";
import { useApi } from "../lib/useApi";
import { Panel, Loading, ErrorState, EmptyState } from "./ui";
import { useAgents } from "../lib/AgentsContext";
import AgentSelect from "./AgentSelect";
import { agentColor } from "../lib/theme";
import { fmtDateTime } from "../lib/format";

// Color-coded event types for the compliance trail.
const EVENT_COLOR: Record<string, string> = {
  RUN_START: "#58a6ff",
  SCENARIOS: "#17becf",
  SOLVE: "#76B900",
  ORDER: "#d29922",
  FILL: "#bc8cff",
  POSITION: "#8b98a5",
  NAV: "#76B900",
  MEMORY: "#bc8cff",
  RUN_END: "#58a6ff",
  ERROR: "#f85149",
};

function summarize(detail: Record<string, unknown> | null): string {
  if (!detail || typeof detail !== "object") return "";
  const parts: string[] = [];
  for (const [k, v] of Object.entries(detail)) {
    if (v == null || typeof v === "object") continue;
    parts.push(`${k}=${v}`);
    if (parts.length >= 4) break;
  }
  return parts.join("  ");
}

export default function Audit() {
  const [agent, setAgent] = useState("all");
  const { agents } = useAgents();
  const { data, error, loading, refreshing } = useApi<AuditEvent[]>(
    (signal) => api.audit({ agent: agent === "all" ? undefined : agent, limit: 120 }, signal),
    [agent],
    20000,
  );

  const nameOf = (id: string) =>
    agents.find((a) => a.agent_id === id)?.display_name ?? id;

  const events = useMemo(() => data ?? [], [data]);

  return (
    <Panel
      id="audit"
      title="Compliance audit trail"
      subtitle="Immutable, append-only record of every material event in the trade lifecycle"
      accent="#58a6ff"
      right={
        <div className="panel-controls">
          {refreshing && <span className="live-pill">● live</span>}
          <AgentSelect value={agent} onChange={setAgent} allowAll />
        </div>
      }
    >
      {loading && !data ? (
        <Loading label="Loading audit trail" />
      ) : error && !data ? (
        <ErrorState message={error} />
      ) : events.length === 0 ? (
        <EmptyState
          title="No audit events yet"
          hint="Every run start, solve, order, fill, position & memory write is logged here."
          icon="§"
        />
      ) : (
        <div className="table-scroll audit-scroll">
          <table className="data-table audit mono">
            <thead>
              <tr>
                <th className="ta-left">Timestamp</th>
                <th className="ta-left">Event</th>
                <th className="ta-left">Agent</th>
                <th className="ta-left">Ticker</th>
                <th className="ta-left">Entity</th>
                <th className="ta-left">Detail</th>
                <th className="ta-left">Actor</th>
              </tr>
            </thead>
            <tbody>
              {events.map((e, i) => {
                const c = EVENT_COLOR[e.event_type] ?? "#8b98a5";
                return (
                  <tr key={`${e.ts}-${i}`}>
                    <td className="ta-left dim">{fmtDateTime(e.ts)}</td>
                    <td className="ta-left">
                      <span
                        className="event-tag"
                        style={{ color: c, borderColor: c }}
                      >
                        {e.event_type}
                      </span>
                    </td>
                    <td className="ta-left">
                      <span className="cell-agent">
                        <span
                          className="mini-swatch"
                          style={{ background: agentColor(e.agent_id) }}
                        />
                        {nameOf(e.agent_id)}
                      </span>
                    </td>
                    <td className="ta-left ticker">{e.ticker ?? "—"}</td>
                    <td className="ta-left dim entity-ref">{e.entity_ref ?? "—"}</td>
                    <td className="ta-left dim detail-cell">{summarize(e.detail)}</td>
                    <td className="ta-left dim">{e.actor}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  );
}
