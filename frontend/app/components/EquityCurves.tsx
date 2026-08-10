"use client";

import { useMemo, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { api, NavResponse } from "../lib/api";
import { useApi } from "../lib/useApi";
import { Panel, Loading, ErrorState, EmptyState } from "./ui";
import { agentColor } from "../lib/theme";
import { signedPct, fmtDate } from "../lib/format";

interface Row {
  date: string;
  [agentId: string]: number | string | null;
}

export default function EquityCurves() {
  const { data, error, loading } = useApi<NavResponse>(
    (signal) => api.nav("all", signal),
    [],
    30000,
  );
  const [hidden, setHidden] = useState<Set<string>>(new Set());

  const series = data?.series ?? [];

  const meta = useMemo(
    () =>
      series.map((s, i) => ({
        id: s.agent_id,
        name: s.display_name,
        color: agentColor(s.agent_id, s.color, i),
      })),
    [series],
  );

  // Pivot per-agent point arrays into one row-per-date frame keyed by agent_id.
  const rows = useMemo<Row[]>(() => {
    const byDate = new Map<string, Row>();
    for (const s of series) {
      for (const p of s.points) {
        let row = byDate.get(p.date);
        if (!row) {
          row = { date: p.date };
          byDate.set(p.date, row);
        }
        row[s.agent_id] = p.cum_return != null ? p.cum_return * 100 : null;
      }
    }
    return Array.from(byDate.values()).sort((a, b) =>
      a.date < b.date ? -1 : a.date > b.date ? 1 : 0,
    );
  }, [series]);

  function toggle(id: string) {
    setHidden((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  const hasData = rows.length > 0 && meta.length > 0;

  return (
    <Panel
      id="equity"
      title="Equity curves"
      subtitle="Cumulative return since inception, one line per agent"
      accent="#76B900"
      right={
        hasData ? (
          <div className="legend">
            {meta.map((m) => (
              <button
                key={m.id}
                className={`legend-item ${hidden.has(m.id) ? "legend-off" : ""}`}
                onClick={() => toggle(m.id)}
                title="Toggle series"
              >
                <span className="legend-swatch" style={{ background: m.color }} />
                {m.name}
              </button>
            ))}
          </div>
        ) : undefined
      }
    >
      {loading && !data ? (
        <Loading label="Loading equity curves" />
      ) : error && !data ? (
        <ErrorState message={error} />
      ) : !hasData ? (
        <EmptyState
          title="No NAV history yet"
          hint="Run the fleet — each rebalance appends to every agent's equity curve."
          icon="∿"
        />
      ) : (
        <div className="chart-wrap">
          <ResponsiveContainer width="100%" height={360}>
            <LineChart data={rows} margin={{ top: 8, right: 16, bottom: 4, left: 4 }}>
              <CartesianGrid stroke="#1e2732" strokeDasharray="2 4" vertical={false} />
              <XAxis
                dataKey="date"
                tick={{ fill: "#8b98a5", fontSize: 11 }}
                tickFormatter={(d: string) => fmtDate(d)}
                minTickGap={48}
                stroke="#2a3441"
              />
              <YAxis
                tick={{ fill: "#8b98a5", fontSize: 11 }}
                tickFormatter={(v: number) => `${v.toFixed(0)}%`}
                stroke="#2a3441"
                width={48}
              />
              <ReferenceLine y={0} stroke="#3a4553" strokeWidth={1} />
              <Tooltip
                content={<EquityTooltip meta={meta} hidden={hidden} />}
                cursor={{ stroke: "#3a4553", strokeWidth: 1 }}
              />
              {meta.map((m) =>
                hidden.has(m.id) ? null : (
                  <Line
                    key={m.id}
                    type="monotone"
                    dataKey={m.id}
                    name={m.name}
                    stroke={m.color}
                    strokeWidth={2}
                    dot={false}
                    activeDot={{ r: 3, strokeWidth: 0 }}
                    connectNulls
                    isAnimationActive={false}
                  />
                ),
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </Panel>
  );
}

function EquityTooltip({
  active,
  payload,
  label,
  meta,
  hidden,
}: {
  active?: boolean;
  payload?: Array<{ dataKey: string; value: number }>;
  label?: string;
  meta: Array<{ id: string; name: string; color: string }>;
  hidden: Set<string>;
}) {
  if (!active || !payload || payload.length === 0) return null;
  const rows = payload
    .filter((p) => !hidden.has(p.dataKey))
    .sort((a, b) => (b.value ?? -Infinity) - (a.value ?? -Infinity));
  return (
    <div className="chart-tooltip">
      <div className="tt-date">{fmtDate(label)}</div>
      {rows.map((p) => {
        const m = meta.find((x) => x.id === p.dataKey);
        if (!m) return null;
        return (
          <div key={p.dataKey} className="tt-row">
            <span className="tt-swatch" style={{ background: m.color }} />
            <span className="tt-name">{m.name}</span>
            <span
              className={`tt-val mono ${p.value >= 0 ? "pos" : "neg"}`}
            >
              {p.value != null ? signedPct(p.value / 100) : "—"}
            </span>
          </div>
        );
      })}
    </div>
  );
}
