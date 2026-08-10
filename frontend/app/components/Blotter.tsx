"use client";

import { useState } from "react";
import { api, Fill } from "../lib/api";
import { useApi } from "../lib/useApi";
import { Panel, Loading, ErrorState, EmptyState } from "./ui";
import { useAgents } from "../lib/AgentsContext";
import AgentSelect from "./AgentSelect";
import { agentColor } from "../lib/theme";
import { qty, money, num, fmtTime, fmtDate } from "../lib/format";

export default function Blotter() {
  const [agent, setAgent] = useState("all");
  const { agents } = useAgents();
  const { data, error, loading, refreshing } = useApi<Fill[]>(
    (signal) => api.blotter(agent, 100, signal),
    [agent],
    12000,
  );

  const nameOf = (id: string) =>
    agents.find((a) => a.agent_id === id)?.display_name ?? id;

  return (
    <Panel
      id="blotter"
      title="Trade blotter"
      subtitle="Most recent fills — every commission, slippage & venue tracked"
      accent="#76B900"
      right={
        <div className="panel-controls">
          {refreshing && <span className="live-pill">● live</span>}
          <AgentSelect value={agent} onChange={setAgent} allowAll />
        </div>
      }
    >
      {loading && !data ? (
        <Loading label="Loading fills" />
      ) : error && !data ? (
        <ErrorState message={error} />
      ) : !data || data.length === 0 ? (
        <EmptyState
          title="No trades yet — run the fleet"
          hint="Fills stream here as agents translate target weights into executions."
          icon="⇄"
        />
      ) : (
        <div className="table-scroll blotter-scroll">
          <table className="data-table blotter mono">
            <thead>
              <tr>
                <th className="ta-left">Time</th>
                <th className="ta-left">Agent</th>
                <th className="ta-left">Ticker</th>
                <th className="ta-center">Side</th>
                <th className="ta-right">Qty</th>
                <th className="ta-right">Fill px</th>
                <th className="ta-right">Notional</th>
                <th className="ta-right">Comm</th>
                <th className="ta-right">Slip bps</th>
                <th className="ta-left">Venue</th>
              </tr>
            </thead>
            <tbody>
              {data.map((f, i) => {
                const buy = (f.side ?? "").toUpperCase() === "BUY";
                const color = agentColor(f.agent_id, undefined, i);
                return (
                  <tr key={`${f.run_id}-${f.ticker}-${i}`}>
                    <td className="ta-left dim" title={fmtDate(f.executed_at)}>
                      {fmtTime(f.executed_at)}
                    </td>
                    <td className="ta-left">
                      <span className="cell-agent">
                        <span
                          className="mini-swatch"
                          style={{ background: color }}
                        />
                        {nameOf(f.agent_id)}
                      </span>
                    </td>
                    <td className="ta-left ticker">{f.ticker}</td>
                    <td className="ta-center">
                      <span className={`side ${buy ? "side-buy" : "side-sell"}`}>
                        {buy ? "BUY" : "SELL"}
                      </span>
                    </td>
                    <td className="ta-right">{qty(Math.abs(f.fill_qty))}</td>
                    <td className="ta-right">{money(f.fill_price)}</td>
                    <td className="ta-right">{money(Math.abs(f.notional))}</td>
                    <td className="ta-right dim">{money(f.commission)}</td>
                    <td className="ta-right dim">{num(f.slippage_bps, 1)}</td>
                    <td className="ta-left dim">{f.venue}</td>
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
