"use client";

import { useEffect, useState } from "react";
import { api, PositionsResponse } from "../lib/api";
import { useApi } from "../lib/useApi";
import { Panel, Loading, ErrorState, EmptyState } from "./ui";
import { useAgents } from "../lib/AgentsContext";
import AgentSelect from "./AgentSelect";
import { agentColor } from "../lib/theme";
import {
  qty,
  money,
  moneyCompact,
  pct,
  signedPct,
  pnlClass,
  fmtDate,
} from "../lib/format";

export default function Positions() {
  const { agents } = useAgents();
  const [agent, setAgent] = useState<string>("");

  // Default to the first agent once the roster loads.
  useEffect(() => {
    if (!agent && agents.length > 0) setAgent(agents[0].agent_id);
  }, [agents, agent]);

  const { data, error, loading, refreshing } = useApi<PositionsResponse>(
    (signal) => api.positions(agent, signal),
    [agent],
    agent ? 15000 : 0,
  );

  const color = agent ? agentColor(agent) : "#76B900";
  const positions = data?.positions ?? [];
  const totalUnreal = positions.reduce((s, p) => s + (p.unrealized_pnl ?? 0), 0);

  return (
    <Panel
      id="positions"
      title="Positions"
      subtitle="Current holdings, weight & unrealized P&L for the selected agent"
      accent={color}
      right={
        <div className="panel-controls">
          {refreshing && <span className="live-pill">● live</span>}
          <AgentSelect value={agent || "all"} onChange={setAgent} />
        </div>
      }
    >
      {!agent ? (
        <EmptyState title="Select an agent" icon="◈" />
      ) : loading && !data ? (
        <Loading label="Loading positions" />
      ) : error && !data ? (
        <ErrorState message={error} />
      ) : positions.length === 0 ? (
        <EmptyState
          title="No open positions"
          hint="This agent hasn't rebalanced into any names yet — run the fleet."
          icon="◈"
        />
      ) : (
        <>
          <div className="pos-summary">
            <div>
              <span className="ls-label">NAV</span>
              <span className="ls-val mono">{moneyCompact(data?.nav)}</span>
            </div>
            <div>
              <span className="ls-label">Cash</span>
              <span className="ls-val mono">{moneyCompact(data?.cash)}</span>
            </div>
            <div>
              <span className="ls-label">Holdings</span>
              <span className="ls-val mono">{positions.length}</span>
            </div>
            <div>
              <span className="ls-label">Unrealized P&L</span>
              <span className={`ls-val mono ${pnlClass(totalUnreal)}`}>
                {moneyCompact(totalUnreal)}
              </span>
            </div>
            {data?.as_of && (
              <div className="pos-asof dim">as of {fmtDate(data.as_of)}</div>
            )}
          </div>
          <div className="table-scroll">
            <table className="data-table positions mono">
              <thead>
                <tr>
                  <th className="ta-left">Ticker</th>
                  <th className="ta-right">Qty</th>
                  <th className="ta-left weight-col">Weight</th>
                  <th className="ta-right">Avg cost</th>
                  <th className="ta-right">Last</th>
                  <th className="ta-right">Mkt value</th>
                  <th className="ta-right">Unreal P&L</th>
                </tr>
              </thead>
              <tbody>
                {positions.map((p) => {
                  const w = Math.max(0, Math.min(1, p.weight ?? 0));
                  const cls = pnlClass(p.unrealized_pnl);
                  return (
                    <tr key={p.ticker}>
                      <td className="ta-left ticker">{p.ticker}</td>
                      <td className="ta-right">{qty(p.qty)}</td>
                      <td className="weight-col">
                        <span className="weight-cell">
                          <span className="weight-bar-track">
                            <span
                              className="weight-bar-fill"
                              style={{
                                width: `${w * 100}%`,
                                background: color,
                              }}
                            />
                          </span>
                          <span className="weight-num">{pct(p.weight, 1)}</span>
                        </span>
                      </td>
                      <td className="ta-right dim">{money(p.avg_cost)}</td>
                      <td className="ta-right">{money(p.last_price)}</td>
                      <td className="ta-right">{money(p.market_value)}</td>
                      <td className={`ta-right ${cls}`}>
                        {signedPct(
                          p.avg_cost && p.qty
                            ? p.unrealized_pnl / Math.abs(p.avg_cost * p.qty)
                            : 0,
                        )}
                        <span className="pnl-abs dim">
                          {" "}
                          {moneyCompact(p.unrealized_pnl)}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
    </Panel>
  );
}
