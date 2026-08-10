"use client";

import { useAgents } from "../lib/AgentsContext";
import { agentColor } from "../lib/theme";

/**
 * Shared agent dropdown. When `allowAll` is set, includes an "All agents"
 * option (value ""/"all"). Colored swatch is rendered alongside the native
 * select via a small facade so it stays keyboard-accessible.
 */
export default function AgentSelect({
  value,
  onChange,
  allowAll = false,
  allValue = "all",
  label,
}: {
  value: string;
  onChange: (v: string) => void;
  allowAll?: boolean;
  allValue?: string;
  label?: string;
}) {
  const { agents } = useAgents();
  const current = agents.find((a) => a.agent_id === value);
  const swatch =
    value === allValue || !current
      ? undefined
      : agentColor(current.agent_id, current.color);

  return (
    <label className="agent-select">
      {label && <span className="agent-select-label">{label}</span>}
      <span className="agent-select-face">
        <span
          className="agent-select-dot"
          style={{ background: swatch ?? "linear-gradient(90deg,#76B900,#1f77b4)" }}
        />
        <select value={value} onChange={(e) => onChange(e.target.value)}>
          {allowAll && <option value={allValue}>All agents</option>}
          {agents.map((a) => (
            <option key={a.agent_id} value={a.agent_id}>
              {a.display_name}
            </option>
          ))}
        </select>
        <span className="agent-select-caret" aria-hidden>
          ▾
        </span>
      </span>
    </label>
  );
}
