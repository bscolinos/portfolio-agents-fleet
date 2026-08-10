// Brand + design tokens for the NVIDIA x SingleStore "portfolio agents" terminal.
// Mirrors staging/portfolio-agents-prep/brand/brand.json fallbacks.

export const brand = {
  nvidiaGreen: "#76B900",
  bg: "#0A0E14",
  panel: "#121820",
  text: "#E6EDF3",
} as const;

// Deterministic per-agent colors. The API also returns `color` per agent; we
// prefer that, and fall back to this map by agent_id / strategy_type.
export const AGENT_COLORS: Record<string, string> = {
  "max-sharpe": "#76B900",
  "min-cvar": "#1f77b4",
  "risk-parity": "#9467bd",
  "max-return": "#d62728",
  "equal-weight": "#7f7f7f",
};

// Fallback color palette for any unmapped agents (colorblind-aware order).
const FALLBACK_PALETTE = [
  "#76B900",
  "#1f77b4",
  "#9467bd",
  "#d62728",
  "#e6a817",
  "#17becf",
  "#7f7f7f",
];

export function agentColor(
  agentId: string,
  apiColor?: string | null,
  index = 0,
): string {
  if (apiColor && /^#?[0-9a-fA-F]{3,8}$/.test(apiColor)) {
    return apiColor.startsWith("#") ? apiColor : `#${apiColor}`;
  }
  return AGENT_COLORS[agentId] ?? FALLBACK_PALETTE[index % FALLBACK_PALETTE.length];
}

export const KIND_META: Record<
  string,
  { label: string; color: string; bg: string }
> = {
  observation: { label: "OBSERVATION", color: "#58a6ff", bg: "rgba(88,166,255,0.14)" },
  decision: { label: "DECISION", color: "#76B900", bg: "rgba(118,185,0,0.16)" },
  reflection: { label: "REFLECTION", color: "#d29922", bg: "rgba(210,153,34,0.16)" },
  learning: { label: "LEARNING", color: "#bc8cff", bg: "rgba(188,140,255,0.16)" },
};
