// Typed client for the Portfolio Agents FastAPI backend (base path /api).
// See demos/portfolio-agents/API_CONTRACT.md for the source of truth.

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") ||
  "http://localhost:8210";

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

export interface Health {
  ok: boolean;
  db_version: string;
  db: string;
}

export interface Stats {
  n_agents: number;
  n_trades: number;
  total_notional: number;
  gpu_solves: number;
  cpu_solves: number;
  avg_solve_ms: number;
  total_memories: number;
  total_audit_events: number;
  universe_size: number;
  as_of: string | null;
}

export interface Agent {
  agent_id: string;
  display_name: string;
  strategy_type: string;
  objective: string;
  engine: string;
  color: string;
  latest_nav: number | null;
  cum_return: number | null;
  daily_return: number | null;
  sharpe: number | null;
  n_positions: number;
  last_run_at: string | null;
  last_engine: string | null;
  last_gpu_name: string | null;
  avg_solve_ms: number | null;
}

export interface LeaderboardRow extends Agent {
  rank: number;
  turnover: number | null;
  max_drawdown: number | null;
  vol: number | null;
}

export interface NavPoint {
  date: string;
  nav: number;
  cum_return: number | null;
  daily_return: number | null;
}

export interface NavSeries {
  agent_id: string;
  display_name: string;
  color: string;
  points: NavPoint[];
}

export interface NavResponse {
  series: NavSeries[];
}

export interface Position {
  ticker: string;
  qty: number;
  avg_cost: number;
  last_price: number;
  market_value: number;
  weight: number;
  unrealized_pnl: number;
}

export interface PositionsResponse {
  agent_id: string;
  as_of: string | null;
  cash: number;
  nav: number;
  positions: Position[];
}

export interface Fill {
  executed_at: string;
  agent_id: string;
  ticker: string;
  side: "BUY" | "SELL" | string;
  fill_qty: number;
  fill_price: number;
  notional: number;
  commission: number;
  slippage_bps: number;
  venue: string;
  run_id: string;
}

export interface Run {
  run_id: string;
  agent_id: string;
  as_of_date: string;
  engine: string;
  gpu_name: string | null;
  num_scenarios: number | null;
  solve_ms: number | null;
  scenario_ms: number | null;
  universe_size: number | null;
  status: string;
  started_at: string;
  finished_at: string | null;
}

export interface Memory {
  memory_id: string;
  kind: string;
  as_of_date: string | null;
  content: string;
  importance: number;
  metrics: Record<string, unknown> | null;
  tags: string[] | Record<string, unknown> | null;
  created_at: string;
}

export interface RecallResult {
  content: string;
  kind: string;
  score: number;
  created_at: string;
  importance: number;
}

export interface RecallResponse {
  query: string;
  agent_id: string;
  results: RecallResult[];
}

export interface RiskPoint {
  as_of_date: string;
  exp_return: number | null;
  volatility: number | null;
  sharpe: number | null;
  cvar: number | null;
  turnover: number | null;
  n_positions: number | null;
}

export interface AuditEvent {
  ts: string;
  agent_id: string;
  run_id: string | null;
  event_type: string;
  entity_ref: string | null;
  ticker: string | null;
  detail: Record<string, unknown> | null;
  actor: string;
}

// ---------------------------------------------------------------------------
// Fetch layer
// ---------------------------------------------------------------------------

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}

async function get<T>(path: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    signal,
    headers: { Accept: "application/json" },
    cache: "no-store",
  });
  if (!res.ok) {
    let body = "";
    try {
      body = await res.text();
    } catch {
      /* ignore */
    }
    throw new ApiError(
      `${res.status} ${res.statusText} — ${path}${body ? `: ${body.slice(0, 200)}` : ""}`,
      res.status,
    );
  }
  return (await res.json()) as T;
}

function qs(params: Record<string, string | number | undefined | null>): string {
  const parts = Object.entries(params)
    .filter(([, v]) => v !== undefined && v !== null && v !== "")
    .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`);
  return parts.length ? `?${parts.join("&")}` : "";
}

export const api = {
  health: (signal?: AbortSignal) => get<Health>("/api/health", signal),
  stats: (signal?: AbortSignal) => get<Stats>("/api/stats", signal),
  agents: (signal?: AbortSignal) => get<Agent[]>("/api/agents", signal),
  leaderboard: (signal?: AbortSignal) =>
    get<LeaderboardRow[]>("/api/leaderboard", signal),
  nav: (agent = "all", signal?: AbortSignal) =>
    get<NavResponse>(`/api/nav${qs({ agent })}`, signal),
  positions: (agent: string, signal?: AbortSignal) =>
    get<PositionsResponse>(`/api/positions${qs({ agent })}`, signal),
  blotter: (agent = "all", limit = 100, signal?: AbortSignal) =>
    get<Fill[]>(`/api/blotter${qs({ agent, limit })}`, signal),
  runs: (agent = "all", limit = 50, signal?: AbortSignal) =>
    get<Run[]>(`/api/runs${qs({ agent, limit })}`, signal),
  memory: (agent: string, kind = "", limit = 50, signal?: AbortSignal) =>
    get<Memory[]>(`/api/memory${qs({ agent, kind, limit })}`, signal),
  recall: (agent: string, q: string, k = 5, signal?: AbortSignal) =>
    get<RecallResponse>(`/api/memory/recall${qs({ agent, q, k })}`, signal),
  risk: (agent: string, limit = 100, signal?: AbortSignal) =>
    get<RiskPoint[]>(`/api/risk${qs({ agent, limit })}`, signal),
  audit: (
    opts: { run_id?: string; agent?: string; limit?: number } = {},
    signal?: AbortSignal,
  ) =>
    get<AuditEvent[]>(
      `/api/audit${qs({ run_id: opts.run_id, agent: opts.agent, limit: opts.limit ?? 100 })}`,
      signal,
    ),
};
