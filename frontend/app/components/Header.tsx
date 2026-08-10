"use client";

import { useState } from "react";
import { api, Stats } from "../lib/api";
import { useApi } from "../lib/useApi";
import { CountUp } from "./ui";
import { compact, moneyCompact, ms, fmtDate } from "../lib/format";
import { brand } from "../lib/theme";

// Graceful <img> that hides itself if the asset 404s.
function Logo({ src, alt, height }: { src: string; alt: string; height: number }) {
  const [ok, setOk] = useState(true);
  if (!ok) return <span className="logo-fallback">{alt}</span>;
  // eslint-disable-next-line @next/next/no-img-element
  return (
    <img
      src={src}
      alt={alt}
      height={height}
      style={{ height, width: "auto" }}
      onError={() => setOk(false)}
    />
  );
}

interface Tile {
  key: keyof Stats | "solves" | "engineSplit";
  label: string;
  render: (s: Stats) => React.ReactNode;
  sub?: (s: Stats) => string;
  hero?: boolean;
}

const TILES: Tile[] = [
  {
    key: "n_agents",
    label: "Strategy agents",
    render: (s) => <CountUp value={s.n_agents} format={(n) => compact(n)} />,
    sub: () => "competing on the S&P 500",
  },
  {
    key: "n_trades",
    label: "Fills executed",
    render: (s) => <CountUp value={s.n_trades} format={(n) => compact(n)} />,
    sub: (s) => `across ${compact(s.universe_size)} names`,
  },
  {
    key: "total_notional",
    label: "Notional traded",
    render: (s) => (
      <CountUp value={s.total_notional} format={(n) => moneyCompact(n)} />
    ),
    sub: () => "orders → fills → positions",
  },
  {
    key: "solves",
    label: "GPU-accelerated solves",
    hero: true,
    render: (s) => <CountUp value={s.gpu_solves} format={(n) => compact(n)} />,
    sub: (s) =>
      `${compact(s.gpu_solves)} on NVIDIA L4 · ${compact(s.cpu_solves)} CPU`,
  },
  {
    key: "avg_solve_ms",
    label: "Avg optimizer solve",
    render: (s) => <CountUp value={s.avg_solve_ms} format={(n) => ms(n)} />,
    sub: () => "cuOpt / cuML wall time",
  },
  {
    key: "total_memories",
    label: "Persisted memories",
    hero: true,
    render: (s) => <CountUp value={s.total_memories} format={(n) => compact(n)} />,
    sub: () => "vector-recalled each run",
  },
  {
    key: "total_audit_events",
    label: "Audit events",
    render: (s) => (
      <CountUp value={s.total_audit_events} format={(n) => compact(n)} />
    ),
    sub: () => "immutable compliance trail",
  },
];

export default function Header() {
  const { data, error, loading, refreshing, reload, lastUpdated } = useApi<Stats>(
    (signal) => api.stats(signal),
    [],
    15000,
  );

  const asOf = data?.as_of ? fmtDate(data.as_of) : null;

  return (
    <header className="app-header">
      <div className="brand-bar">
        <div className="brand-lockup">
          <Logo src="/nvidia-logo.svg" alt="NVIDIA" height={26} />
          <span className="brand-x">×</span>
          <Logo src="/singlestore-logo-white.svg" alt="SingleStore" height={22} />
        </div>
        <div className="brand-meta">
          {asOf && (
            <span className="asof">
              <span className="asof-dot" /> as of {asOf}
            </span>
          )}
          <button
            className="refresh-btn"
            onClick={reload}
            disabled={loading}
            title="Refresh all live tiles"
          >
            <span className={refreshing ? "spin" : ""}>↻</span> Refresh
          </button>
        </div>
      </div>

      <div className="title-block">
        <h1>
          Portfolio <span className="accent">Agents</span>
        </h1>
        <p className="tagline">
          A fleet of autonomous strategy agents competing on the S&amp;P 500 —
          each one GPU-accelerated on{" "}
          <strong style={{ color: brand.nvidiaGreen }}>NVIDIA</strong> and
          equipped with{" "}
          <strong style={{ color: brand.nvidiaGreen }}>
            truly persisted memory
          </strong>{" "}
          in SingleStore.
        </p>
      </div>

      <div className="kpi-row">
        {loading && !data ? (
          Array.from({ length: 7 }).map((_, i) => (
            <div key={i} className="kpi kpi-skeleton" />
          ))
        ) : error && !data ? (
          <div className="kpi-error">Stats unavailable — {error}</div>
        ) : data ? (
          TILES.map((t) => (
            <div key={t.key} className={`kpi ${t.hero ? "kpi-hero" : ""}`}>
              <div className="kpi-label">{t.label}</div>
              <div className="kpi-value mono">{t.render(data)}</div>
              {t.sub && <div className="kpi-sub">{t.sub(data)}</div>}
            </div>
          ))
        ) : null}
      </div>
      {lastUpdated && (
        <div className="updated-hint">
          Live · auto-refresh 15s · updated{" "}
          {new Date(lastUpdated).toLocaleTimeString("en-US", { hour12: false })}
        </div>
      )}
    </header>
  );
}
