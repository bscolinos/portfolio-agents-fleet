"use client";

import { useEffect, useRef, useState } from "react";

// ---------------------------------------------------------------------------
// Panel / section shell
// ---------------------------------------------------------------------------

export function Panel({
  title,
  subtitle,
  accent,
  right,
  children,
  className,
  id,
}: {
  title?: React.ReactNode;
  subtitle?: React.ReactNode;
  accent?: string;
  right?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  id?: string;
}) {
  return (
    <section
      id={id}
      className={`panel ${className ?? ""}`}
      style={accent ? { ["--accent" as string]: accent } : undefined}
    >
      {(title || right) && (
        <header className="panel-head">
          <div className="panel-head-titles">
            {title && <h2 className="panel-title">{title}</h2>}
            {subtitle && <p className="panel-sub">{subtitle}</p>}
          </div>
          {right && <div className="panel-head-right">{right}</div>}
        </header>
      )}
      {children}
    </section>
  );
}

// ---------------------------------------------------------------------------
// Loading / error / empty states
// ---------------------------------------------------------------------------

export function Loading({ label = "Loading" }: { label?: string }) {
  return (
    <div className="state state-loading">
      <span className="spinner" aria-hidden />
      <span>{label}…</span>
    </div>
  );
}

export function ErrorState({ message }: { message: string }) {
  return (
    <div className="state state-error">
      <strong>Backend unavailable.</strong>
      <span className="mono">{message}</span>
      <span className="muted">
        Start it: <code>cd backend && uvicorn main:app --port 8210</code>
      </span>
    </div>
  );
}

export function EmptyState({
  title = "Nothing here yet",
  hint,
  icon = "◇",
}: {
  title?: string;
  hint?: string;
  icon?: string;
}) {
  return (
    <div className="state state-empty">
      <span className="empty-icon" aria-hidden>
        {icon}
      </span>
      <strong>{title}</strong>
      {hint && <span className="muted">{hint}</span>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Badges
// ---------------------------------------------------------------------------

export function EngineBadge({
  engine,
  gpuName,
}: {
  engine?: string | null;
  gpuName?: string | null;
}) {
  const isGpu = (engine ?? "").toLowerCase() === "gpu";
  return (
    <span className={`badge ${isGpu ? "badge-gpu" : "badge-cpu"}`} title={engine ?? ""}>
      {isGpu ? (
        <>
          <span className="badge-dot" /> GPU · {gpuName || "NVIDIA L4"}
        </>
      ) : (
        <>CPU · cvxpy</>
      )}
    </span>
  );
}

export function Chip({
  children,
  color,
  onClick,
  active,
  title,
}: {
  children: React.ReactNode;
  color?: string;
  onClick?: () => void;
  active?: boolean;
  title?: string;
}) {
  const Tag = onClick ? "button" : "span";
  return (
    <Tag
      className={`chip ${active ? "chip-active" : ""} ${onClick ? "chip-btn" : ""}`}
      onClick={onClick}
      title={title}
      style={color ? { ["--chip" as string]: color } : undefined}
    >
      {color && <span className="chip-dot" style={{ background: color }} />}
      {children}
    </Tag>
  );
}

// ---------------------------------------------------------------------------
// CountUp — animated number for KPI tiles
// ---------------------------------------------------------------------------

export function CountUp({
  value,
  format,
  duration = 900,
}: {
  value: number;
  format: (n: number) => string;
  duration?: number;
}) {
  const [display, setDisplay] = useState(value);
  const fromRef = useRef(value);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    const from = fromRef.current;
    const to = value;
    if (from === to) {
      setDisplay(to);
      return;
    }
    const start = performance.now();
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3); // easeOutCubic
      setDisplay(from + (to - from) * eased);
      if (t < 1) {
        rafRef.current = requestAnimationFrame(tick);
      } else {
        fromRef.current = to;
      }
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      fromRef.current = to;
    };
  }, [value, duration]);

  return <>{format(display)}</>;
}
