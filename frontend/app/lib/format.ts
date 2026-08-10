// Number / date formatting helpers. All API returns are fractions (0.1234 == 12.34%).

const nf = (min: number, max: number) =>
  new Intl.NumberFormat("en-US", {
    minimumFractionDigits: min,
    maximumFractionDigits: max,
  });

const num0 = nf(0, 0);
const num2 = nf(2, 2);

export function pct(x: number | null | undefined, digits = 2): string {
  if (x == null || Number.isNaN(x)) return "—";
  return `${(x * 100).toFixed(digits)}%`;
}

export function signedPct(x: number | null | undefined, digits = 2): string {
  if (x == null || Number.isNaN(x)) return "—";
  const v = x * 100;
  return `${v >= 0 ? "+" : ""}${v.toFixed(digits)}%`;
}

export function num(x: number | null | undefined, digits = 2): string {
  if (x == null || Number.isNaN(x)) return "—";
  return digits === 0 ? num0.format(x) : nf(digits, digits).format(x);
}

export function money(x: number | null | undefined, digits = 2): string {
  if (x == null || Number.isNaN(x)) return "—";
  return `$${(digits === 2 ? num2 : nf(digits, digits)).format(x)}`;
}

// Compact money with unit suffix, e.g. $1.24M, $980.5K, $3.1B.
export function moneyCompact(x: number | null | undefined): string {
  if (x == null || Number.isNaN(x)) return "—";
  const abs = Math.abs(x);
  const sign = x < 0 ? "-" : "";
  if (abs >= 1e9) return `${sign}$${(abs / 1e9).toFixed(2)}B`;
  if (abs >= 1e6) return `${sign}$${(abs / 1e6).toFixed(2)}M`;
  if (abs >= 1e3) return `${sign}$${(abs / 1e3).toFixed(1)}K`;
  return `${sign}$${abs.toFixed(2)}`;
}

export function compact(x: number | null | undefined): string {
  if (x == null || Number.isNaN(x)) return "—";
  const abs = Math.abs(x);
  if (abs >= 1e9) return `${(x / 1e9).toFixed(2)}B`;
  if (abs >= 1e6) return `${(x / 1e6).toFixed(2)}M`;
  if (abs >= 1e3) return `${(x / 1e3).toFixed(1)}K`;
  return num0.format(x);
}

export function qty(x: number | null | undefined): string {
  if (x == null || Number.isNaN(x)) return "—";
  return nf(0, 2).format(x);
}

// Milliseconds, e.g. 42.7 ms
export function ms(x: number | null | undefined): string {
  if (x == null || Number.isNaN(x)) return "—";
  return `${x < 10 ? x.toFixed(2) : x.toFixed(1)} ms`;
}

export function fmtDate(d: string | null | undefined): string {
  if (!d) return "—";
  const dt = new Date(d.length <= 10 ? `${d}T00:00:00` : d);
  if (Number.isNaN(dt.getTime())) return d;
  return dt.toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "2-digit",
  });
}

export function fmtTime(d: string | null | undefined): string {
  if (!d) return "—";
  const dt = new Date(d.length <= 10 ? `${d}T00:00:00` : d);
  if (Number.isNaN(dt.getTime())) return d;
  return dt.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

export function fmtDateTime(d: string | null | undefined): string {
  if (!d) return "—";
  const dt = new Date(d.length <= 10 ? `${d}T00:00:00` : d);
  if (Number.isNaN(dt.getTime())) return d;
  return `${fmtDate(d)} ${dt.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  })}`;
}

export function pnlClass(x: number | null | undefined): "pos" | "neg" | "flat" {
  if (x == null || Number.isNaN(x) || x === 0) return "flat";
  return x > 0 ? "pos" : "neg";
}
