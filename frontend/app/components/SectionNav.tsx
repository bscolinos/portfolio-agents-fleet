"use client";

const SECTIONS = [
  { id: "leaderboard", label: "Leaderboard" },
  { id: "equity", label: "Equity" },
  { id: "memory", label: "Memory + Recall" },
  { id: "runs", label: "GPU Runs" },
  { id: "blotter", label: "Blotter" },
  { id: "positions", label: "Positions" },
  { id: "audit", label: "Audit" },
];

export default function SectionNav() {
  return (
    <nav className="section-nav">
      {SECTIONS.map((s) => (
        <a key={s.id} href={`#${s.id}`} className="section-nav-link">
          {s.label}
        </a>
      ))}
    </nav>
  );
}
