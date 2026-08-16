"""Apply sweep_schema.sql to the portfolio_agents database. Idempotent (IF NOT EXISTS).

The parameter-sweep schema (sweep_results, sweep_runs, sweep_analysis). Mirrors
apply_schema.py / apply_risk_schema.py, but points at this package's
sweep_schema.sql (which lives next to this module, not at the demo root).

Run from the demo root:
    python -m research_fleet.research_agent.apply_sweep_schema
"""
import os, sys
from pathlib import Path

import singlestoredb as s2

# .env lives at the demo root (three levels up: research_agent -> research_fleet -> demo root)
ENV = Path(__file__).resolve().parents[2] / ".env"
for line in ENV.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

sql_path = Path(__file__).resolve().parent / "sweep_schema.sql"
# strip full-line comments first, then split on ';'
raw = sql_path.read_text()
no_comments = "\n".join(l for l in raw.splitlines() if not l.strip().startswith("--"))
statements = [s.strip() for s in no_comments.split(";") if s.strip()]


def apply() -> list[str]:
    conn = s2.connect(
        host=os.environ["SINGLESTORE_HOST"], port=int(os.environ["SINGLESTORE_PORT"]),
        user=os.environ["SINGLESTORE_USER"], password=os.environ["SINGLESTORE_PASSWORD"],
        database=os.environ.get("SINGLESTORE_DATABASE", "portfolio_agents"),
    )
    cur = conn.cursor()
    ok = 0
    for stmt in statements:
        try:
            cur.execute(stmt)
            ok += 1
        except Exception as e:
            print("ERR on statement:\n", stmt[:300], "\n ->", e)
            cur.close(); conn.close()
            sys.exit(1)
    print(f"applied {ok} statements")
    cur.execute("SHOW TABLES")
    tables = [r[0] for r in cur.fetchall()]
    sweep_tables = [t for t in tables if t in ("sweep_results", "sweep_runs", "sweep_analysis")]
    print("sweep tables:", sweep_tables)
    cur.close(); conn.close()
    return sweep_tables


if __name__ == "__main__":
    apply()
