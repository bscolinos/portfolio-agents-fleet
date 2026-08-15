"""Apply risk_schema.sql to the portfolio_agents database. Idempotent (IF NOT EXISTS).

The safety-layer schema (risk_decisions, kill_switches, paper_orders,
paper_positions, paper_nav_history). Mirrors apply_schema.py exactly, but points
at risk_schema.sql instead of schema.sql.
"""
import os, sys, singlestoredb as s2
from pathlib import Path

ENV = Path(__file__).resolve().parent / ".env"
for line in ENV.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("="); os.environ.setdefault(k.strip(), v.strip())

sql_path = Path(__file__).resolve().parent / "risk_schema.sql"
# strip full-line comments first, then split on ';'
raw = sql_path.read_text()
no_comments = "\n".join(l for l in raw.splitlines() if not l.strip().startswith("--"))
statements = [s.strip() for s in no_comments.split(";") if s.strip()]

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
        sys.exit(1)
print(f"applied {ok} statements")
cur.execute("SHOW TABLES")
tables = [r[0] for r in cur.fetchall()]
risk_tables = [t for t in tables if t in (
    "risk_decisions", "kill_switches", "paper_orders",
    "paper_positions", "paper_nav_history")]
print("risk tables:", risk_tables)
cur.close(); conn.close()
