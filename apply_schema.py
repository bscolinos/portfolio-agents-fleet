"""Apply schema.sql to the portfolio_agents database. Idempotent (IF NOT EXISTS)."""
import os, sys, singlestoredb as s2
from pathlib import Path

ENV = Path(__file__).resolve().parent.parent.parent / "demos" / "portfolio-agents" / ".env"
for line in ENV.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("="); os.environ.setdefault(k.strip(), v.strip())

sql_path = Path(__file__).resolve().parent / "schema.sql"
# strip full-line comments first, then split on ';'
raw = sql_path.read_text()
no_comments = "\n".join(l for l in raw.splitlines() if not l.strip().startswith("--"))
statements = [s.strip() for s in no_comments.split(";") if s.strip()]

conn = s2.connect(
    host=os.environ["SINGLESTORE_HOST"], port=int(os.environ["SINGLESTORE_PORT"]),
    user=os.environ["SINGLESTORE_USER"], password=os.environ["SINGLESTORE_PASSWORD"],
    database="portfolio_agents",
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
print("tables:", tables)
cur.close(); conn.close()
