#!/usr/bin/env python3
"""Research Fleet live console — a single-file, read-only dashboard.

Serves both a JSON API and an auto-refreshing HTML page over the *real*
research_* tables in SingleStore (the ones the 5-node fleet writes 24/7):

  research_agents      — who is alive + last heartbeat
  research_activity    — the streaming action feed (START/TOOL/END/ERROR),
                         and END rows carry the per-cycle NeMo Switchyard
                         model-tier decisions (haiku/sonnet/opus)
  research_hypotheses  — hypotheses the model formed
  research_experiments — real backtests + metrics (sharpe vs 1/N, dd, turnover)
  research_findings    — the honest natural-language conclusions

No writes. No LLM calls. Reads config from the demo's own .env.

Run:  python -m research_fleet.console            # serves :8215
      python research_fleet/console.py --port 8215
Open: http://localhost:8215
"""
from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

# Load the demo's .env (this file lives in research_fleet/, .env is one dir up).
_HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_HERE, os.pardir, ".env"))

import singlestoredb as s2  # noqa: E402


def _conn_kwargs() -> dict:
    return {
        "host": os.environ["SINGLESTORE_HOST"],
        "port": int(os.environ.get("SINGLESTORE_PORT", "3306")),
        "user": os.environ.get("SINGLESTORE_USER", "admin"),
        "password": os.environ.get("SINGLESTORE_PASSWORD", ""),
        "database": os.environ.get("SINGLESTORE_DATABASE") or "portfolio_agents",
    }


@contextmanager
def _cursor():
    conn = s2.connect(results_type="dicts", autocommit=True, **_conn_kwargs())
    try:
        cur = conn.cursor()
        try:
            yield cur
        finally:
            cur.close()
    finally:
        conn.close()


def _q(sql: str, params: tuple | None = None) -> list[dict]:
    with _cursor() as cur:
        cur.execute(sql, params)
        return list(cur.fetchall() or [])


def _loads(v: Any) -> Any:
    """Coerce a JSON column (str/bytes/None/already-parsed) to a Python value."""
    if v is None:
        return None
    if isinstance(v, (bytes, bytearray)):
        v = v.decode("utf-8", "replace")
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return v
    return v


def _iso(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.isoformat()
    return str(v)


# ----------------------------------------------------------------------------
# Data queries (read-only)
# ----------------------------------------------------------------------------
def agents() -> list[dict]:
    """Fleet roster + liveness (fresh = heartbeat within 3 min)."""
    rows = _q(
        """
        SELECT a.agent_id, a.display_name, a.focus_area, a.model,
               a.instance_id, a.private_ip, a.status, a.heartbeat_at,
               TIMESTAMPDIFF(SECOND, a.heartbeat_at, NOW()) AS beat_age_s
        FROM research_agents a
        ORDER BY a.agent_id
        """
    )
    out = []
    for r in rows:
        age = r.get("beat_age_s")
        r["heartbeat_at"] = _iso(r.get("heartbeat_at"))
        r["live"] = age is not None and age <= 180
        out.append(r)
    return out


def totals() -> dict:
    def c(t: str) -> int:
        return _q(f"SELECT COUNT(*) n FROM {t}")[0]["n"]

    last10 = _q(
        """
        SELECT COUNT(*) n FROM research_findings
        WHERE created_at > NOW() - INTERVAL 10 MINUTE
        """
    )[0]["n"]
    exp10 = _q(
        """
        SELECT COUNT(*) n FROM research_experiments
        WHERE finished_at > NOW() - INTERVAL 10 MINUTE
        """
    )[0]["n"]
    return {
        "hypotheses": c("research_hypotheses"),
        "experiments": c("research_experiments"),
        "findings": c("research_findings"),
        "activity": c("research_activity"),
        "findings_last_10m": last10,
        "experiments_last_10m": exp10,
        "server_time": datetime.now(timezone.utc).isoformat(),
    }


def activity(limit: int = 60) -> list[dict]:
    rows = _q(
        """
        SELECT ts, agent_id, phase, detail
        FROM research_activity
        ORDER BY ts DESC
        LIMIT %s
        """,
        (int(limit),),
    )
    for r in rows:
        r["ts"] = _iso(r["ts"])
        r["detail"] = _loads(r["detail"])
    return rows


def findings(limit: int = 12) -> list[dict]:
    rows = _q(
        """
        SELECT created_at, agent_id, strategy_family, kind, title,
               LEFT(content, 400) AS content, metrics
        FROM research_findings
        ORDER BY created_at DESC
        LIMIT %s
        """,
        (int(limit),),
    )
    for r in rows:
        r["created_at"] = _iso(r["created_at"])
        r["metrics"] = _loads(r["metrics"])
    return rows


def experiments(limit: int = 12) -> list[dict]:
    rows = _q(
        """
        SELECT finished_at, agent_id, strategy_family, universe,
               sharpe, benchmark_sharpe, ann_return, max_drawdown,
               turnover, status, params
        FROM research_experiments
        ORDER BY finished_at DESC
        LIMIT %s
        """,
        (int(limit),),
    )
    for r in rows:
        r["finished_at"] = _iso(r["finished_at"])
        r["params"] = _loads(r["params"])
        bs = r.get("benchmark_sharpe")
        sh = r.get("sharpe")
        r["beats_benchmark"] = (
            sh is not None and bs is not None and sh > bs
        )
    return rows


def analyst_queries(limit: int = 15) -> list[dict]:
    """Recent NL questions the fleet asked SingleStore Aura Analyst (Portal text-to-SQL).

    Every agent question routed through the real Aura Analyst domain lands in
    research_analyst_queries — the generated SQL, row count, answer, and latency.
    Degrades to [] if the table isn't present yet (fleet not wired to Aura here).
    """
    try:
        rows = _q(
            """
            SELECT query_id, agent_id, question,
                   LEFT(generated_sql, 600) AS generated_sql,
                   row_count, LEFT(answer, 400) AS answer,
                   latency_ms, status, created_at
            FROM research_analyst_queries
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (int(limit),),
        )
    except Exception:  # noqa: BLE001
        return []
    for r in rows:
        r["created_at"] = _iso(r["created_at"])
    return rows


def routing() -> dict:
    """Parse NeMo Switchyard tier decisions out of recent END activity rows.

    Each END detail carries `tiers: [...]` — the model tier chosen per turn
    of that cycle (fast=haiku / balanced=sonnet / reasoning=opus).
    """
    rows = _q(
        """
        SELECT ts, agent_id, detail
        FROM research_activity
        WHERE phase = 'END' AND ts > NOW() - INTERVAL 6 HOUR
        ORDER BY ts DESC
        LIMIT 400
        """
    )
    tally: dict[str, int] = {}
    cycles = 0
    for r in rows:
        d = _loads(r["detail"]) or {}
        tiers = d.get("tiers") if isinstance(d, dict) else None
        if not tiers:
            continue
        cycles += 1
        for t in tiers:
            key = str(t).lower()
            tally[key] = tally.get(key, 0) + 1
    total = sum(tally.values()) or 1
    ranked = sorted(tally.items(), key=lambda kv: -kv[1])
    return {
        "cycles_with_tiers": cycles,
        "turns_total": sum(tally.values()),
        "by_tier": [
            {"tier": k, "turns": v, "pct": round(100 * v / total, 1)}
            for k, v in ranked
        ],
    }


# ----------------------------------------------------------------------------
# App + routes
# ----------------------------------------------------------------------------
app = FastAPI(title="Research Fleet Console", version="1.0.0")


@app.get("/api/health")
def api_health() -> JSONResponse:
    try:
        _q("SELECT 1 AS ok")
        return JSONResponse({"ok": True})
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(e)}, status_code=502)


@app.get("/api/snapshot")
def api_snapshot() -> JSONResponse:
    """Everything the page needs in one round trip."""
    try:
        return JSONResponse(
            {
                "ok": True,
                "totals": totals(),
                "agents": agents(),
                "routing": routing(),
                "activity": activity(60),
                "findings": findings(10),
                "experiments": experiments(10),
                "analyst": analyst_queries(15),
            }
        )
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(e)}, status_code=502)


@app.get("/api/agents")
def api_agents() -> JSONResponse:
    return JSONResponse(agents())


@app.get("/api/activity")
def api_activity(limit: int = 60) -> JSONResponse:
    return JSONResponse(activity(limit))


@app.get("/api/findings")
def api_findings(limit: int = 12) -> JSONResponse:
    return JSONResponse(findings(limit))


@app.get("/api/experiments")
def api_experiments(limit: int = 12) -> JSONResponse:
    return JSONResponse(experiments(limit))


@app.get("/api/routing")
def api_routing() -> JSONResponse:
    return JSONResponse(routing())


@app.get("/api/analyst")
def api_analyst(limit: int = 15) -> JSONResponse:
    return JSONResponse(analyst_queries(limit))


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(PAGE)


# ----------------------------------------------------------------------------
# The page (self-contained; polls /api/snapshot)
# ----------------------------------------------------------------------------
PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Research Fleet — Live Console</title>
<style>
  :root{
    --bg:#0a0e17; --panel:#111726; --panel2:#0d1320; --line:#1e2942;
    --txt:#e6ecf7; --dim:#8091b0; --accent:#7c5cff; --green:#31d07a;
    --amber:#f5b942; --red:#ff5c6c; --cyan:#38bdf8;
    --haiku:#38bdf8; --sonnet:#7c5cff; --opus:#f5b942;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--txt);
    font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
  header{display:flex;align-items:center;gap:16px;padding:14px 22px;
    border-bottom:1px solid var(--line);background:linear-gradient(90deg,#0d1320,#111726)}
  header h1{font-size:16px;margin:0;font-weight:650;letter-spacing:.2px}
  header .sub{color:var(--dim);font-size:12px}
  .pill{margin-left:auto;display:flex;align-items:center;gap:8px;font-size:12px;color:var(--dim)}
  .dot{width:9px;height:9px;border-radius:50%;background:var(--green);box-shadow:0 0 8px var(--green)}
  .dot.stale{background:var(--red);box-shadow:0 0 8px var(--red)}
  .wrap{padding:18px 22px;max-width:1500px;margin:0 auto}
  .kpis{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin-bottom:16px}
  .kpi{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px 14px}
  .kpi .v{font-size:24px;font-weight:700}
  .kpi .l{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.5px}
  .kpi .d{font-size:11px;color:var(--green);margin-top:2px}
  .grid{display:grid;grid-template-columns:340px 1fr;gap:16px}
  .col{display:flex;flex-direction:column;gap:16px}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden;margin-bottom:16px}
  .card h2{font-size:12px;text-transform:uppercase;letter-spacing:.6px;color:var(--dim);
    margin:0;padding:12px 14px;border-bottom:1px solid var(--line);background:var(--panel2)}
  .card .body{padding:8px 14px 14px}
  .agent{display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid var(--line)}
  .agent:last-child{border-bottom:none}
  .agent .nm{font-weight:600}
  .agent .fa{color:var(--dim);font-size:12px}
  .agent .beat{margin-left:auto;font-size:11px;color:var(--dim)}
  .tierbar{display:flex;height:26px;border-radius:6px;overflow:hidden;margin:4px 0 10px}
  .tierbar span{display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:600;color:#0a0e17}
  .legend{display:flex;gap:14px;font-size:11px;color:var(--dim);flex-wrap:wrap}
  .legend b{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px}
  .feed{max-height:520px;overflow:auto;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
  .ev{display:flex;gap:8px;padding:5px 0;border-bottom:1px solid #182238}
  .ev .t{color:var(--dim);white-space:nowrap}
  .ev .ag{color:var(--cyan);min-width:88px}
  .ph{font-weight:700;min-width:74px}
  .ph.START{color:var(--dim)} .ph.TOOL{color:var(--accent)}
  .ph.END{color:var(--green)} .ph.ERROR{color:var(--red)}
  .ev .dt{color:var(--txt);opacity:.85}
  .find{padding:10px 0;border-bottom:1px solid var(--line)}
  .find:last-child{border-bottom:none}
  .find .ttl{font-weight:600;margin-bottom:2px}
  .find .meta{font-size:11px;color:var(--dim);margin-bottom:4px}
  .find .body{color:#c3cde0;font-size:12.5px}
  .fam{display:inline-block;padding:1px 7px;border-radius:6px;
    background:#1b2440;color:var(--cyan);font-size:11px;margin-right:6px}
  .aq{padding:10px 0;border-bottom:1px solid var(--line)}
  .aq:last-child{border-bottom:none}
  .aq .q{font-weight:600;margin-bottom:4px}
  .aq .meta{font-size:11px;color:var(--dim);margin-bottom:5px}
  .aq .sql{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11.5px;
    color:#9fe0c0;background:var(--panel2);border:1px solid var(--line);border-radius:6px;
    padding:6px 8px;white-space:pre-wrap;word-break:break-word;max-height:96px;overflow:auto}
  .st{display:inline-block;padding:1px 7px;border-radius:6px;font-size:11px;margin-left:6px}
  .st.ok{background:#12321f;color:var(--green)} .st.err{background:#331519;color:var(--red)}
  table{width:100%;border-collapse:collapse;font-size:12.5px}
  th{text-align:right;color:var(--dim);font-weight:500;font-size:11px;text-transform:uppercase;
    padding:6px 8px;border-bottom:1px solid var(--line)}
  th:first-child,td:first-child{text-align:left}
  td{padding:6px 8px;border-bottom:1px solid #182238}
  .win{color:var(--green)} .lose{color:var(--dim)}
  .num{font-family:ui-monospace,Menlo,monospace}
  a{color:var(--accent)}
  .err{color:var(--red);padding:12px}
</style>
</head>
<body>
<header>
  <h1>Research Fleet <span style="color:var(--accent)">·</span> Live Console</h1>
  <span class="sub">autonomous strategy agents → SingleStore · NeMo Switchyard routing</span>
  <span class="pill"><span id="livedot" class="dot"></span><span id="livetxt">connecting…</span></span>
</header>
<div class="wrap">
  <div id="err"></div>
  <div class="kpis" id="kpis"></div>
  <div class="grid">
    <div class="col">
      <div class="card">
        <h2>Fleet</h2>
        <div class="body" id="agents"></div>
      </div>
      <div class="card">
        <h2>Model routing — NeMo Switchyard (last 6h)</h2>
        <div class="body" id="routing"></div>
      </div>
    </div>
    <div class="col">
      <div class="card">
        <h2>Aura Analyst — NL→SQL over SingleStore</h2>
        <div class="body" id="analyst"></div>
      </div>
      <div class="card">
        <h2>Live activity feed</h2>
        <div class="body feed" id="feed"></div>
      </div>
      <div class="card">
        <h2>Latest experiments (real backtests vs 1/N)</h2>
        <div class="body" id="exp"></div>
      </div>
      <div class="card">
        <h2>Latest findings</h2>
        <div class="body" id="findings"></div>
      </div>
    </div>
  </div>
</div>
<script>
const TIER_COLORS={haiku:"#38bdf8","haiku-fast":"#38bdf8","haiku-classifier":"#38bdf8",
  fast:"#38bdf8",sonnet:"#7c5cff",balanced:"#7c5cff",opus:"#f5b942",reasoning:"#f5b942"};
function esc(s){return (s==null?"":String(s)).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));}
function num(v,d=2){return (v==null||isNaN(v))?"—":Number(v).toFixed(d);}
function ago(iso){if(!iso)return "—";const s=(Date.now()-new Date(iso+ (iso.endsWith("Z")||iso.includes("+")?"":"Z")).getTime())/1000;
  if(s<60)return Math.max(0,Math.round(s))+"s ago";if(s<3600)return Math.round(s/60)+"m ago";return Math.round(s/3600)+"h ago";}
function tclr(t){return TIER_COLORS[String(t).toLowerCase()]||"#8091b0";}

function kpi(v,l,d){return `<div class="kpi"><div class="v">${v}</div><div class="l">${l}</div>${d?`<div class="d">${d}</div>`:""}</div>`;}

function render(s){
  document.getElementById("err").innerHTML="";
  const t=s.totals||{};
  document.getElementById("kpis").innerHTML=[
    kpi((s.agents||[]).filter(a=>a.live).length+"/"+(s.agents||[]).length,"agents live"),
    kpi(t.hypotheses??"—","hypotheses"),
    kpi(t.experiments??"—","experiments",t.experiments_last_10m?("+"+t.experiments_last_10m+" / 10m"):""),
    kpi(t.findings??"—","findings",t.findings_last_10m?("+"+t.findings_last_10m+" / 10m"):""),
    kpi(t.activity??"—","activity events"),
    kpi((s.routing&&s.routing.cycles_with_tiers)||0,"routed cycles / 6h"),
  ].join("");

  // agents
  document.getElementById("agents").innerHTML=(s.agents||[]).map(a=>`
    <div class="agent">
      <span class="dot ${a.live?"":"stale"}"></span>
      <div><div class="nm">${esc(a.display_name||a.agent_id)}</div>
        <div class="fa">${esc(a.focus_area||"")}${a.model?" · "+esc(a.model):""}</div></div>
      <span class="beat">${a.live?"live":"stale"} · ${ago(a.heartbeat_at)}</span>
    </div>`).join("")||'<div class="fa">no agents</div>';

  // routing
  const r=s.routing||{by_tier:[]};
  const bar=(r.by_tier||[]).map(x=>`<span style="flex:${x.turns};background:${tclr(x.tier)}"
     title="${esc(x.tier)} ${x.pct}%">${x.pct>=8?esc(x.tier)+" "+x.pct+"%":""}</span>`).join("");
  const leg=(r.by_tier||[]).map(x=>`<span><b style="background:${tclr(x.tier)}"></b>${esc(x.tier)} · ${x.turns} turns</span>`).join("");
  document.getElementById("routing").innerHTML=
    (r.turns_total? `<div class="tierbar">${bar}</div><div class="legend">${leg}</div>
      <div class="fa" style="margin-top:8px">${r.turns_total} model calls across ${r.cycles_with_tiers} cycles — the classifier grades each cycle and routes fast→Haiku / balanced→Sonnet / reasoning→Opus.</div>`
     : '<div class="fa">no routed cycles yet (agents may be on the direct-Bedrock fallback transport)</div>');

  // feed
  document.getElementById("feed").innerHTML=(s.activity||[]).map(e=>{
    let d="";const x=e.detail;
    if(x&&typeof x==="object"){
      if(x.tool)d=`${x.tool}${x.ok===false?" ✗":""}${x.id?" → "+x.id:""}`;
      else if(x.tiers)d=`${x.steps} steps · tiers: ${x.tiers.join(", ")}${x.error?" · ERR "+x.error:""}`;
      else if(x.focus)d=`focus: ${x.focus}`;
      else d=JSON.stringify(x).slice(0,120);
    } else if(x){d=String(x).slice(0,120);}
    return `<div class="ev"><span class="t">${(e.ts||"").slice(11,19)}</span>
      <span class="ag">${esc(e.agent_id)}</span>
      <span class="ph ${esc(e.phase)}">${esc(e.phase)}</span>
      <span class="dt">${esc(d)}</span></div>`;
  }).join("")||'<div class="fa">no activity</div>';

  // experiments
  document.getElementById("exp").innerHTML=`<table><thead><tr>
    <th>agent</th><th>family</th><th>Sharpe</th><th>1/N</th><th>ann.ret</th>
    <th>maxDD</th><th>turnover</th><th>vs 1/N</th></tr></thead><tbody>`+
    (s.experiments||[]).map(e=>`<tr>
      <td>${esc(e.agent_id)}</td><td><span class="fam">${esc(e.strategy_family||"")}</span></td>
      <td class="num">${num(e.sharpe)}</td><td class="num">${num(e.benchmark_sharpe)}</td>
      <td class="num">${e.ann_return!=null?(100*e.ann_return).toFixed(1)+"%":"—"}</td>
      <td class="num">${e.max_drawdown!=null?(100*e.max_drawdown).toFixed(1)+"%":"—"}</td>
      <td class="num">${num(e.turnover,3)}</td>
      <td class="${e.beats_benchmark?"win":"lose"}">${e.beats_benchmark?"✓ beats":"—"}</td></tr>`).join("")+
    `</tbody></table>`;

  // findings
  document.getElementById("findings").innerHTML=(s.findings||[]).map(f=>`
    <div class="find">
      <div class="ttl">${esc(f.title||"(untitled)")}</div>
      <div class="meta"><span class="fam">${esc(f.strategy_family||f.kind||"")}</span>
        ${esc(f.agent_id)} · ${ago(f.created_at)}</div>
      <div class="body">${esc((f.content||"").replace(/\*\*/g,""))}</div>
    </div>`).join("")||'<div class="fa">no findings</div>';

  // analyst (Aura Analyst NL→SQL)
  document.getElementById("analyst").innerHTML=(s.analyst||[]).map(a=>{
    const bad=String(a.status||"").toLowerCase()!=="ok"&&String(a.status||"").toLowerCase()!=="success";
    return `<div class="aq">
      <div class="q">${esc(a.question||"(no question)")}
        <span class="st ${bad?"err":"ok"}">${esc(a.status||"—")}</span></div>
      <div class="meta"><span class="fam">Aura Analyst</span>${esc(a.agent_id)} ·
        ${a.row_count!=null?esc(a.row_count)+" rows":"—"} ·
        ${a.latency_ms!=null?Math.round(a.latency_ms)+"ms":"—"} · ${ago(a.created_at)}</div>
      ${a.generated_sql?`<div class="sql">${esc(a.generated_sql)}</div>`:""}
    </div>`;
  }).join("")||'<div class="fa">no Aura Analyst queries yet</div>';
}

let lastOk=0;
async function tick(){
  try{
    const res=await fetch("/api/snapshot",{cache:"no-store"});
    const s=await res.json();
    if(!s.ok){throw new Error(s.error||"backend error");}
    render(s); lastOk=Date.now();
    document.getElementById("livedot").className="dot";
    document.getElementById("livetxt").textContent="live · refreshed "+new Date().toLocaleTimeString();
  }catch(e){
    document.getElementById("livedot").className="dot stale";
    document.getElementById("livetxt").textContent="reconnecting…";
    if(Date.now()-lastOk>15000)
      document.getElementById("err").innerHTML=`<div class="err">⚠ ${esc(e.message)}</div>`;
  }
}
tick(); setInterval(tick,4000);
</script>
</body>
</html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Research Fleet live console")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8215)
    args = ap.parse_args()
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
