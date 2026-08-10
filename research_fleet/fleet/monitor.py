"""Hour-long fleet monitor: snapshots health every 5 min, keeps the research
queue fed so agents stay busy, and logs deltas. Self-terminates after ~1 hour.

Run detached; tail the log file it prints.
"""
import sys, time, uuid, json
sys.path.insert(0, "/Users/billscolinos/Documents/code_factory/staging/portfolio-agents-prep")
from research_agent import research_db as rdb

VALID = "('equal_weight','momentum','mean_reversion','vol_target','low_vol','factor','risk_parity','regime')"
FAMS = ["momentum", "mean_reversion", "vol_target", "factor", "risk_parity", "regime"]
INTERVAL = 300        # 5 min
DURATION = 3600 + 60  # ~1 hour
MIN_PENDING = 8       # keep at least this many tasks queued


def snap():
    ag = rdb.query("SELECT agent_id,status,TIMESTAMPDIFF(SECOND,heartbeat_at,NOW()) age FROM research_agents ORDER BY agent_id")
    tasks = {r["status"]: r["c"] for r in rdb.query("SELECT status,COUNT(*) c FROM research_tasks GROUP BY status")}
    tot = rdb.query("SELECT (SELECT COUNT(*) FROM research_findings) f,(SELECT COUNT(*) FROM research_experiments) e,(SELECT COUNT(*) FROM research_activity) a")[0]
    sig = rdb.query("SELECT SUM(beats_benchmark) b, COUNT(*) c, ROUND(AVG(sharpe),3) avg, ROUND(MAX(sharpe),3) mx FROM research_experiments WHERE sharpe IS NOT NULL")[0]
    err = rdb.query("SELECT COUNT(*) c FROM research_activity WHERE phase='ERROR'")[0]["c"]
    uni = rdb.query(f"SELECT COUNT(*) c FROM research_findings WHERE embedding IS NULL OR strategy_family NOT IN {VALID}")[0]["c"]
    return {"agents": ag, "tasks": tasks, "tot": tot, "sig": sig, "err": err, "uni": uni}


def top_up():
    pend = rdb.query("SELECT COUNT(*) c FROM research_tasks WHERE status='pending'")[0]["c"]
    added = 0
    if pend < MIN_PENDING:
        for fa in FAMS:
            rdb.execute(
                "INSERT INTO research_tasks (task_id,title,focus_area,prompt,priority,status,created_at) "
                "VALUES (%s,%s,%s,%s,%s,'pending',NOW(6))",
                (f"task-{uuid.uuid4().hex[:10]}", f"{fa}: monitored sweep", fa,
                 f"As the {fa} specialist, design and backtest a novel {fa} configuration on the S&P 500 "
                 f"(2018-2024) you have not tested; recall prior fleet findings to avoid repeats. Compare "
                 f"Sharpe/vol/maxDD/turnover vs 1/N net of cost and write an honest finding via the templated tool.",
                 5))
            added += 1
    return pend, added


def main():
    t0 = time.time()
    prev_findings = None
    print(f"# fleet monitor started {time.strftime('%H:%M:%S UTC', time.gmtime())}, interval={INTERVAL}s, duration~{DURATION//60}min", flush=True)
    while time.time() - t0 < DURATION:
        try:
            pend, added = top_up()
            s = snap()
            f = s["tot"]["f"]
            delta = "" if prev_findings is None else f" (+{f - prev_findings} findings since last)"
            prev_findings = f
            active = sum(1 for a in s["agents"] if a["age"] is not None and a["age"] < 240)
            ts = time.strftime("%H:%M:%S", time.gmtime())
            print(f"[{ts}] findings={f}{delta} exp={s['tot']['e']} act={s['tot']['a']} | "
                  f"tasks={s['tasks']} (topped +{added}) | agents_active(<4m)={active}/5 | "
                  f"beats={s['sig']['b']}/{s['sig']['c']} avgSh={s['sig']['avg']} maxSh={s['sig']['mx']} | "
                  f"ERR={s['err']} nonuniform={s['uni']}", flush=True)
            hb = " ".join(f"{a['agent_id'].split('-')[-1]}:{a['status'][:4]}/{a['age']}s" for a in s["agents"])
            print(f"           hb -> {hb}", flush=True)
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S', time.gmtime())}] monitor error: {e}", flush=True)
        time.sleep(INTERVAL)
    print(f"# fleet monitor done {time.strftime('%H:%M:%S UTC', time.gmtime())}", flush=True)


if __name__ == "__main__":
    main()
