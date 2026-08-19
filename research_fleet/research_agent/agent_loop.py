"""Auto-research agent loop — the program each tiny EC2 runs continuously.

One cycle:
  1. CLAIM a research task from the shared SingleStore queue (atomic).
  2. RECALL relevant prior findings across the whole fleet (Qwen VECTOR search) —
     so the agent builds on collective knowledge, not from scratch.
  3. HYPOTHESIZE: Claude proposes a concrete, testable strategy parameterization
     (as JSON) for the task, informed by the recalled findings.
  4. EXPERIMENT: run a real backtest of that parameterization against the sp500
     prices in SingleStore; persist the metrics.
  5. (optional) ANALYZE via Aura Analyst: ask an NL question over SingleStore
     through the REAL Portal endpoint (only if configured — never a local
     NL->SQL substitute).
  6. FINDING: Claude writes a durable, quantitative finding; embed + persist it.
  7. Mark the task done, heartbeat, loop.

Everything persists to SingleStore, so results accumulate OVER TIME and survive
the box restarting. Designed to be launched by OpenClaw-through-NemoClaw on the
EC2 (NemoClaw routes the agent's inference at our endpoint); the structured
sub-calls here use the same code-factory Anthropic gateway via llm_driver.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from datetime import datetime

from . import research_db as rdb
from . import backtest as bt
from . import analyst
from . import llm_driver
from . import write_tool as wt
from . import prompts as pr


def _instance_meta() -> dict:
    """Best-effort EC2 instance identity via IMDSv2 (empty off-EC2)."""
    out = {"instance_id": "", "private_ip": "", "az": ""}
    try:
        import requests
        tok = requests.put("http://169.254.169.254/latest/api/token",
                           headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"}, timeout=2).text
        h = {"X-aws-ec2-metadata-token": tok}
        b = "http://169.254.169.254/latest/meta-data"
        out["instance_id"] = requests.get(f"{b}/instance-id", headers=h, timeout=2).text
        out["private_ip"] = requests.get(f"{b}/local-ipv4", headers=h, timeout=2).text
        out["az"] = requests.get(f"{b}/placement/availability-zone", headers=h, timeout=2).text
    except Exception:
        pass
    return out


def _default_window(prompt: str) -> tuple[str, str]:
    """Pick a sensible backtest window; prefer years named in the brief."""
    import re
    years = [int(y) for y in re.findall(r"20\d\d", prompt)]
    if len(years) >= 2:
        return f"{min(years)}-01-01", f"{max(years)}-12-31"
    return "2018-01-01", "2024-06-30"


def run_one_cycle(agent_id: str, focus_area: str | None, *, model: str = "sonnet",
                  agent_focus: str | None = None) -> bool:
    """Claim + complete one research task. Returns False if nothing to claim.

    ``focus_area`` biases which task is claimed; ``agent_focus`` is the node's
    own specialty and selects the specialist prompt (defaults to focus_area).
    """
    agent_focus = agent_focus or focus_area
    task = rdb.claim_task(agent_id, focus_area=focus_area)
    if not task:
        return False
    tid = task["task_id"]
    fa = task["focus_area"]
    brief = task["prompt"]
    # The node's own specialty prompt drives its reasoning (agent_focus); fall
    # back to the task's focus_area, then a generalist.
    specialty = agent_focus or fa
    system_prompt = pr.system_for(specialty)
    # ALL writes go through the templated tool (write_tool) so every row is
    # uniform regardless of which agent/node writes it — the sandbox agent uses
    # the exact same validated path via the HTTP tool server.
    T = wt.TOOLS
    T["write_activity"](agent_id=agent_id, phase="START", task_id=tid,
                        detail={"title": task["title"], "focus": fa, "specialty": specialty})
    rdb.set_task_status(tid, "running")
    try:
        # 2) RECALL prior fleet findings
        prior = rdb.recall_findings(f"{task['title']}. {brief}", k=5, strategy_family=None)
        prior_ctx = "\n".join(f"- ({p['strategy_family']}) {p['content']}" for p in prior) or "(no prior findings yet)"
        T["write_activity"](agent_id=agent_id, phase="RECALL", task_id=tid, detail={"recalled": len(prior)})

        # 3) HYPOTHESIZE (Claude -> concrete params JSON), framed by the specialist task template
        task_framing = pr.task_for(specialty, brief=brief, prior_findings=prior_ctx, agent_id=agent_id)
        hyp = llm_driver.complete_json(
            f"{task_framing}\n\nAs the FIRST step of this arc, propose ONE concrete, testable strategy "
            f"configuration to backtest now. Return JSON with keys: statement (the hypothesis, one sentence), "
            f"rationale, strategy_family (one of: equal_weight, momentum, mean_reversion, vol_target, low_vol, "
            f"factor, risk_parity, regime), params (object with any of: lookback_days, skip_days, top_n, "
            f"bottom_n, reversal_days, keep_n, target_vol, ma_days, rebalance_days, w_max), confidence (0..1).",
            model=model, system=system_prompt, max_tokens=800)
        family = (hyp.get("strategy_family") or fa or "equal_weight").strip()
        params = hyp.get("params") or {}
        statement = hyp.get("statement") or task["title"]
        hrec = T["write_hypothesis"](agent_id=agent_id, statement=statement, task_id=tid,
                                     rationale=hyp.get("rationale", ""), strategy_family=family,
                                     params=params, confidence=hyp.get("confidence", 0.5))
        hid, family = hrec["id"], hrec["strategy_family"]  # use the tool's normalized family
        T["write_activity"](agent_id=agent_id, phase="HYPOTHESIS", task_id=tid,
                            detail={"hypothesis_id": hid, "family": family, "params": params})

        # 4) EXPERIMENT (real backtest over SingleStore prices)
        start, end = _default_window(brief)
        try:
            metrics = bt.run_backtest(family, params, start=start, end=end,
                                      universe_n=int(params.get("universe_n", 60) or 60))
        except Exception as be:
            metrics = {"error": f"backtest exception: {be}"}
        if not isinstance(metrics, dict):
            metrics = {"error": "backtest returned non-dict"}
        exp_status = "failed" if metrics.get("error") else "ok"
        erec = T["write_experiment"](agent_id=agent_id, strategy_family=family, params=params,
                                     metrics=metrics, hypothesis_id=hid, task_id=tid,
                                     universe=f"sp500-top{int(params.get('universe_n',60) or 60)}",
                                     method="python-backtest", engine="cpu", lookback_start=start,
                                     lookback_end=end, status=exp_status, error=metrics.get("error"))
        eid = erec["id"]
        T["write_activity"](agent_id=agent_id, phase="EXPERIMENT", task_id=tid,
                            detail={"experiment_id": eid, "sharpe": erec.get("sharpe"),
                                    "beats_benchmark": erec.get("beats_benchmark")})

        # 5) OPTIONAL: Aura Analyst NL analysis over SingleStore (only if configured).
        # Instead of one canned question, ask a small battery of genuinely analytical,
        # cross-cutting questions over the REAL Portal domain — situating THIS run
        # against the whole fleet's record — so the finding is informed by live data,
        # not just this single backtest. Each is audited; one failure never aborts the
        # rest, and Aura being unavailable simply skips the phase (no local NL->SQL).
        analyst_note = ""
        if analyst.available():
            analyst_questions = [
                (f"Across research_experiments for strategy_family '{family}', what is the "
                 f"average sharpe, the average max_drawdown, and how many experiments beat "
                 f"the benchmark versus the total?"),
                (f"Among all strategy families in research_experiments, rank them by average "
                 f"sharpe and show how '{family}' compares."),
                (f"For strategy_family '{family}', what were the best and worst sharpe values "
                 f"recorded and their turnover, so I can see the dispersion of outcomes?"),
            ]
            analyst_findings: list[str] = []
            for q in analyst_questions:
                try:
                    a = analyst.ask(q, output_modes=["sql", "data", "text"], agent_id=agent_id)
                    T["record_analyst_query"](agent_id=agent_id, question=q, task_id=tid,
                                              generated_sql=a.get("sql") or "", row_count=a.get("row_count") or 0,
                                              answer=(a.get("text") or ""), latency_ms=a.get("latency_ms") or 0.0,
                                              status="ok" if not a.get("error") else "error")
                    T["write_activity"](agent_id=agent_id, phase="ANALYST", task_id=tid,
                                        detail={"question": q, "row_count": a.get("row_count"),
                                                "error": a.get("error")})
                    if a.get("error"):
                        continue
                    # Prefer Aura's narrated answer; fall back to the raw rows.
                    detail = a.get("text") or (str(a.get("rows"))[:600] if a.get("rows") else "")
                    if detail:
                        analyst_findings.append(f"Q: {q}\nAura: {detail}")
                except Exception as e:
                    T["write_activity"](agent_id=agent_id, phase="ANALYST", task_id=tid,
                                        detail={"question": q, "error": str(e)[:200]})
            if analyst_findings:
                analyst_note = ("\n\nAura Analyst cross-experiment analysis over SingleStore:\n"
                                + "\n\n".join(analyst_findings))

        # 6) FINDING (Claude writes the durable, quantitative conclusion)
        finding_txt = llm_driver.complete(
            f"Brief: {brief}\n\nHypothesis: {statement}\nConfig tested: {json.dumps(params)}\n"
            f"Backtest window {start}..{end}. Metrics: {json.dumps({k:v for k,v in metrics.items() if k!='error'})}"
            f"{analyst_note}\n\nWrite a 2-4 sentence QUANTITATIVE finding for the research fleet: did it "
            f"beat the 1/N benchmark on Sharpe? note the risk (vol, max drawdown), turnover cost, and one "
            f"concrete next step. Be honest if the edge is marginal or absent.",
            model=model, system=system_prompt, max_tokens=400)
        beats = bool(metrics.get("beats_benchmark"))
        frec = T["write_finding"](agent_id=agent_id, content=finding_txt, title=statement[:200],
                                  strategy_family=family, experiment_id=eid, hypothesis_id=hid, task_id=tid,
                                  metrics=metrics, tags=[family, "auto-research"])
        fid = frec["id"]
        T["write_activity"](agent_id=agent_id, phase="FINDING", task_id=tid,
                            detail={"finding_id": fid, "beats_benchmark": beats})

        summary = (f"{family}: Sharpe {metrics.get('sharpe')}, beats_bench={beats}. {finding_txt[:200]}")
        rdb.set_task_status(tid, "done", result_summary=summary)
        # update hypothesis status from the result (via the templated tool)
        T["set_hypothesis_status"](hypothesis_id=hid, status="supported" if beats else "inconclusive")
        T["write_activity"](agent_id=agent_id, phase="END", task_id=tid, detail={"status": "done"})
        return True
    except Exception as e:
        try:
            wt.TOOLS["write_activity"](agent_id=agent_id, phase="ERROR", task_id=tid,
                                       detail={"error": str(e)[:400], "tb": traceback.format_exc()[-800:]})
        except Exception:
            rdb.log_activity(agent_id, "ERROR", task_id=tid, detail={"error": str(e)[:400]})
        rdb.set_task_status(tid, "failed", result_summary=str(e)[:400])
        return True


def main(argv=None):
    ap = argparse.ArgumentParser("research-agent")
    ap.add_argument("--agent-id", default=os.environ.get("AGENT_ID", "research-01"))
    ap.add_argument("--display-name", default=os.environ.get("AGENT_NAME", ""))
    ap.add_argument("--focus", default=os.environ.get("AGENT_FOCUS", ""))  # empty = any
    ap.add_argument("--model", default=os.environ.get("AGENT_MODEL", "sonnet"))
    ap.add_argument("--once", action="store_true", help="run a single cycle then exit")
    ap.add_argument("--idle-sleep", type=int, default=120, help="seconds to wait when queue empty")
    ap.add_argument("--max-cycles", type=int, default=0, help="0 = unbounded")
    args = ap.parse_args(argv)

    meta = _instance_meta()
    display = args.display_name or f"Researcher {args.agent_id}"
    focus = args.focus or None
    rdb.register_agent(args.agent_id, display, focus or "general", persona=pr.system_for(focus),
                       model=llm_driver.MODELS.get(args.model, ("", ""))[0] or args.model,
                       instance_id=meta["instance_id"], private_ip=meta["private_ip"], az=meta["az"])
    print(f"[{args.agent_id}] registered (focus={focus}, model={args.model}, instance={meta['instance_id']})", flush=True)

    cycles = 0
    while True:
        rdb.heartbeat(args.agent_id)
        did = run_one_cycle(args.agent_id, focus, model=args.model, agent_focus=focus)
        cycles += 1 if did else 0
        if args.once:
            break
        if args.max_cycles and cycles >= args.max_cycles:
            print(f"[{args.agent_id}] reached max cycles {cycles}", flush=True)
            break
        if not did:
            # queue empty: on the shared queue this agent has nothing to do; idle.
            rdb.heartbeat(args.agent_id, status="idle")
            print(f"[{args.agent_id}] queue empty; sleeping {args.idle_sleep}s", flush=True)
            time.sleep(args.idle_sleep)
        else:
            print(f"[{args.agent_id}] completed a research cycle ({cycles} total)", flush=True)
            time.sleep(3)

    rdb.heartbeat(args.agent_id, status="idle")


if __name__ == "__main__":
    main()
