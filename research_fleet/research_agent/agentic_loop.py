"""Agent-driven, host-side Claude tool-use research loop — the 24/7 brain.

Where :mod:`agent_loop` runs a FIXED pipeline (claim task -> recall -> hypothesize
-> backtest -> finding), this loop hands the wheel to the MODEL. Via Bedrock
``Converse`` tool-use (the verified inference path in :mod:`llm_driver`), Claude
decides what is most valuable to research next: it recalls what it and the fleet
already know, inspects the completed parameter sweep, forms a falsifiable
hypothesis, runs a REAL backtest, interprets the numbers honestly, and persists
the hypothesis->experiment->finding arc to SingleStore — then chooses the next
experiment. There is no queue to drain: "decide the next experiment" is always
available, so the agent runs continuously.

Two hard invariants carry over from :mod:`agent_tools` (real-money research):

  * NO FABRICATION. Metrics come ONLY from the ``run_backtest`` tool, which runs
    the actual backtester over the real SingleStore prices. The system prompt and
    tool descriptions tell the model plainly never to invent a number.
  * UNIFORM WRITES. Every write is funnelled through :func:`agent_tools.dispatch`,
    which injects ``agent_id`` host-side and routes through the validated
    ``write_tool`` path — the model cannot spoof identity or skip validation.

This module owns ONLY the host-side conversation loop + the 24/7 driver. It
REUSES ``llm_driver._client`` / ``llm_driver.MODELS`` for the unsigned
Bearer-JWT Bedrock client (never rebuilds it), ``agent_tools.TOOL_SPECS`` /
``dispatch`` / ``system_prompt`` for the tools, and ``research_db`` for
register/heartbeat/activity. It mirrors ``agent_loop``'s CLI conventions.
"""

from __future__ import annotations

import argparse
import os
import time
import traceback
from typing import Any

from . import agent_tools as at
from . import llm_driver
from . import research_db as rdb
from . import switchyard_transport


# Which write tools count toward the {hypotheses, experiments, findings} tally.
_WRITE_COUNTERS = {
    "write_hypothesis": "hypotheses",
    "write_experiment": "experiments",
    "write_finding": "findings",
}

# Transient converse failures get a couple of retries with a short backoff.
_CONVERSE_RETRIES = 2
_CONVERSE_BACKOFF = 3.0

# Default per-cycle inference budget.
_MAX_TOKENS = 2048
_TEMPERATURE = 0.3


# Research cycles rotate through work-types of genuinely different complexity so
# the Switchyard classifier (which grades the OPENING task of each cycle) spreads
# load across tiers instead of sending every cycle to the top tier: light survey
# work -> fast/haiku, a standard single-backtest -> balanced/sonnet, deep design/
# validation for a real-money decision -> reasoning/opus. Each cycle is still
# fully autonomous; only the opening objective differs. The weights bias toward
# real experiments while keeping a steady share of cheap survey + expensive design
# cycles. (Bedrock transport ignores the complexity hint; it just runs the task.)
_CYCLE_TYPES = [
    ("light", 0.30),
    ("standard", 0.45),
    ("deep", 0.25),
]


def _cycle_type(counter: int) -> str:
    """Deterministic rotation over cycle types honoring the target weights.

    Uses the cycle counter (no RNG — keeps runs reproducible and avoids the
    Date/random restrictions elsewhere in the stack). Expands the weights into a
    fixed 20-slot schedule and indexes by counter.
    """
    schedule: list[str] = []
    for name, w in _CYCLE_TYPES:
        schedule.extend([name] * max(1, round(w * 20)))
    return schedule[counter % len(schedule)]


def _kickoff_message(focus: str, cycle_type: str = "standard") -> dict:
    """The single user turn that starts one autonomous research cycle.

    ``cycle_type`` sets the opening objective's complexity so per-cycle routing
    has something to discriminate on:
      * light    — survey/summarize what the fleet already knows (cheap).
      * standard — run and interpret ONE real backtest (moderate).
      * deep     — design/validate/reconcile for a real-money decision (hard).
    """
    focus = (focus or "generalist").strip()
    common = (
        "Do not fabricate any metric — always obtain metrics from run_backtest. "
        "A strategy that does not beat the 1/N benchmark net of cost is a valid, "
        "reportable result. When the arc is complete, reply with a short plain-text summary."
    )
    if cycle_type == "light":
        text = (
            f"Begin a LIGHT survey cycle (low complexity). Recall what you and the "
            f"fleet already know about {focus} (recall_findings), scan the completed "
            f"sweep (query_sweep) and recent experiments (list_recent_experiments), "
            f"and write a short consolidating insight (write_finding, kind='insight') "
            f"summarizing the current state of {focus} — no new backtest required "
            f"unless a quick confirming run is clearly warranted. {common}"
        )
    elif cycle_type == "deep":
        text = (
            f"Begin a DEEP design cycle (HIGH complexity — this may inform a "
            f"real-money allocation). Recall prior {focus} findings and the sweep, "
            f"then rigorously reason about robustness: reconcile any in-sample vs "
            f"out-of-sample gaps, design and RUN a validation-oriented backtest "
            f"(e.g. a different regime/window or a stress of the current best {focus} "
            f"config), and record a hypothesis, an experiment (EXACT run_backtest "
            f"metrics), and an honest finding that states whether the edge is robust "
            f"or likely overfit, then set the hypothesis status. {common}"
        )
    else:  # standard
        text = (
            f"Begin a STANDARD research cycle (moderate complexity). Recall prior "
            f"{focus} findings (recall_findings) and check the sweep (query_sweep, "
            f"list_recent_experiments), decide the single most valuable {focus} "
            f"experiment to run next, run a REAL backtest (run_backtest), and record "
            f"a hypothesis, an experiment (with the EXACT run_backtest metrics), and "
            f"an honest finding, then set the hypothesis status. {common}"
        )
    return {"role": "user", "content": [{"text": text}]}


def _is_error_result(result: Any) -> bool:
    """A dispatch dict is an error if it carries an 'error' key or ok is false."""
    if not isinstance(result, dict):
        return True
    if result.get("ok") is False:
        return True
    if "error" in result and result["error"]:
        return True
    return False


def _converse(client, *, model_id: str, messages: list[dict], system: str,
              max_tokens: int) -> dict:
    """One converse call with a couple of retries on transient gateway errors.

    Raises the last exception if every attempt fails; the caller ends the cycle
    gracefully so one bad cycle never crashes the 24/7 process.
    """
    last: Exception | None = None
    for attempt in range(_CONVERSE_RETRIES + 1):
        try:
            return client.converse(
                modelId=model_id,
                messages=messages,
                toolConfig={"tools": at.TOOL_SPECS},
                system=[{"text": system}],
                inferenceConfig={"maxTokens": max_tokens, "temperature": _TEMPERATURE},
            )
        except Exception as e:  # noqa: BLE001 — transient gateway/socket errors
            last = e
            if attempt < _CONVERSE_RETRIES:
                time.sleep(_CONVERSE_BACKOFF * (attempt + 1))
    assert last is not None
    raise last


def _final_text(message: dict) -> str:
    """Concatenate the text blocks of an assistant message."""
    parts = message.get("content") or []
    return "".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()


def run_cycle(agent_id: str, focus: str, *, model: str = "sonnet",
              max_steps: int = 16, client=None, transport: str = "bedrock",
              cycle_type: str = "standard") -> dict:
    """Run ONE full agent-driven research cycle (one Converse tool-use conversation).

    The model drives: it calls tools until it emits a final text answer (one
    cycle done) or ``max_steps`` converse turns are hit (capped, to bound spend).
    Every ``toolUse`` block in an assistant turn is answered with a matching
    ``toolResult`` (same ``toolUseId``) in the very next user turn, or the API
    errors — multiple tool calls in one turn are all answered.

    ``transport`` selects the inference path when ``client`` is None:

      * ``"bedrock"`` (default) — the original, unchanged path: a boto3 Bedrock
        client built from :func:`llm_driver._client` with the focus model's Bearer
        JWT, calling the FIXED ``model`` tier.
      * ``"switchyard"`` — a :class:`switchyard_transport.SwitchyardTransport`
        (duck-compatible ``.converse``) that routes each turn through the NeMo
        Switchyard proxy, which classifies complexity and picks a tier per turn.
        ``model_id`` is passed through but ignored by that transport.

    ``client`` is injectable for tests and takes precedence over ``transport``;
    when None the transport chooses how the client is built.

    Returns a summary dict:
        {agent_id, focus, model, transport, steps, tool_calls: [names],
         wrote: {hypotheses, experiments, findings}, final_text, capped,
         tiers: [chosen model per turn], error?}
    """
    model_id, key = llm_driver.MODELS.get(model, llm_driver.MODELS.get("sonnet", ("", "")))
    summary: dict[str, Any] = {
        "agent_id": agent_id,
        "focus": focus,
        "model": model,
        "transport": transport,
        "cycle_type": cycle_type,
        "steps": 0,
        "tool_calls": [],
        "wrote": {"hypotheses": 0, "experiments": 0, "findings": 0},
        "final_text": "",
        "capped": False,
        "tiers": [],
    }

    if client is None:
        if transport == "switchyard":
            try:
                client = switchyard_transport.make_transport()
            except Exception as e:  # noqa: BLE001
                summary["error"] = f"switchyard transport build failed: {type(e).__name__}: {e}"
                return summary
        else:
            if not model_id or not key or not llm_driver.LLM_ENDPOINT:
                summary["error"] = f"LLM not configured for model '{model}'"
                return summary
            try:
                client = llm_driver._client(key)
            except Exception as e:  # noqa: BLE001
                summary["error"] = f"client build failed: {type(e).__name__}: {e}"
                return summary

    system = at.system_prompt(focus)
    messages: list[dict] = [_kickoff_message(focus, cycle_type)]

    _safe_activity(agent_id, "START",
                   {"focus": focus, "model": model, "loop": "agentic", "transport": transport})

    try:
        for _ in range(max_steps):
            summary["steps"] += 1
            resp = _converse(client, model_id=model_id, messages=messages,
                             system=system, max_tokens=_MAX_TOKENS)
            message = resp.get("output", {}).get("message", {})
            stop = resp.get("stopReason")
            # If the transport routed this turn (Switchyard), record which tier
            # handled it so we can SEE hard turns go to opus/sonnet, easy to haiku.
            tier = resp.get("_switchyard_model")
            if tier:
                summary["tiers"].append(tier)
            # Record the assistant turn verbatim so tool ids line up on the next turn.
            messages.append(message)

            if stop != "tool_use":
                summary["final_text"] = _final_text(message)
                break

            # Answer EVERY toolUse block with a matching toolResult (same id).
            tool_results: list[dict] = []
            for block in message.get("content") or []:
                if not isinstance(block, dict) or "toolUse" not in block:
                    continue
                tu = block["toolUse"]
                name = tu.get("name")
                tool_use_id = tu.get("toolUseId")
                tool_input = tu.get("input") or {}
                summary["tool_calls"].append(name)

                result = at.dispatch(name, tool_input, agent_id=agent_id)
                is_err = _is_error_result(result)
                if not is_err and name in _WRITE_COUNTERS:
                    summary["wrote"][_WRITE_COUNTERS[name]] += 1

                _safe_activity(agent_id, "TOOL",
                               {"tool": name, "ok": not is_err,
                                "id": result.get("id") if isinstance(result, dict) else None})

                tool_results.append({
                    "toolResult": {
                        "toolUseId": tool_use_id,
                        "content": [{"json": result}],
                        "status": "error" if is_err else "success",
                    }
                })

            if not tool_results:
                # stopReason said tool_use but no toolUse block was present — end
                # gracefully rather than send an empty user turn (which the API rejects).
                summary["final_text"] = _final_text(message)
                break

            messages.append({"role": "user", "content": tool_results})
        else:
            # for-loop exhausted without break => hit the step cap.
            summary["capped"] = True
    except Exception as e:  # noqa: BLE001 — never let one bad cycle crash the process
        summary["error"] = f"{type(e).__name__}: {e}"
        _safe_activity(agent_id, "ERROR",
                       {"error": str(e)[:400], "tb": traceback.format_exc()[-600:]})

    _safe_activity(agent_id, "END", {
        "steps": summary["steps"], "tool_calls": summary["tool_calls"],
        "wrote": summary["wrote"], "capped": summary["capped"],
        "transport": transport, "tiers": summary["tiers"],
        "error": summary.get("error"),
    })
    return summary


def _safe_activity(agent_id: str, phase: str, detail: dict) -> None:
    """Log a lightweight activity row; swallow any DB error (auditing is best-effort)."""
    try:
        rdb.log_activity(agent_id, phase, detail=detail)
    except Exception:  # noqa: BLE001
        pass


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
    except Exception:  # noqa: BLE001
        pass
    return out


def main(argv=None):
    ap = argparse.ArgumentParser("research-agent-agentic")
    ap.add_argument("--agent-id", default=os.environ.get("AGENT_ID", "research-agentic-01"))
    ap.add_argument("--display-name", default=os.environ.get("AGENT_NAME", ""))
    ap.add_argument("--focus", default=os.environ.get("AGENT_FOCUS", ""))
    ap.add_argument("--model", default=os.environ.get("AGENT_MODEL", "sonnet"))
    ap.add_argument("--transport", choices=["bedrock", "switchyard"],
                    default=os.environ.get("AGENT_TRANSPORT", "bedrock"),
                    help="inference path: 'bedrock' (default, fixed model tier) or "
                         "'switchyard' (NeMo proxy picks a tier per turn)")
    ap.add_argument("--once", action="store_true", help="run a single cycle then exit")
    ap.add_argument("--max-cycles", type=int, default=0, help="0 = unbounded")
    ap.add_argument("--max-steps", type=int, default=16, help="max converse turns per cycle")
    ap.add_argument("--sleep", type=float, default=15.0, help="seconds between cycles")
    args = ap.parse_args(argv)

    meta = _instance_meta()
    focus = args.focus or "generalist"
    display = args.display_name or f"Agentic Researcher {args.agent_id}"
    model_id = llm_driver.MODELS.get(args.model, ("", ""))[0] or args.model
    rdb.register_agent(args.agent_id, display, focus,
                       persona=at.system_prompt(focus), model=model_id,
                       instance_id=meta["instance_id"], private_ip=meta["private_ip"],
                       az=meta["az"])
    print(f"[{args.agent_id}] registered agentic loop (focus={focus}, model={args.model}, "
          f"transport={args.transport}, instance={meta['instance_id'] or 'local'})", flush=True)

    cycles = 0
    try:
        while True:
            rdb.heartbeat(args.agent_id)
            ctype = _cycle_type(cycles)
            summary = run_cycle(args.agent_id, focus, model=args.model,
                                max_steps=args.max_steps, transport=args.transport,
                                cycle_type=ctype)
            cycles += 1
            w = summary["wrote"]
            note = f" ERROR={summary['error']}" if summary.get("error") else ""
            cap = " CAPPED" if summary.get("capped") else ""
            tiers = f" tiers={summary['tiers']}" if summary.get("tiers") else ""
            print(f"[{args.agent_id}] cycle {cycles} [{ctype}]: steps={summary['steps']} "
                  f"tools={summary['tool_calls']} "
                  f"wrote(h/e/f)={w['hypotheses']}/{w['experiments']}/{w['findings']}"
                  f"{tiers}{cap}{note}", flush=True)
            rdb.heartbeat(args.agent_id)

            if args.once:
                break
            if args.max_cycles and cycles >= args.max_cycles:
                print(f"[{args.agent_id}] reached max cycles {cycles}", flush=True)
                break
            time.sleep(args.sleep)
    except KeyboardInterrupt:
        print(f"[{args.agent_id}] interrupted; going idle", flush=True)
    finally:
        try:
            rdb.heartbeat(args.agent_id, status="idle")
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
