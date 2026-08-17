"""Tool suite for the autonomous, agent-driven trading-strategy research loop.

A host-side Claude tool-use loop (Bedrock ``Converse``, built separately) lets the
model DECIDE what to research next; this module hands it the REAL tools to do it.
It exposes three things the loop needs and nothing else:

  1. :data:`TOOL_SPECS` — the Bedrock ``toolSpec`` list passed straight into
     ``toolConfig.tools``. Each entry declares a tool name + JSON input schema +
     a description telling the model WHEN to reach for it.
  2. :func:`dispatch` — routes one tool call to the real implementation
     (``backtest``/``research_db``/``write_tool``) and returns a JSON-serializable
     dict the loop wraps into a ``toolResult`` block. It NEVER raises — a raise
     would crash the agent loop — so every path returns a dict (a success payload
     or ``{"error": ...}`` / ``{"ok": false, "error": ...}`` the model can read
     and correct against).
  3. :func:`system_prompt` — a per-focus research directive for the loop.

Two hard invariants, because this is a real-money research system:

  * NO FABRICATION. ``run_backtest`` runs the ACTUAL backtester over the real
    SingleStore ``prices`` — the metrics the model sees are the metrics that
    happened, never invented. The agent is told plainly (in the tool descriptions
    and the system prompt) that metrics come ONLY from ``run_backtest``.
  * UNIFORM WRITES. Every write is funnelled through the validated
    :mod:`write_tool` path (controlled ``STRATEGY_FAMILIES`` enum, canonical
    ``METRIC_KEYS``), with ``agent_id``/``task_id`` injected host-side by
    :func:`dispatch` so the model cannot spoof identity or skip validation.

Read tools are shaped to stay token-efficient: rows are capped, floats rounded,
embeddings dropped, long text clamped — complete enough to reason on, small
enough that one tool result isn't enormous.
"""

from __future__ import annotations

from typing import Any

from . import backtest as bt
from . import research_db as rdb
from . import write_tool as wt
from . import prompts as pr


# The eight controlled strategy families the backtester + write path understand.
# Reused from write_tool so there is a single source of truth for the enum.
STRATEGY_FAMILIES = sorted(wt.STRATEGY_FAMILIES)

# Backtest window defaults (match the demo's 2005-2024 price coverage).
DEFAULT_START = "2015-01-01"
DEFAULT_END = "2024-12-31"
DEFAULT_UNIVERSE_N = 60

# Result-shaping caps so a single tool result never balloons the context.
MAX_RECALL_ROWS = 12
MAX_SWEEP_ROWS = 50
MAX_EXPERIMENT_ROWS = 50
_CONTENT_CLAMP = 1200      # chars of finding content returned on recall
_PARAMS_CLAMP = 800        # chars of a serialized params blob


# ---------------------------------------------------------------------------
# Bedrock toolSpec list (toolConfig.tools) — declares each tool to the model
# ---------------------------------------------------------------------------

TOOL_SPECS: list[dict] = [
    {
        "toolSpec": {
            "name": "recall_findings",
            "description": (
                "Semantic recall of prior findings from the WHOLE research fleet "
                "(Qwen vector search over research_findings). Use this FIRST, before "
                "hypothesizing, so you build on what you and the fleet already learned "
                "instead of re-running settled experiments. Returns the most relevant "
                "prior findings (content, family, metrics, importance)."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural-language description of what you want to recall, "
                                           "e.g. 'does 12-1 momentum beat 1/N net of cost'.",
                        },
                        "k": {
                            "type": "integer",
                            "description": "How many prior findings to return (default 6, max 12).",
                        },
                        "strategy_family": {
                            "type": "string",
                            "description": "Optional: restrict recall to one family.",
                            "enum": STRATEGY_FAMILIES,
                        },
                    },
                    "required": ["query"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "run_backtest",
            "description": (
                "Run a REAL backtest over the S&P 500 daily prices in SingleStore and "
                "return the actual metrics (sharpe, ann_return, ann_vol, max_drawdown, "
                "turnover, cvar_95, win_rate, sortino, total_return, n_rebalances, "
                "benchmark_sharpe, beats_benchmark, all_in_cost_bps, data_caveats). This "
                "is the ONLY source of truth for metrics — NEVER invent numbers, always "
                "get them here. Costs are charged on real turnover and results are always "
                "compared to the 1/N equal-weight benchmark net of cost. A strategy that "
                "does NOT beat 1/N is a valid, reportable result."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "strategy_family": {
                            "type": "string",
                            "description": "Which strategy to evaluate.",
                            "enum": STRATEGY_FAMILIES,
                        },
                        "params": {
                            "type": "object",
                            "description": "Strategy parameters, e.g. {lookback_days, skip_days, top_n, "
                                           "bottom_n, reversal_days, keep_n, target_vol, ma_days, "
                                           "rebalance_days, w_max, turnover_cost_bps, slippage_bps}. "
                                           "Unknown keys are ignored by the relevant family.",
                        },
                        "start": {
                            "type": "string",
                            "description": f"Backtest start date YYYY-MM-DD (default {DEFAULT_START}).",
                        },
                        "end": {
                            "type": "string",
                            "description": f"Backtest end date YYYY-MM-DD (default {DEFAULT_END}).",
                        },
                        "universe_n": {
                            "type": "integer",
                            "description": f"Number of names in the universe (default {DEFAULT_UNIVERSE_N}).",
                        },
                    },
                    "required": ["strategy_family", "params"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "query_sweep",
            "description": (
                "Read the already-completed parameter sweep (thousands of configs "
                "backtested on an in-sample and out-of-sample window, in sweep_results). "
                "Use this to see what has ALREADY been tested and how it held up out of "
                "sample before proposing your own config. Returns top rows with family, "
                "params, is_sharpe, oos_sharpe, the IS->OOS gap, and whether it beat its "
                "benchmark. Rank by oos_sharpe (honest) or is_sharpe (the overfit view)."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "sort_by": {
                            "type": "string",
                            "description": "Rank by out-of-sample Sharpe (default, honest) or in-sample "
                                           "Sharpe (the overfit-prone view).",
                            "enum": ["oos_sharpe", "is_sharpe"],
                        },
                        "family": {
                            "type": "string",
                            "description": "Optional: restrict to one family.",
                            "enum": STRATEGY_FAMILIES,
                        },
                        "limit": {
                            "type": "integer",
                            "description": "How many rows to return (default 10, max 50).",
                        },
                        "robust_only": {
                            "type": "boolean",
                            "description": "If true, only rows whose edge survived out of sample "
                                           "(oos_sharpe>0, beat OOS benchmark, contained IS->OOS gap).",
                        },
                    },
                    "required": [],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "list_recent_experiments",
            "description": (
                "List recent research_experiments the fleet has already run (most recent "
                "first), so you don't repeat a config that was just tested. Returns family, "
                "params, the key metrics, and status."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "How many recent experiments to return (default 15, max 50).",
                        },
                        "strategy_family": {
                            "type": "string",
                            "description": "Optional: restrict to one family.",
                            "enum": STRATEGY_FAMILIES,
                        },
                    },
                    "required": [],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "write_hypothesis",
            "description": (
                "Record a falsifiable, quantitative hypothesis BEFORE you backtest it. "
                "State the specific design and the edge you expect to survive net of cost. "
                "Returns the new hypothesis id to link the experiment + finding to."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "statement": {"type": "string", "description": "The hypothesis, one clear sentence."},
                        "strategy_family": {"type": "string", "enum": STRATEGY_FAMILIES},
                        "rationale": {"type": "string", "description": "Why you expect this edge."},
                        "params": {"type": "object", "description": "The config you intend to test."},
                        "confidence": {
                            "type": "number",
                            "description": "Prior confidence 0..1 (default 0.5).",
                        },
                    },
                    "required": ["statement", "strategy_family"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "write_experiment",
            "description": (
                "Record a backtest run and its metrics. Pass the EXACT metrics dict "
                "returned by run_backtest (only the canonical keys are stored). Link it to "
                "the hypothesis via hypothesis_id. Returns the experiment id."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "strategy_family": {"type": "string", "enum": STRATEGY_FAMILIES},
                        "params": {"type": "object", "description": "The config that was tested."},
                        "metrics": {
                            "type": "object",
                            "description": "Metrics from run_backtest (sharpe, ann_vol, ann_return, "
                                           "max_drawdown, turnover, cvar_95, win_rate, n_rebalances, "
                                           "benchmark_sharpe, beats_benchmark, sortino, total_return, "
                                           "universe_size).",
                        },
                        "hypothesis_id": {"type": "string", "description": "The hypothesis this tests."},
                        "universe": {"type": "string", "description": "e.g. 'sp500-top60'."},
                        "lookback_start": {"type": "string", "description": "Backtest start YYYY-MM-DD."},
                        "lookback_end": {"type": "string", "description": "Backtest end YYYY-MM-DD."},
                        "status": {
                            "type": "string",
                            "description": "'ok' (default) or 'failed'.",
                            "enum": ["ok", "failed"],
                        },
                        "error": {"type": "string", "description": "Error text if the run failed."},
                    },
                    "required": ["strategy_family", "params", "metrics"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "write_finding",
            "description": (
                "Record a durable, QUANTITATIVE finding for the fleet (auto-embedded for "
                "future recall). Cite the real metrics (Sharpe, vol, max drawdown, "
                "turnover) and the head-to-head vs the 1/N benchmark, be honest when there "
                "is no net edge, and end with a concrete next step. Link to the experiment "
                "and hypothesis. Returns the finding id."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "The finding text (quantitative + honest)."},
                        "strategy_family": {"type": "string", "enum": STRATEGY_FAMILIES},
                        "title": {"type": "string", "description": "Short title."},
                        "kind": {
                            "type": "string",
                            "description": "finding (default) | insight | caveat | next_step.",
                            "enum": sorted(wt.FINDING_KINDS),
                        },
                        "metrics": {"type": "object", "description": "Metrics from run_backtest to attach."},
                        "experiment_id": {"type": "string"},
                        "hypothesis_id": {"type": "string"},
                        "importance": {
                            "type": "number",
                            "description": "0..1 importance (default derived from beats_benchmark).",
                        },
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["content", "strategy_family"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "set_hypothesis_status",
            "description": (
                "Update a hypothesis's status after evaluating it, consistent with the "
                "numbers: supported | rejected | inconclusive (or open | testing)."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "hypothesis_id": {"type": "string"},
                        "status": {"type": "string", "enum": sorted(wt.HYPOTHESIS_STATUS)},
                    },
                    "required": ["hypothesis_id", "status"],
                }
            },
        }
    },
]

# Names the loop is allowed to dispatch (kept in sync with TOOL_SPECS).
_READ_TOOLS = {"recall_findings", "run_backtest", "query_sweep", "list_recent_experiments"}
_WRITE_TOOLS = {"write_hypothesis", "write_experiment", "write_finding", "set_hypothesis_status"}
TOOL_NAMES = [s["toolSpec"]["name"] for s in TOOL_SPECS]


# ---------------------------------------------------------------------------
# Result-shaping helpers (token-efficient, JSON-serializable)
# ---------------------------------------------------------------------------

def _round(v: Any, nd: int = 6) -> Any:
    """Round a float for compact output; pass non-numerics through untouched."""
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, float):
        return None if v != v else round(v, nd)  # drop NaN
    return v


def _round_metrics(m: dict) -> dict:
    """Round every numeric in a metrics dict, keep bools/strings (e.g. data_caveats)."""
    return {k: _round(v) for k, v in m.items()}


def _shorten(s: Any, n: int) -> Any:
    if isinstance(s, str) and len(s) > n:
        return s[:n] + "…"
    return s


def _params_of(row: dict) -> Any:
    """params comes back from SingleStore JSON already parsed to a dict; clamp its size."""
    p = row.get("params")
    if isinstance(p, (dict, list)):
        return p
    return _shorten(p, _PARAMS_CLAMP)


# ---------------------------------------------------------------------------
# Read-tool implementations
# ---------------------------------------------------------------------------

def _recall_findings(tool_input: dict) -> dict:
    query = tool_input.get("query")
    if not query or not str(query).strip():
        return {"error": "recall_findings requires a non-empty 'query'"}
    k = int(tool_input.get("k") or 6)
    k = max(1, min(k, MAX_RECALL_ROWS))
    fam = tool_input.get("strategy_family") or None
    rows = rdb.recall_findings(str(query), k=k, strategy_family=fam)
    out = []
    for r in rows:
        metrics = r.get("metrics")
        out.append({
            "finding_id": r.get("finding_id"),
            "agent_id": r.get("agent_id"),
            "kind": r.get("kind"),
            "title": r.get("title"),
            "content": _shorten(r.get("content"), _CONTENT_CLAMP),
            "strategy_family": r.get("strategy_family"),
            "metrics": _round_metrics(metrics) if isinstance(metrics, dict) else metrics,
            "importance": _round(r.get("importance"), 3),
            "score": _round(r.get("score"), 4),
            "created_at": str(r.get("created_at")) if r.get("created_at") is not None else None,
        })
    return {"count": len(out), "findings": out}


def _run_backtest(tool_input: dict) -> dict:
    fam = str(tool_input.get("strategy_family") or "").strip()
    if fam not in wt.STRATEGY_FAMILIES:
        return {"error": f"invalid strategy_family='{fam}'. allowed: {STRATEGY_FAMILIES}"}
    params = tool_input.get("params")
    if params is None:
        params = {}
    if not isinstance(params, dict):
        return {"error": "params must be a JSON object"}
    start = str(tool_input.get("start") or DEFAULT_START)
    end = str(tool_input.get("end") or DEFAULT_END)
    try:
        universe_n = int(tool_input.get("universe_n") or DEFAULT_UNIVERSE_N)
    except (TypeError, ValueError):
        universe_n = DEFAULT_UNIVERSE_N
    metrics = bt.run_backtest(fam, params, start=start, end=end, universe_n=universe_n)
    if not isinstance(metrics, dict):
        return {"error": "backtest returned a non-dict result"}
    shaped = _round_metrics(metrics)
    shaped["strategy_family"] = fam
    shaped["start"] = start
    shaped["end"] = end
    return shaped


def _query_sweep(tool_input: dict) -> dict:
    sort_by = tool_input.get("sort_by") or "oos_sharpe"
    if sort_by not in ("oos_sharpe", "is_sharpe"):
        sort_by = "oos_sharpe"
    limit = int(tool_input.get("limit") or 10)
    limit = max(1, min(limit, MAX_SWEEP_ROWS))
    fam = tool_input.get("family") or None
    robust_only = bool(tool_input.get("robust_only"))

    where = ["error IS NULL", f"{sort_by} IS NOT NULL"]
    params: list[Any] = []
    if fam:
        where.append("family=%s")
        params.append(fam)
    if robust_only:
        # "survived the walk-forward": positive OOS Sharpe, beat its OOS benchmark,
        # and the IS->OOS degradation stayed contained (gap <= 1.0). Mirrors the
        # robustness gate in sweep_analyze without importing its constants here.
        where.append("oos_sharpe > 0")
        where.append("oos_beats_benchmark = 1")
        where.append("(is_oos_sharpe_gap IS NULL OR is_oos_sharpe_gap <= 1.0)")
    wc = " AND ".join(where)
    rows = rdb.query(
        f"""SELECT result_id, family, params, is_sharpe, oos_sharpe, is_oos_sharpe_gap,
                   oos_ann_return, oos_max_drawdown, oos_turnover, is_beats_benchmark,
                   oos_beats_benchmark, all_in_cost_bps, universe_n
            FROM sweep_results
            WHERE {wc}
            ORDER BY {sort_by} DESC
            LIMIT %s""",
        [*params, limit])
    out = []
    for r in rows:
        out.append({
            "result_id": r.get("result_id"),
            "family": r.get("family"),
            "params": _params_of(r),
            "is_sharpe": _round(r.get("is_sharpe"), 4),
            "oos_sharpe": _round(r.get("oos_sharpe"), 4),
            "is_oos_sharpe_gap": _round(r.get("is_oos_sharpe_gap"), 4),
            "oos_ann_return": _round(r.get("oos_ann_return"), 4),
            "oos_max_drawdown": _round(r.get("oos_max_drawdown"), 4),
            "oos_turnover": _round(r.get("oos_turnover"), 4),
            "is_beats_benchmark": int(r["is_beats_benchmark"]) if r.get("is_beats_benchmark") is not None else None,
            "oos_beats_benchmark": int(r["oos_beats_benchmark"]) if r.get("oos_beats_benchmark") is not None else None,
            "all_in_cost_bps": _round(r.get("all_in_cost_bps"), 2),
            "universe_n": int(r["universe_n"]) if r.get("universe_n") is not None else None,
        })
    return {"count": len(out), "sort_by": sort_by, "robust_only": robust_only, "rows": out}


def _list_recent_experiments(tool_input: dict) -> dict:
    limit = int(tool_input.get("limit") or 15)
    limit = max(1, min(limit, MAX_EXPERIMENT_ROWS))
    fam = tool_input.get("strategy_family") or None
    where = ""
    params: list[Any] = []
    if fam:
        where = "WHERE strategy_family=%s"
        params.append(fam)
    rows = rdb.query(
        f"""SELECT experiment_id, agent_id, strategy_family, params, sharpe, ann_return,
                   ann_vol, max_drawdown, turnover, benchmark_sharpe, beats_benchmark,
                   status, lookback_start, lookback_end, finished_at
            FROM research_experiments
            {where}
            ORDER BY finished_at DESC, started_at DESC
            LIMIT %s""",
        [*params, limit])
    out = []
    for r in rows:
        out.append({
            "experiment_id": r.get("experiment_id"),
            "agent_id": r.get("agent_id"),
            "strategy_family": r.get("strategy_family"),
            "params": _params_of(r),
            "sharpe": _round(r.get("sharpe"), 4),
            "ann_return": _round(r.get("ann_return"), 4),
            "ann_vol": _round(r.get("ann_vol"), 4),
            "max_drawdown": _round(r.get("max_drawdown"), 4),
            "turnover": _round(r.get("turnover"), 4),
            "benchmark_sharpe": _round(r.get("benchmark_sharpe"), 4),
            "beats_benchmark": int(r["beats_benchmark"]) if r.get("beats_benchmark") is not None else None,
            "status": r.get("status"),
            "lookback_start": str(r.get("lookback_start")) if r.get("lookback_start") is not None else None,
            "lookback_end": str(r.get("lookback_end")) if r.get("lookback_end") is not None else None,
            "finished_at": str(r.get("finished_at")) if r.get("finished_at") is not None else None,
        })
    return {"count": len(out), "experiments": out}


# ---------------------------------------------------------------------------
# Dispatch — route one tool call to the real implementation (NEVER raises)
# ---------------------------------------------------------------------------

def dispatch(name: str, tool_input: dict, *, agent_id: str, task_id: str | None = None) -> dict:
    """Execute one tool call and return a JSON-serializable dict.

    Read tools hit the DB / run the real backtest and return the actual data.
    Write tools inject ``agent_id`` (and ``task_id`` if given) then go through the
    validated :func:`write_tool.call_tool` path, so every row is uniform and the
    model cannot spoof identity or bypass validation. Validation failures bubble
    up as ``{"ok": false, "error": ...}`` (mirroring ``write_tool``'s ToolError
    text) so the model can see and correct them.

    This function NEVER raises: a raise out of a tool would crash the host agent
    loop, so every path is wrapped and returns a dict.
    """
    try:
        if not isinstance(tool_input, dict):
            return {"error": "tool_input must be a JSON object"}

        if name == "recall_findings":
            return _recall_findings(tool_input)
        if name == "run_backtest":
            return _run_backtest(tool_input)
        if name == "query_sweep":
            return _query_sweep(tool_input)
        if name == "list_recent_experiments":
            return _list_recent_experiments(tool_input)

        if name in _WRITE_TOOLS:
            payload = dict(tool_input)  # copy — never mutate the model's input
            # Identity is injected host-side; the model cannot set/override it.
            if name != "set_hypothesis_status":
                payload["agent_id"] = agent_id
                if task_id is not None:
                    payload["task_id"] = task_id
            try:
                return wt.call_tool(name, payload)
            except wt.ToolError as te:
                # Surface validation errors so the model can correct itself.
                return {"ok": False, "error": str(te)}

        return {"error": f"unknown tool '{name}'. tools: {TOOL_NAMES}"}
    except Exception as e:  # noqa: BLE001 — must never raise into the agent loop
        return {"error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# System prompt — the per-focus research directive the loop uses
# ---------------------------------------------------------------------------

def system_prompt(focus_area: str) -> str:
    """A strong, loop-specific research directive for the ``focus_area`` desk.

    Composes on top of the specialist's deep-domain prompt from :mod:`prompts`
    (``system_for``) when one exists, then appends the autonomous-loop directive
    that tells the model how to DRIVE the tool loop: recall + inspect the sweep
    first, hypothesize, run a REAL backtest, interpret honestly, and persist the
    hypothesis->experiment->finding arc via the write tools.
    """
    focus = (focus_area or "generalist").strip()
    base = pr.system_for(focus_area)  # specialist domain expertise + write-tool rules
    directive = (
        f"\n\n## Your mission (autonomous research loop)\n"
        f"You are the {focus} specialist on an autonomous quant research fleet. "
        f"Continuously investigate {focus} strategies against REAL S&P 500 daily "
        f"price data in SingleStore. Decide what is most valuable to test next based "
        f"on what you and the fleet have ALREADY found: call recall_findings and "
        f"query_sweep FIRST (and list_recent_experiments) so you extend collective "
        f"knowledge rather than repeat settled experiments. Then form a falsifiable "
        f"hypothesis (write_hypothesis), run a REAL backtest (run_backtest), and "
        f"interpret it HONESTLY — a strategy that does NOT beat the 1/N equal-weight "
        f"benchmark net of cost is a valid, reportable finding. Persist the full arc: "
        f"write_hypothesis -> write_experiment (with the exact run_backtest metrics) "
        f"-> write_finding (quantitative, honest, with a concrete next step) -> "
        f"set_hypothesis_status. Then choose the next experiment. NEVER invent "
        f"metrics — every number must come from run_backtest. Be quantitative and "
        f"honest about overfitting: weigh in-sample vs out-of-sample, treat tiny "
        f"Sharpe gaps as noise, and prefer edges that survive out of sample."
    )
    return base + directive
