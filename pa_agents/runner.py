"""The memory-driven agent loop — the heart of the demo.

For each rebalance date, every agent:
  1. RECALLS its most relevant persisted memories (semantic vector search over
     SingleStore) — literally re-reading what it learned on prior runs.
  2. LOADS its current best hyperparameters (strategy_params, versioned).
  3. SOLVES for target weights via the NVIDIA optimizer (GPU cuOpt / cuML).
  4. TRADES: translates weights into Goldman-style orders/fills/positions/NAV.
  5. REFLECTS: asks Claude to distill the outcome into a natural-language
     learning, embeds it, and WRITES it back to agent_memory + audit.

Because every artifact lives in SingleStore, an agent that is killed and
restarted a week later resumes with its full memory and book intact. That is the
"truly persisted agent memory" thesis, running against real GPU-optimized trades.
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from . import db, llm
from . import risk_gate
from .strategies import AgentSpec, ROSTER, ROSTER_BY_ID, optimize
from .trading import CostModel, rebalance


def _uid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def ensure_repo_on_path(repo_root: str | None = None) -> None:
    """Put the NVIDIA blueprint checkout on sys.path so `import src` works."""
    root = repo_root or str(Path.home() / "portfolio-optimization")
    if root not in sys.path:
        sys.path.insert(0, root)


def register_agents() -> None:
    """Upsert the agent roster + an initial strategy_params version each."""
    for a in ROSTER:
        db.execute(
            """INSERT INTO agents
               (agent_id, display_name, strategy_type, objective, engine,
                default_params, color, status, created_at)
               VALUES (%s,%s,%s,%s,%s,%s,%s,'active',NOW(6))
               ON DUPLICATE KEY UPDATE display_name=VALUES(display_name),
                 objective=VALUES(objective), engine=VALUES(engine),
                 default_params=VALUES(default_params), color=VALUES(color)""",
            (a.agent_id, a.display_name, a.strategy_type, a.objective, a.engine,
             json.dumps(a.params), a.color),
        )
        existing = db.query(
            "SELECT param_id FROM strategy_params WHERE agent_id=%s AND is_current=1",
            (a.agent_id,),
        )
        if not existing:
            db.execute(
                """INSERT INTO strategy_params
                   (param_id, agent_id, version, params, rationale, is_current, created_at)
                   VALUES (%s,%s,1,%s,%s,1,NOW(6))""",
                (_uid("prm"), a.agent_id, json.dumps(a.params),
                 "Initial default parameters."),
            )


def current_params(agent_id: str) -> dict:
    rows = db.query(
        "SELECT params FROM strategy_params WHERE agent_id=%s AND is_current=1 "
        "ORDER BY version DESC LIMIT 1",
        (agent_id,),
    )
    if rows:
        p = rows[0]["params"]
        return json.loads(p) if isinstance(p, str) else p
    return ROSTER_BY_ID[agent_id].params


def load_returns(tickers: list[str], start: str, end: str) -> tuple[list[str], np.ndarray, dict]:
    """Load the (T,N) daily-return matrix + as-of prices from SingleStore.

    Returns (tickers_kept, returns[T,N], last_price{ticker->px}). Only tickers
    with a full history over the window are kept, so the matrix is dense.
    """
    rows = db.query(
        """SELECT ticker, trade_date, adj_close, daily_return
           FROM prices
           WHERE trade_date BETWEEN %s AND %s AND ticker IN (%s)
           ORDER BY trade_date, ticker""" % (
            "%s", "%s", ",".join(["%s"] * len(tickers))
        ),
        [start, end, *tickers],
    )
    by_ticker: dict[str, list] = {}
    dates: set = set()
    px_last: dict[str, float] = {}
    for r in rows:
        by_ticker.setdefault(r["ticker"], []).append(r)
        dates.add(r["trade_date"])
    n_dates = len(dates)
    kept = [t for t in tickers if len(by_ticker.get(t, [])) >= n_dates * 0.98 and n_dates > 5]
    kept.sort()
    # build aligned return matrix
    date_list = sorted(dates)
    date_idx = {d: i for i, d in enumerate(date_list)}
    mat = np.full((len(date_list), len(kept)), np.nan)
    for j, t in enumerate(kept):
        for r in by_ticker[t]:
            i = date_idx[r["trade_date"]]
            mat[i, j] = r["daily_return"] if r["daily_return"] is not None else 0.0
            px_last[t] = float(r["adj_close"])
    # forward/zero fill any gaps, drop the first row (no return)
    mat = np.nan_to_num(mat, nan=0.0)[1:]
    return kept, mat, px_last


def run_rebalance(
    agent_id: str,
    *,
    as_of_date: str,
    lookback_start: str,
    lookback_end: str,
    universe: list[str],
    prices: dict[str, float] | None = None,
    starting_nav: float | None = None,
    do_reflect: bool = True,
    enforce_risk: bool = False,
    risk_limits: "risk_gate.RiskLimits | None" = None,
) -> dict:
    """Run one full memory->solve->trade->reflect cycle for a single agent.

    ``enforce_risk`` (default False) inserts the pre-trade :mod:`risk_gate`
    between SOLVE and TRADE. It is OFF by default so a historical backtest replay
    is unaffected (the gate's drawdown/turnover/loss checks would otherwise block
    legitimate simulated rebalances and corrupt the equity curve). Turn it ON for
    a *gated* replay or any forward/real path: a rejected rebalance is skipped
    (no orders) and audited, and the run is marked ``blocked``. When the gate
    clips weights (over-weight names), the adjusted vector is what gets traded.
    """
    spec = ROSTER_BY_ID[agent_id]
    run_id = _uid(f"run-{agent_id}")
    started = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")

    # 1) RECALL relevant memory (semantic) ----------------------------------
    recall_q = (f"{spec.display_name}: rebalance decision on {as_of_date}. "
                f"What did I learn about {spec.strategy_type} risk, turnover, "
                f"drawdown and parameter choices in prior runs?")
    try:
        memories = db.recall_memory(agent_id, recall_q, k=5)
    except Exception:
        memories = []
    memory_context = "\n".join(f"- [{m['kind']}] {m['content']}" for m in memories)

    # 2) LOAD current params -------------------------------------------------
    params = current_params(agent_id)

    # open the run record
    tickers, returns, px = load_returns(universe, lookback_start, lookback_end)
    prices = prices or px
    db.execute(
        """INSERT INTO agent_runs
           (run_id, agent_id, run_type, universe_size, lookback_start, lookback_end,
            as_of_date, params, engine, status, started_at)
           VALUES (%s,%s,'rebalance',%s,%s,%s,%s,%s,%s,'running',%s)""",
        (run_id, agent_id, len(tickers), lookback_start, lookback_end, as_of_date,
         json.dumps(params), spec.engine, started),
    )
    db.audit(_uid("aud"), agent_id, "RUN_START", run_id=run_id,
             detail={"as_of_date": as_of_date, "universe": len(tickers),
                     "recalled_memories": len(memories)})

    # 3) SOLVE ---------------------------------------------------------------
    t_solve = time.perf_counter()
    try:
        res = optimize(spec, tickers, returns, params=params)
    except Exception as exc:
        db.execute(
            "UPDATE agent_runs SET status='failed', error=%s, finished_at=NOW(6) WHERE run_id=%s",
            (str(exc)[:500], run_id),
        )
        db.audit(_uid("aud"), agent_id, "ERROR", run_id=run_id, detail={"error": str(exc)[:500]})
        raise
    db.audit(_uid("aud"), agent_id, "SOLVE", run_id=run_id,
             detail={"engine": res.engine, "solve_ms": res.solve_ms,
                     "scenario_ms": res.scenario_ms, "num_scenarios": res.num_scenarios,
                     "n_weights": len(res.weights), "metrics": res.metrics})

    # 3.5) RISK GATE (opt-in) — pre-trade check between SOLVE and TRADE.
    # Fail-closed: a rejected rebalance places NO orders and is audited.
    trade_weights = res.weights
    if enforce_risk:
        # On inception (no nav_history yet) supply starting_nav so the gate isn't
        # forced to fail-closed on the very first trade; an established agent's
        # live marked NAV is read from the book by the gate itself.
        has_hist = db.query(
            "SELECT 1 FROM nav_history WHERE agent_id=%s LIMIT 1", (agent_id,))
        seed_nav = None if has_hist else (starting_nav or 100_000_000.0)
        gate_limits = risk_limits or risk_gate.RiskLimits()
        if not has_hist:
            # Inception funding (cash -> invested) is 100% one-way turnover by
            # construction, not churn; its size is already bound by the gross
            # cap. Relax only the turnover limit for the very first trade so the
            # gate doesn't reject legitimate initial deployment.
            import dataclasses as _dc
            gate_limits = _dc.replace(gate_limits, max_turnover=gate_limits.max_gross_exposure)
        decision = risk_gate.evaluate(
            agent_id=agent_id, as_of_date=as_of_date, target_weights=res.weights,
            prices=prices, nav=seed_nav, limits=gate_limits, run_id=run_id, mode="live")
        if not decision.approved:
            db.execute(
                "UPDATE agent_runs SET status='blocked', error=%s, finished_at=NOW(6) WHERE run_id=%s",
                (decision.reason[:500], run_id))
            db.audit(_uid("aud"), agent_id, "RUN_END", run_id=run_id,
                     detail={"blocked": True, "reason": decision.reason,
                             "violations": decision.violations})
            return {"run_id": run_id, "agent_id": agent_id, "as_of_date": as_of_date,
                    "engine": res.engine, "gpu_name": res.gpu_name,
                    "solve_ms": res.solve_ms, "scenario_ms": res.scenario_ms,
                    "num_scenarios": res.num_scenarios, "n_orders": 0, "n_positions": 0,
                    "nav_after": None, "turnover": 0.0, "sharpe": res.metrics.get("sharpe"),
                    "recalled": len(memories), "wrote_learning": False,
                    "blocked": True, "reason": decision.reason,
                    "violations": decision.violations}
        # honor any gate-clipped weights (e.g. over-weight names capped + renormalized)
        trade_weights = decision.adjusted_weights or res.weights

    # 4) TRADE ---------------------------------------------------------------
    reb = rebalance(
        run_id=run_id, agent_id=agent_id, as_of_date=as_of_date,
        target_weights=trade_weights, prices=prices,
        cost_model=CostModel(), starting_nav=starting_nav,
    )

    # risk metrics row
    db.execute(
        """INSERT INTO risk_metrics
           (run_id, agent_id, as_of_date, exp_return, volatility, sharpe, cvar,
            turnover, n_positions, gross_exposure, net_exposure, created_at)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW(6))""",
        (run_id, agent_id, as_of_date,
         res.metrics.get("exp_return"), res.metrics.get("volatility"),
         res.metrics.get("sharpe"), res.metrics.get("cvar", 0.0),
         reb.turnover, reb.n_positions, 1.0, 1.0),
    )

    # close the run
    db.execute(
        """UPDATE agent_runs SET status='ok', engine=%s, gpu_name=%s,
           num_scenarios=%s, solve_ms=%s, scenario_ms=%s, finished_at=NOW(6)
           WHERE run_id=%s""",
        (res.engine, res.gpu_name, res.num_scenarios, res.solve_ms,
         res.scenario_ms, run_id),
    )

    # 5) REFLECT + WRITE MEMORY ---------------------------------------------
    obs = (f"On {as_of_date}, {spec.display_name} rebalanced {reb.n_orders} orders "
           f"across {reb.n_positions} names. NAV {reb.nav_before:,.0f}->{reb.nav_after:,.0f} "
           f"(turnover {reb.turnover:.1%}, cost ${reb.tcost:,.0f}). "
           f"Sharpe {res.metrics.get('sharpe', 0):.2f}, vol {res.metrics.get('volatility', 0):.1%}, "
           f"exp ret {res.metrics.get('exp_return', 0):.1%}, engine {res.engine}"
           f"{' on ' + res.gpu_name if res.gpu_name else ''}.")
    db.write_memory(
        _uid("mem"), agent_id, "observation", obs, run_id=run_id, as_of_date=as_of_date,
        importance=0.5,
        metrics={"sharpe": res.metrics.get("sharpe"), "vol": res.metrics.get("volatility"),
                 "turnover": reb.turnover, "nav": reb.nav_after,
                 "solve_ms": res.solve_ms, "engine": res.engine},
        tags=[spec.strategy_type, "rebalance"],
    )

    if do_reflect:
        prompt = (
            f"You are the '{spec.display_name}' portfolio strategy agent. Mandate: "
            f"{spec.objective}\n\nPast memories you recalled:\n{memory_context or '(none yet)'}\n\n"
            f"Today's outcome:\n{obs}\n\n"
            "In 2-3 sentences, write a durable LEARNING for your future self: what "
            "worked, what to watch (turnover, tail risk, concentration), and any "
            "parameter you'd nudge next time. Be specific and quantitative."
        )
        learning = llm.reflect(prompt)
        if learning:
            db.write_memory(
                _uid("mem"), agent_id, "learning", learning, run_id=run_id,
                as_of_date=as_of_date, importance=0.75,
                metrics={"sharpe": res.metrics.get("sharpe")},
                tags=[spec.strategy_type, "reflection"],
            )
            db.audit(_uid("aud"), agent_id, "MEMORY", run_id=run_id,
                     detail={"kind": "learning", "chars": len(learning)})

    return {
        "run_id": run_id, "agent_id": agent_id, "as_of_date": as_of_date,
        "engine": res.engine, "gpu_name": res.gpu_name,
        "solve_ms": res.solve_ms, "scenario_ms": res.scenario_ms,
        "num_scenarios": res.num_scenarios,
        "n_orders": reb.n_orders, "n_positions": reb.n_positions,
        "nav_after": reb.nav_after, "turnover": reb.turnover,
        "sharpe": res.metrics.get("sharpe"), "recalled": len(memories),
        "wrote_learning": bool(do_reflect),
    }
