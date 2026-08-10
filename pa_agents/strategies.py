"""Strategy roster + optimizer adapter.

Each *agent* is a distinct optimization mandate. Five ship by default, spanning
the risk/return spectrum so the dashboard shows them genuinely competing:

  * max-sharpe    — Mean-Variance, maximize risk-adjusted return (GPU cuOpt)
  * min-cvar      — Mean-CVaR, minimize tail risk at 95% (GPU cuOpt + cuML KDE)
  * risk-parity   — inverse-volatility contribution balance (CPU closed-form)
  * max-return    — Mean-Variance with a variance cap, return-seeking (GPU)
  * equal-weight  — 1/N benchmark (no solve) — the bar every agent must beat

The optimizer adapter (:func:`optimize`) is the single choke point that calls
into the NVIDIA blueprint (`src/`: ``MeanVariance``, ``CVaR``,
``generate_cvar_data``). It is written against the API mapped from the repo, and
is verified/adjusted on the GPU box where the package is actually importable.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


# --------------------------------------------------------------------------
# Agent roster
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentSpec:
    agent_id: str
    display_name: str
    strategy_type: str          # mean_variance | cvar | risk_parity | max_return | equal_weight
    objective: str
    engine: str                 # gpu | cpu
    params: dict
    color: str


ROSTER: list[AgentSpec] = [
    AgentSpec(
        "max-sharpe", "Max-Sharpe (Mean-Variance)", "mean_variance",
        "Maximize risk-adjusted return (Sharpe) via GPU Mean-Variance.",
        "gpu",
        {"risk_aversion": 1.0, "w_min": 0.0, "w_max": 0.10, "long_only": True,
         "scale_risk_aversion": False},
        "#76b900",  # NVIDIA green
    ),
    AgentSpec(
        "min-cvar", "Min-CVaR (Tail-Risk)", "cvar",
        "Minimize 95% Conditional Value-at-Risk over KDE scenarios (GPU).",
        "gpu",
        {"alpha": 0.95, "risk_aversion": 5.0, "w_min": 0.0, "w_max": 0.10,
         "num_scenarios": 3000, "kde_device": "GPU", "scale_risk_aversion": False},
        "#1f77b4",
    ),
    AgentSpec(
        "risk-parity", "Risk-Parity (Inverse-Vol)", "risk_parity",
        "Balance risk contribution across names (inverse-volatility).",
        "cpu",
        {"w_min": 0.0, "w_max": 0.10},
        "#9467bd",
    ),
    AgentSpec(
        "max-return", "Max-Return (Variance-Capped)", "max_return",
        "Maximize expected return subject to a variance cap (GPU).",
        "gpu",
        {"risk_aversion": 0.25, "w_min": 0.0, "w_max": 0.12,
         "scale_risk_aversion": False},
        "#d62728",
    ),
    AgentSpec(
        "equal-weight", "Equal-Weight (1/N Benchmark)", "equal_weight",
        "Naive 1/N allocation — the benchmark every strategy must beat.",
        "cpu",
        {},
        "#7f7f7f",
    ),
]

ROSTER_BY_ID = {a.agent_id: a for a in ROSTER}


# --------------------------------------------------------------------------
# Optimizer adapter
# --------------------------------------------------------------------------

@dataclass
class OptimizeResult:
    weights: dict[str, float]       # ticker -> weight (only nonzero)
    engine: str                     # gpu | cpu
    solve_ms: float
    scenario_ms: float
    num_scenarios: int
    metrics: dict                   # exp_return, volatility, sharpe, cvar, ...
    gpu_name: str | None = None


def _annualize_stats(weights: np.ndarray, mean_daily: np.ndarray,
                     cov_daily: np.ndarray) -> dict:
    ann = 252.0
    exp_ret = float(weights @ mean_daily) * ann
    var = float(weights @ cov_daily @ weights) * ann
    vol = float(np.sqrt(max(var, 1e-18)))
    sharpe = exp_ret / vol if vol > 1e-9 else 0.0
    return {"exp_return": exp_ret, "volatility": vol, "sharpe": sharpe}


def _pack_weights(tickers: list[str], w: np.ndarray, eps: float = 1e-5) -> dict[str, float]:
    return {t: float(wi) for t, wi in zip(tickers, w) if wi > eps}


def optimize(
    spec: AgentSpec,
    tickers: list[str],
    returns: np.ndarray,      # shape (T, N) daily simple returns, columns == tickers
    *,
    params: dict | None = None,
    seed: int = 42,
    force_cpu: bool = False,
) -> OptimizeResult:
    """Solve ``spec``'s optimization for the given return matrix.

    Returns target weights + timing + risk metrics. GPU strategies use the
    NVIDIA cuOpt/cuML path; on ImportError (no GPU) they fall back to the CVXPY
    CPU path so the fleet still runs anywhere. The heavy lifting for
    mean_variance / cvar / max_return is delegated to the NVIDIA optimizers.
    """
    import time

    p = {**spec.params, **(params or {})}
    N = len(tickers)
    mean_daily = returns.mean(axis=0)
    cov_daily = np.cov(returns, rowvar=False)
    t0 = time.perf_counter()
    scenario_ms = 0.0
    num_scen = 0
    gpu_name = None

    # ---- Equal-weight benchmark: no solve ----------------------------------
    if spec.strategy_type == "equal_weight":
        w = np.full(N, 1.0 / N)
        solve_ms = (time.perf_counter() - t0) * 1e3
        m = _annualize_stats(w, mean_daily, cov_daily)
        return OptimizeResult(_pack_weights(tickers, w), "cpu", solve_ms, 0.0, 0, m)

    # ---- Risk-parity (inverse-vol closed form) -----------------------------
    if spec.strategy_type == "risk_parity":
        vol = np.sqrt(np.clip(np.diag(cov_daily), 1e-12, None))
        inv = 1.0 / vol
        w = inv / inv.sum()
        wmax = p.get("w_max", 0.10)
        # simple cap-and-renormalize to respect concentration limit
        for _ in range(50):
            over = w > wmax
            if not over.any():
                break
            excess = (w[over] - wmax).sum()
            w[over] = wmax
            under = ~over
            if under.any():
                w[under] += excess * (w[under] / w[under].sum())
        w = w / w.sum()
        solve_ms = (time.perf_counter() - t0) * 1e3
        m = _annualize_stats(w, mean_daily, cov_daily)
        return OptimizeResult(_pack_weights(tickers, w), "cpu", solve_ms, 0.0, 0, m)

    # ---- GPU strategies via NVIDIA optimizers ------------------------------
    # NOTE: verified against the installed `src` package on the GPU box.
    # The adapter targets the mapped API:
    #   MeanVariance(returns_dict, params, api_settings) ; CVaR(...) ;
    #   generate_cvar_data(returns_dict, cvar_params, kde_settings)
    #   solve_optimization_problem(solver_settings) -> (row, portfolio)
    #   portfolio.weights (np.ndarray aligned to tickers)
    try:
        w, solve_ms2, scenario_ms, num_scen, gpu_name, extra = _solve_nvidia(
            spec, tickers, returns, mean_daily, cov_daily, p, seed, force_cpu
        )
        solve_ms = solve_ms2
        # Report the engine that ACTUALLY ran: gpu_name is set only when the
        # cuOpt/cuML GPU path executed. The adapter returns gpu_name=None when
        # it had to honestly route to CPU (e.g. SOCP var-cap on cuOpt 26.4).
        engine = "gpu" if (gpu_name and not force_cpu) else "cpu"
        m = _annualize_stats(w, mean_daily, cov_daily)
        m.update(extra)
        return OptimizeResult(_pack_weights(tickers, w), engine, solve_ms,
                              scenario_ms, num_scen, m, gpu_name)
    except Exception as exc:  # pragma: no cover - defensive fallback
        # A GENUINE failure of the NVIDIA path (import/solve). Surface it loudly
        # so a silent CPU fallback can't masquerade as a working GPU run, then
        # fall back to CVXPY so a run never dies outright.
        import logging
        logging.getLogger(__name__).warning(
            "NVIDIA GPU solve failed for %s (%s: %s) -- falling back to CVXPY CPU",
            spec.agent_id, type(exc).__name__, str(exc)[:300],
        )
        w, solve_ms = _solve_cvxpy_meanvar(mean_daily, cov_daily, p)
        m = _annualize_stats(w, mean_daily, cov_daily)
        m["fallback"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        return OptimizeResult(_pack_weights(tickers, w), "cpu", solve_ms, 0.0, 0, m)


def _solve_cvxpy_meanvar(mean_daily, cov_daily, p) -> tuple[np.ndarray, float]:
    """Pure-CPU mean-variance QP via CVXPY (fallback + risk_parity-free path)."""
    import time
    import cvxpy as cp

    t0 = time.perf_counter()
    N = len(mean_daily)
    w = cp.Variable(N)
    ra = float(p.get("risk_aversion", 1.0))
    wmax = float(p.get("w_max", 0.10))
    wmin = float(p.get("w_min", 0.0))
    ann = 252.0
    ret = mean_daily @ w * ann
    risk = cp.quad_form(w, cp.psd_wrap(cov_daily * ann))
    prob = cp.Problem(cp.Maximize(ret - ra * risk),
                      [cp.sum(w) == 1, w >= wmin, w <= wmax])
    prob.solve(solver=cp.CLARABEL)
    wv = np.clip(np.asarray(w.value).flatten(), 0, None)
    wv = wv / wv.sum() if wv.sum() > 0 else np.full(N, 1.0 / N)
    return wv, (time.perf_counter() - t0) * 1e3


# The NVIDIA-specific solve is defined in nvidia_adapter.py so it can be
# swapped/verified in isolation on the GPU box without touching this module.
from .nvidia_adapter import solve_nvidia as _solve_nvidia  # noqa: E402
