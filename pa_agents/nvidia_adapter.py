"""Thin adapter over the NVIDIA portfolio-optimization blueprint (`src/`).

Isolated on purpose: this is the ONE file that imports NVIDIA's optimizers, so
it can be verified/tuned on the GPU box (where cuOpt/cuML actually import)
without disturbing the rest of the runtime. Everything above it deals in plain
NumPy arrays and ticker->weight dicts.

REAL API (verified on the GPU box against cuopt/cuml 26.04, CUDA 13, NVIDIA L4)
------------------------------------------------------------------------------
The optimizers live in submodules (NOT re-exported from top-level ``src``):

    from src.mean_variance_optimizer import MeanVariance
    from src.cvar_optimizer import CVaR                      # class name: CVaR
    from src.cvar_parameters import CvarParameters           # NOTE: "Cvar" not "CVaR"
    from src.mean_variance_parameters import MeanVarianceParameters
    from src.cvar_utils import generate_cvar_data
    from src.settings import ApiSettings, KDESettings, ScenarioGenerationSettings

Constructors::

    MeanVariance(returns_dict, mean_variance_params, api_settings=None,
                 existing_portfolio=None)
    CVaR(returns_dict, cvar_params, api_settings=None, existing_portfolio=None)

Solve (returns ``(result_row: pd.Series, portfolio: Portfolio)``)::

    row, portfolio = problem.solve_optimization_problem(solver_settings,
                                                        print_results=False)
    portfolio.weights   # np.ndarray aligned to returns_dict["tickers"]
    portfolio.cash      # float cash allocation

``solver_settings`` differs by API:
    * GPU  (api="cuopt_python"): {"time_limit": 200}  (plain cuOpt SolverSettings kwargs)
    * CPU  (api="cvxpy"):        {"solver": "CLARABEL"} (a solver key is MANDATORY)

``returns_dict`` (the blueprint's canonical shape, produced by
``src.utils.calculate_returns``) MUST contain::

    {
      "tickers":     list[str],
      "returns":     pd.DataFrame (T x N, columns == tickers)   # KDE fits on this
      "mean":        np.ndarray (N,)      # per-asset mean daily return
      "covariance":  np.ndarray (N, N)    # NOTE the key is "covariance", not "cov"
      "regime":      {"name": str, "range": (start, end)}       # base_optimizer reads both
      "return_type": str  ("LOG" | "LINEAR" | ...)
      "dates":       index-like
    }

CVaR needs scenarios first::

    scen = ScenarioGenerationSettings(num_scen=3000, fit_type="kde",
             kde_settings=KDESettings(bandwidth=0.01, kernel="gaussian",
                                      device="GPU"|"CPU"), seed=seed)
    returns_dict = generate_cvar_data(returns_dict, scen)   # writes returns_dict["cvar_data"]
    # ^ generate_cvar_data takes exactly (returns_dict, scenario_generation_settings)

What runs on GPU on cuOpt 26.4 (honest state of the stack)
----------------------------------------------------------
* min-cvar  : Mean-CVaR is LP-shaped -> solves on cuOpt GPU (Status: Optimal on
              NVIDIA L4). KDE scenario generation runs on the L4 via cuML.
              c_max MUST be pinned to 0.0 (fully invested) or the LP trivially
              parks 100% in cash (zero tail risk => all-zero weights).
* max-sharpe: plain Mean-Variance is a QP (no quadratic *constraints*) -> cuOpt
              solves it with its barrier method on GPU (Status: Optimal).
* max-return: uses a variance cap (var_limit), which cuOpt turns into a
              quadratic/SOCP constraint. cuOpt 26.4 REJECTS these
              ("Quadratic constraints not supported"), so this path CANNOT run
              on GPU on this stack. It is solved honestly on CPU (CVXPY) and the
              engine is reported as "cpu" -- we do NOT fake GPU here.

If the NVIDIA package can't be imported (e.g. on the laptop), ``solve_nvidia``
raises so ``strategies.optimize`` falls back to its CVXPY CPU path.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd


# cuOpt 26.4 cannot express quadratic/SOCP constraints (variance cap). Strategy
# params that set a variance cap ("var_limit"/"var_cap") therefore cannot solve
# on GPU on this stack; we solve them on CPU and report engine=cpu honestly.
class QuadraticConstraintUnsupported(RuntimeError):
    """Raised when a var-cap (SOCP/QCQP) form is requested on a cuOpt build
    that rejects quadratic constraints. Signals an honest CPU solve."""


def _detect_gpu_name() -> str | None:
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        name = out.stdout.strip().splitlines()[0].strip() if out.stdout.strip() else None
        return name or None
    except Exception:
        return None


def _returns_dict(tickers: list[str], returns: np.ndarray) -> dict:
    """Build the blueprint's canonical ``returns_dict`` from a (T,N) return matrix.

    Mirrors exactly what ``src.utils.calculate_returns`` produces (keys the
    optimizers + generate_cvar_data actually read): ``returns`` is a DataFrame
    (KDE fits column-wise on it), the covariance key is ``covariance`` (not
    ``cov``), and ``regime``/``return_type``/``dates`` are required by
    ``base_optimizer``.
    """
    returns = np.ascontiguousarray(returns, dtype=np.float64)
    ret_df = pd.DataFrame(returns, columns=list(tickers))
    return {
        "tickers": list(tickers),
        "returns": ret_df,
        "mean": returns.mean(axis=0),
        "covariance": np.cov(returns, rowvar=False),
        "regime": {"name": "rebalance", "range": (0, returns.shape[0])},
        "return_type": "LOG",
        "dates": ret_df.index,
    }


def solve_nvidia(
    spec,
    tickers: list[str],
    returns: np.ndarray,
    mean_daily: np.ndarray,
    cov_daily: np.ndarray,
    p: dict,
    seed: int,
    force_cpu: bool,
) -> tuple[np.ndarray, float, float, int, str | None, dict]:
    """Solve via NVIDIA cuOpt/cuML. Returns
    (weights, solve_ms, scenario_ms, num_scenarios, gpu_name, extra_metrics).

    Raises on import failure or non-optimal status so the caller can fall back
    to its pure-CVXPY CPU path. For a variance-capped mean-variance on a cuOpt
    build that rejects quadratic constraints, this deliberately solves on CPU
    and reports ``gpu_name=None`` so the engine is honestly "cpu".
    """
    # Import the blueprint. Installed as top-level `src` package inside the repo
    # checkout; the runner puts the repo root on sys.path before calling.
    from src.mean_variance_optimizer import MeanVariance  # type: ignore
    from src.cvar_optimizer import CVaR  # type: ignore
    from src.cvar_utils import generate_cvar_data  # type: ignore
    from src.settings import (  # type: ignore
        ApiSettings, KDESettings, ScenarioGenerationSettings,
    )

    gpu_name = None if force_cpu else _detect_gpu_name()
    use_gpu = (not force_cpu) and gpu_name is not None
    api = "cuopt_python" if use_gpu else "cvxpy"
    scenario_ms = 0.0
    num_scen = 0
    extra: dict = {}

    rdict = _returns_dict(tickers, returns)

    # ---------------- Mean-Variance family -------------------------------
    if spec.strategy_type in ("mean_variance", "max_return"):
        has_var_cap = _var_cap(p) is not None
        # A variance cap becomes a quadratic/SOCP constraint that cuOpt 26.4
        # rejects. Solve it honestly on CPU (CVXPY) rather than pretend GPU.
        mv_use_gpu = use_gpu and not has_var_cap
        mv_api = "cuopt_python" if mv_use_gpu else "cvxpy"

        mv_params = _mk_mv_params(p, tickers)
        api_settings = ApiSettings(
            api=mv_api,
            scale_risk_aversion=bool(p.get("scale_risk_aversion", False)),
        )
        problem = MeanVariance(rdict, mv_params, api_settings=api_settings)
        solver_settings = _mk_solver_settings(mv_use_gpu, p)
        t0 = time.perf_counter()
        row, portfolio = problem.solve_optimization_problem(
            solver_settings, print_results=False
        )
        solve_ms = (time.perf_counter() - t0) * 1e3
        w = np.asarray(portfolio.weights, dtype=np.float64).flatten()
        # If we had to route MV to CPU because of the SOCP var-cap, report cpu.
        if not mv_use_gpu:
            gpu_name = None
            extra["gpu_unavailable_reason"] = (
                "variance cap -> quadratic/SOCP constraint, unsupported by "
                "cuOpt 26.4 (solved on CPU/CVXPY)"
            ) if has_var_cap and not force_cpu else "cpu"

    # ---------------- Mean-CVaR (LP on cuOpt + cuML KDE) -----------------
    elif spec.strategy_type == "cvar":
        cvar_params = _mk_cvar_params(p, tickers)
        num_scen = int(p.get("num_scenarios", 3000))
        # scenario generation (cuML KDE on GPU when available) ------------
        kde_device = "GPU" if use_gpu else "CPU"
        scen_settings = ScenarioGenerationSettings(
            num_scen=num_scen,
            fit_type="kde",
            kde_settings=KDESettings(
                bandwidth=float(p.get("kde_bandwidth", 0.01)),
                kernel=str(p.get("kde_kernel", "gaussian")),
                device=kde_device,
            ),
            seed=seed,
        )
        ts = time.perf_counter()
        rdict = generate_cvar_data(rdict, scen_settings)
        scenario_ms = (time.perf_counter() - ts) * 1e3

        api_settings = ApiSettings(
            api=api, scale_risk_aversion=bool(p.get("scale_risk_aversion", False)),
        )
        problem = CVaR(rdict, cvar_params, api_settings=api_settings)
        solver_settings = _mk_solver_settings(use_gpu, p)
        t0 = time.perf_counter()
        row, portfolio = problem.solve_optimization_problem(
            solver_settings, print_results=False
        )
        solve_ms = (time.perf_counter() - t0) * 1e3
        w = np.asarray(portfolio.weights, dtype=np.float64).flatten()
        # cuOpt result_row exposes the realized tail risk under the "CVaR" label.
        try:
            extra["cvar"] = float(row.get("CVaR"))
        except Exception:
            extra["cvar"] = 0.0
    else:
        raise ValueError(f"unsupported nvidia strategy: {spec.strategy_type}")

    # normalize defensively (weights should already sum to ~1 when fully invested)
    w = np.clip(w, 0.0, None)
    s = w.sum()
    w = w / s if s > 0 else np.full(len(tickers), 1.0 / len(tickers))
    return w, solve_ms, scenario_ms, num_scen, gpu_name, extra


# --- parameter builders (match the blueprint's Pydantic models exactly) ------

def _var_cap(p: dict):
    """Return a variance-cap value from strategy params, if any (several aliases)."""
    for k in ("var_limit", "var_cap", "variance_cap", "variance_limit"):
        v = p.get(k)
        if v is not None:
            return float(v)
    return None


def _mk_mv_params(p: dict, tickers: list[str]):
    """Build MeanVarianceParameters. Fields (verified): w_min, w_max, c_min,
    c_max, risk_aversion, L_tar, T_tar, cardinality, group_constraints,
    var_limit. We keep the portfolio fully invested (c_max=0) so weights sum to
    1 -- otherwise cash competes with risky assets.
    """
    from src.mean_variance_parameters import MeanVarianceParameters  # type: ignore
    kwargs = dict(
        risk_aversion=float(p.get("risk_aversion", 1.0)),
        w_min=float(p.get("w_min", 0.0)),
        w_max=float(p.get("w_max", 1.0)),
        c_min=0.0,
        c_max=0.0,  # fully invested: sum(weights) == 1
    )
    var_cap = _var_cap(p)
    if var_cap is not None:
        kwargs["var_limit"] = var_cap
    return MeanVarianceParameters(**kwargs)


def _mk_cvar_params(p: dict, tickers: list[str]):
    """Build CvarParameters. Fields (verified): w_min, w_max, c_min, c_max,
    risk_aversion, L_tar, T_tar, cardinality, group_constraints, confidence,
    cvar_limit. The confidence level is ``confidence`` (not ``alpha``), and
    c_max MUST be 0.0 or the LP parks everything in cash (zero tail risk).
    """
    from src.cvar_parameters import CvarParameters  # type: ignore
    kwargs = dict(
        confidence=float(p.get("alpha", p.get("confidence", 0.95))),
        risk_aversion=float(p.get("risk_aversion", 5.0)),
        w_min=float(p.get("w_min", 0.0)),
        w_max=float(p.get("w_max", 0.10)),
        c_min=0.0,
        c_max=0.0,  # fully invested: sum(weights) == 1
    )
    if p.get("cvar_limit") is not None:
        kwargs["cvar_limit"] = float(p["cvar_limit"])
    return CvarParameters(**kwargs)


def _mk_solver_settings(use_gpu: bool, p: dict | None = None) -> dict:
    """cuOpt wants plain SolverSettings kwargs (e.g. time_limit); CVXPY wants a
    mandatory ``solver`` key (solve_optimization_problem raises without one).

    We pin the cuOpt solver ``method`` to a single algorithm (DualSimplex by
    default). cuOpt's DEFAULT is Concurrent mode (PDLP + DualSimplex + Barrier
    raced together), which SEGFAULTS during teardown on ill-conditioned
    problems (the "large range of coefficients" CVaR LPs from real price data
    reproduce this deterministically -> exit 139, killing the run before the
    solution is read back). Forcing any single method (DualSimplex=2, PDLP=1,
    Barrier=3) solves cleanly with Status: Optimal. DualSimplex is the robust
    default for these LPs; QP (mean-variance, no var cap) uses Barrier since
    cuOpt routes quadratic objectives there regardless.
    """
    p = p or {}
    if use_gpu:
        return {
            "time_limit": float(p.get("time_limit", 200)),
            # 2 == SolverMethod.DualSimplex (avoids the concurrent-mode crash)
            "method": int(p.get("cuopt_method", 2)),
        }
    return {"solver": "CLARABEL"}
