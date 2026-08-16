"""Honest, overfitting-aware ranking of a thousands-config strategy sweep.

A companion engine backtests THOUSANDS of strategy configs on real S&P 500
prices, scoring each on an in-sample (IS) window and an out-of-sample (OOS)
window, and appends the rows to ``sweep_results``. With that many trials the
naive "highest Sharpe" is almost certainly a FALSE DISCOVERY: the best in-sample
Sharpe is inflated by selection across N trials, and it typically collapses out
of sample. This module turns the noisy pile into a defensible, real-money-cautious
ranking. It does five things and refuses to pretend otherwise:

1. RANK BY OOS, NOT IS. The headline ranking is ``oos_sharpe`` among rows with
   ``error IS NULL`` and non-null metrics. IS Sharpe is explicitly the WRONG
   thing to rank on (it is what every config was implicitly selected to maximize),
   and the report says so out loud.

2. OVERFITTING FLAGS per candidate:
     * ``is_oos_sharpe_gap`` = is_sharpe - oos_sharpe (already stored). Large
       positive gap = looked great in-sample, mediocre out = overfit.
     * ``robust`` boolean: the edge survived the walk forward. We require
       oos_sharpe > 0 AND the config beat its OOS benchmark AND the IS->OOS
       degradation is contained: gap <= ROBUST_MAX_GAP (default 1.0) AND
       oos_sharpe >= ROBUST_RETENTION * is_sharpe (default 0.5, i.e. it kept at
       least half its in-sample Sharpe). Thresholds are module constants and are
       documented below.

3. MULTIPLE-TESTING / FALSE-DISCOVERY CORRECTION (deflated-Sharpe style). See
   ``mtc_threshold`` for the full derivation. In one line: with N trials the
   expected MAX Sharpe under the null (true edge = 0) is inflated, so we compute
   the Sharpe a strategy must clear to be distinguishable from the luckiest of N
   coin-flips at a given family-wise confidence, and flag everything below it as
   "likely noise". KEY ASSUMPTION (stated in the output): the N trials are treated
   as INDEPENDENT, which is OPTIMISTIC — real sweeps share overlapping signals, so
   the effective number of independent trials is smaller and the true hurdle is a
   touch lower; treating them as independent makes the test slightly CONSERVATIVE
   (harder to pass), which is the safe direction for a real-money gate.

4. FAMILY-LEVEL VIEW: per family, trial count, median/best OOS Sharpe, and median
   IS->OOS gap. A family whose edge is STABLE across many configs is far more
   trustworthy than one lucky config; a family with a big median gap overfits.

5. NAIVE-PICK CROSS-CHECK: show the config with the best IS Sharpe and where it
   lands in the OOS ranking. The IS->OOS collapse is the money-shot for the
   real-money-caution narrative.

Output: a printed report (ranked tables + the narrative), a persisted compact
summary row in ``sweep_analysis``, and the top-K (default 25) emitted as JSON to
stdout so an orchestrator can consume it.

Run from the demo root:
    python -m research_fleet.research_agent.sweep_analyze --latest
    python -m research_fleet.research_agent.sweep_analyze --sweep-id <id> --top 25
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import uuid
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any

from . import research_db as rdb

# --------------------------------------------------------------------------
# Tunable thresholds (documented; conservative for a real-money gate)
# --------------------------------------------------------------------------

# Robustness gate (flag #2).
ROBUST_MAX_GAP = 1.0        # max tolerated IS->OOS Sharpe degradation (is_sharpe - oos_sharpe)
ROBUST_RETENTION = 0.5      # OOS Sharpe must retain >= this fraction of IS Sharpe
ROBUST_MIN_OOS_SHARPE = 0.5 # a floor: a "robust" edge must be at least this good OOS

# Multiple-testing correction (flag #3).
MTC_CONFIDENCE = 0.95       # family-wise confidence: P(max of N nulls < hurdle)
DEFAULT_TRADING_DAYS = 252  # periods/year, used to turn an annualized Sharpe into a t-stat


# --------------------------------------------------------------------------
# Normal CDF / inverse-CDF (closed-form; scipy fast-path if available)
# --------------------------------------------------------------------------

try:  # scipy is optional in this venv; fall back to closed forms if absent.
    from scipy.stats import norm as _norm  # type: ignore

    def _norm_cdf(x: float) -> float:
        return float(_norm.cdf(x))

    def _norm_ppf(p: float) -> float:
        return float(_norm.ppf(p))

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - exercised only where scipy is missing
    _HAVE_SCIPY = False

    def _norm_cdf(x: float) -> float:
        """Standard-normal CDF via the erf identity (exact to float precision)."""
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def _norm_ppf(p: float) -> float:
        """Inverse standard-normal CDF (Acklam's rational approximation, ~1e-9)."""
        if not 0.0 < p < 1.0:
            if p <= 0.0:
                return -math.inf
            if p >= 1.0:
                return math.inf
        a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
             1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
        b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
             6.680131188771972e+01, -1.328068155288572e+01]
        c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
             -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
        d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
             3.754408661907416e+00]
        plow, phigh = 0.02425, 1 - 0.02425
        if p < plow:
            q = math.sqrt(-2 * math.log(p))
            return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                   ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        if p > phigh:
            q = math.sqrt(-2 * math.log(1 - p))
            return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                    ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def _uid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


# --------------------------------------------------------------------------
# The multiple-testing / deflated-Sharpe hurdle (the statistical heart)
# --------------------------------------------------------------------------

def expected_max_z(n_trials: int) -> float:
    """E[max of N iid standard normals] — the selection bias, in std-dev units.

    If you draw N Sharpe-like statistics whose TRUE mean is zero (no edge) and
    standardize them, the single BEST one is not ~0 — it is pulled up simply by
    taking the max of N draws. The classic extreme-value approximation is

        E[max] ~= (1 - g) * z_{1 - 1/N}  +  g * z_{1 - 1/(N*e)}

    (Bailey & Lopez de Prado's deflated-Sharpe form, g = Euler-Mascheroni), which
    for large N tracks sqrt(2*ln N). This is the amount by which the luckiest of N
    zero-edge trials is expected to shine purely by chance. We report both this
    refined value and the sqrt(2*ln N) shorthand mentioned in the brief.
    """
    n = max(int(n_trials), 1)
    if n == 1:
        return 0.0
    gamma = 0.5772156649015329  # Euler-Mascheroni
    z1 = _norm_ppf(1.0 - 1.0 / n)
    z2 = _norm_ppf(1.0 - 1.0 / (n * math.e))
    return (1.0 - gamma) * z1 + gamma * z2


def sharpe_se(n_periods: int) -> float:
    """Standard error of an (annualized) Sharpe estimate ~= sqrt(1/n_periods).

    Under iid returns the sampling SD of the Sharpe ratio is approximately
    sqrt((1 + 0.5*SR^2)/n). For a real-money GATE we use the conservative,
    SR-independent floor sqrt(1/n): it is the value at SR=0 (the null), it does
    not shrink the SE using the very quantity under test, and larger n (more OOS
    days) tightens it. `n_periods` here is the number of OOS return periods.
    """
    n = max(int(n_periods), 2)
    return math.sqrt(1.0 / n)


def mtc_threshold(n_trials: int, n_periods: int, *, confidence: float = MTC_CONFIDENCE) -> dict:
    """Minimum OOS Sharpe a config must clear to beat the luckiest of N nulls.

    Construction (deflated-Sharpe / Bonferroni-on-the-tail logic):

      1. A Sharpe estimated over ``n_periods`` OOS periods has standard error
         ``se = sharpe_se(n_periods)`` (~ sqrt(1/n)).  Its t-stat vs. the null
         (true Sharpe = 0) is ``t = oos_sharpe / se``, which is ~ standard normal.
      2. Across N independent trials, the BEST t-stat under the null is not 0 but
         ``E[max]`` (see ``expected_max_z``).  We add a confidence cushion so we
         are not merely at the *expected* max but above it with probability
         ``confidence``: the (1 - 1/N)-style tail already encodes the max; we
         further require clearing the ``confidence`` quantile of that maximum,
         ``z_gate = E[max] + z_{confidence}``  (a Gumbel-tail cushion).
      3. Convert that standardized hurdle back to Sharpe units:
         ``sharpe_hurdle = z_gate * se``.

    A config's OOS Sharpe clears the correction iff ``oos_sharpe >= sharpe_hurdle``.

    KEY ASSUMPTION: the N trials are treated as INDEPENDENT. That is OPTIMISTIC —
    a real sweep reuses overlapping signals/windows, so the number of *effectively
    independent* trials is smaller than N and the true hurdle is somewhat lower.
    Treating them as independent inflates N, raising the bar, so the test is
    slightly CONSERVATIVE (it may reject a real but marginal edge before it ships
    real money — the safe direction to err).
    """
    n = max(int(n_trials), 1)
    se = sharpe_se(n_periods)
    e_max = expected_max_z(n)
    cushion = _norm_ppf(confidence) if 0.0 < confidence < 1.0 else 0.0
    z_gate = e_max + cushion
    sharpe_hurdle = z_gate * se
    return {
        "n_trials": n,
        "n_periods": int(n_periods),
        "sharpe_se": se,
        "expected_max_z": e_max,
        "sqrt_2lnN": math.sqrt(2.0 * math.log(n)) if n > 1 else 0.0,
        "confidence": confidence,
        "z_cushion": cushion,
        "z_gate": z_gate,
        "sharpe_hurdle": sharpe_hurdle,
        "used_scipy": _HAVE_SCIPY,
    }


# --------------------------------------------------------------------------
# Data loading + normalization
# --------------------------------------------------------------------------

def _as_float(x) -> float | None:
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _as_params(p) -> dict:
    if isinstance(p, dict):
        return p
    if not p:
        return {}
    try:
        v = json.loads(p)
        return v if isinstance(v, dict) else {"value": v}
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


def latest_sweep_id() -> str | None:
    """The most recently STARTED sweep_runs row (default when --sweep-id omitted)."""
    rows = rdb.query(
        "SELECT sweep_id FROM sweep_runs ORDER BY started_at DESC, sweep_id DESC LIMIT 1")
    if rows:
        return rows[0]["sweep_id"]
    # fall back to sweep_results if sweep_runs is empty (engine may write results first)
    rows = rdb.query(
        "SELECT sweep_id FROM sweep_results WHERE sweep_id IS NOT NULL "
        "GROUP BY sweep_id ORDER BY MAX(created_at) DESC LIMIT 1")
    return rows[0]["sweep_id"] if rows else None


def _oos_periods_for(sweep_id: str) -> int:
    """Number of OOS return periods, from the sweep_runs window if present.

    Used to turn an annualized OOS Sharpe into a t-stat. Falls back to a single
    year of trading days when the window is unknown.
    """
    rows = rdb.query(
        "SELECT oos_start, oos_end FROM sweep_runs WHERE sweep_id=%s", (sweep_id,))
    if rows and rows[0].get("oos_start") and rows[0].get("oos_end"):
        s, e = rows[0]["oos_start"], rows[0]["oos_end"]
        try:
            days = (e - s).days
            # ~252 trading days per 365 calendar days
            periods = int(round(days * DEFAULT_TRADING_DAYS / 365.0))
            if periods >= 20:
                return periods
        except (TypeError, AttributeError):
            pass
    return DEFAULT_TRADING_DAYS


def load_valid_rows(sweep_id: str) -> list[dict]:
    """All ranked-eligible rows for a sweep: error IS NULL and OOS Sharpe present.

    IS-Sharpe presence is also required so the naive cross-check + gap are defined.
    Returns normalized dicts (floats coerced, params parsed).
    """
    raw = rdb.query(
        """SELECT result_id, sweep_id, family, params, is_sharpe, is_ann_return,
                  is_ann_vol, is_max_drawdown, is_turnover, is_beats_benchmark,
                  oos_sharpe, oos_ann_return, oos_ann_vol, oos_max_drawdown,
                  oos_turnover, oos_beats_benchmark, is_oos_sharpe_gap,
                  all_in_cost_bps, universe_n, error
           FROM sweep_results
           WHERE sweep_id=%s AND error IS NULL
             AND oos_sharpe IS NOT NULL AND is_sharpe IS NOT NULL
           ORDER BY oos_sharpe DESC""",
        (sweep_id,))
    out: list[dict] = []
    for r in raw:
        is_sharpe = _as_float(r["is_sharpe"])
        oos_sharpe = _as_float(r["oos_sharpe"])
        if is_sharpe is None or oos_sharpe is None:
            continue
        gap = _as_float(r["is_oos_sharpe_gap"])
        if gap is None:  # recompute if the engine left it null
            gap = is_sharpe - oos_sharpe
        out.append({
            "result_id": r["result_id"],
            "family": r["family"] or "unknown",
            "params": _as_params(r["params"]),
            "is_sharpe": is_sharpe,
            "oos_sharpe": oos_sharpe,
            "is_ann_return": _as_float(r["is_ann_return"]),
            "oos_ann_return": _as_float(r["oos_ann_return"]),
            "oos_ann_vol": _as_float(r["oos_ann_vol"]),
            "oos_max_drawdown": _as_float(r["oos_max_drawdown"]),
            "oos_turnover": _as_float(r["oos_turnover"]),
            "is_beats_benchmark": int(r["is_beats_benchmark"]) if r["is_beats_benchmark"] is not None else None,
            "oos_beats_benchmark": int(r["oos_beats_benchmark"]) if r["oos_beats_benchmark"] is not None else None,
            "is_oos_sharpe_gap": gap,
            "all_in_cost_bps": _as_float(r["all_in_cost_bps"]),
            "universe_n": int(r["universe_n"]) if r["universe_n"] is not None else None,
        })
    return out


# --------------------------------------------------------------------------
# Per-candidate flags
# --------------------------------------------------------------------------

def is_robust(row: dict) -> bool:
    """Did the edge SURVIVE the walk forward? (overfitting-robustness gate)

    True iff ALL of:
      * OOS Sharpe positive AND >= ROBUST_MIN_OOS_SHARPE (a real, usable edge OOS),
      * beat its OOS benchmark (added value vs. the passive 1/N alternative),
      * IS->OOS degradation contained: gap <= ROBUST_MAX_GAP, AND OOS retained
        >= ROBUST_RETENTION of the IS Sharpe (kept at least half its shine).

    The retention test is skipped/relaxed when IS Sharpe <= 0 (a strategy that was
    unremarkable in-sample yet works OOS cannot "degrade"; the gap+floor tests
    still apply).
    """
    oos = row["oos_sharpe"]
    is_s = row["is_sharpe"]
    gap = row["is_oos_sharpe_gap"]
    if oos is None or oos <= 0 or oos < ROBUST_MIN_OOS_SHARPE:
        return False
    if not row.get("oos_beats_benchmark"):
        return False
    if gap is not None and gap > ROBUST_MAX_GAP:
        return False
    if is_s is not None and is_s > 0 and oos < ROBUST_RETENTION * is_s:
        return False
    return True


def annotate(rows: list[dict], hurdle: float) -> list[dict]:
    """Attach oos_rank, robust, survives_mtc to each row; return OOS-sorted."""
    ranked = sorted(rows, key=lambda r: r["oos_sharpe"], reverse=True)
    for i, r in enumerate(ranked, 1):
        r["oos_rank"] = i
        r["robust"] = is_robust(r)
        r["survives_mtc"] = r["oos_sharpe"] >= hurdle
    return ranked


# --------------------------------------------------------------------------
# Family-level view
# --------------------------------------------------------------------------

def family_view(rows: list[dict]) -> list[dict]:
    """Per family: n, median/best OOS Sharpe, median IS->OOS gap, robust/survivor
    counts. A family whose edge is STABLE across many configs (high median OOS
    Sharpe, small median gap) is more trustworthy than one lucky config."""
    fams: dict[str, list[dict]] = {}
    for r in rows:
        fams.setdefault(r["family"], []).append(r)
    view = []
    for fam, rs in fams.items():
        oos = [r["oos_sharpe"] for r in rs]
        gaps = [r["is_oos_sharpe_gap"] for r in rs if r["is_oos_sharpe_gap"] is not None]
        view.append({
            "family": fam,
            "n": len(rs),
            "median_oos_sharpe": round(median(oos), 4),
            "best_oos_sharpe": round(max(oos), 4),
            "median_is_oos_gap": round(median(gaps), 4) if gaps else None,
            "n_robust": sum(1 for r in rs if r["robust"]),
            "n_survive_mtc": sum(1 for r in rs if r["survives_mtc"]),
        })
    # worst-overfitting families surface via median gap; sort by trustworthiness
    view.sort(key=lambda v: v["median_oos_sharpe"], reverse=True)
    return view


# --------------------------------------------------------------------------
# Analyze — assemble the full honest picture
# --------------------------------------------------------------------------

def analyze(sweep_id: str, *, top: int = 25) -> dict:
    """Run the full overfitting-aware analysis for one sweep. Pure (no writes)."""
    rows = load_valid_rows(sweep_id)
    n_total = len(rows)
    n_periods = _oos_periods_for(sweep_id)
    mtc = mtc_threshold(n_total, n_periods)
    hurdle = mtc["sharpe_hurdle"]

    ranked = annotate(rows, hurdle)
    n_robust = sum(1 for r in ranked if r["robust"])
    n_survive = sum(1 for r in ranked if r["survives_mtc"])

    # Naive cross-check: the highest-IS-Sharpe config (the WRONG pick) and its OOS rank.
    naive = None
    if ranked:
        naive = max(ranked, key=lambda r: r["is_sharpe"])

    fam_view = family_view(ranked)

    def _slim(r: dict) -> dict:
        return {
            "oos_rank": r["oos_rank"], "family": r["family"], "params": r["params"],
            "oos_sharpe": round(r["oos_sharpe"], 4), "is_sharpe": round(r["is_sharpe"], 4),
            "is_oos_sharpe_gap": round(r["is_oos_sharpe_gap"], 4) if r["is_oos_sharpe_gap"] is not None else None,
            "oos_ann_return": r["oos_ann_return"], "oos_max_drawdown": r["oos_max_drawdown"],
            "oos_beats_benchmark": r["oos_beats_benchmark"],
            "robust": r["robust"], "survives_mtc": r["survives_mtc"],
            "result_id": r["result_id"],
        }

    top_k = [_slim(r) for r in ranked[:top]]
    best = ranked[0] if ranked else None

    return {
        "sweep_id": sweep_id,
        "n_total": n_total,
        "n_robust": n_robust,
        "n_survive_mtc": n_survive,
        "n_periods_oos": n_periods,
        "mtc": mtc,
        "thresholds": {
            "robust_max_gap": ROBUST_MAX_GAP,
            "robust_retention": ROBUST_RETENTION,
            "robust_min_oos_sharpe": ROBUST_MIN_OOS_SHARPE,
            "mtc_confidence": MTC_CONFIDENCE,
        },
        "best_oos": _slim(best) if best else None,
        "naive_is_best": (_slim(naive) | {"is_best_of": "IS Sharpe"}) if naive else None,
        "family_view": fam_view,
        "top_k": top_k,
    }


# --------------------------------------------------------------------------
# Persist the compact summary (one row per analyze run)
# --------------------------------------------------------------------------

def persist(analysis: dict) -> str:
    """Write one ``sweep_analysis`` row and return the analysis_id."""
    aid = _uid("ana")
    best = analysis["best_oos"] or {}
    naive = analysis["naive_is_best"] or {}
    summary = {
        "n_total": analysis["n_total"],
        "n_robust": analysis["n_robust"],
        "n_survive_mtc": analysis["n_survive_mtc"],
        "n_periods_oos": analysis["n_periods_oos"],
        "mtc": analysis["mtc"],
        "thresholds": analysis["thresholds"],
        "best_oos": analysis["best_oos"],
        "naive_is_best": analysis["naive_is_best"],
        "family_view": analysis["family_view"],
        "top_k": analysis["top_k"],
    }
    rdb.execute(
        """INSERT INTO sweep_analysis
           (analysis_id, sweep_id, best_oos_family, best_oos_sharpe, best_oos_params,
            n_total, n_robust, n_survive_mtc, naive_is_best_params,
            naive_is_best_oos_rank, summary_json, created_at)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, NOW(6))""",
        (aid, analysis["sweep_id"], best.get("family"), best.get("oos_sharpe"),
         json.dumps(best.get("params") or {}),
         analysis["n_total"], analysis["n_robust"], analysis["n_survive_mtc"],
         json.dumps(naive.get("params") or {}), naive.get("oos_rank"),
         json.dumps(summary)),
    )
    return aid


# --------------------------------------------------------------------------
# Printed report
# --------------------------------------------------------------------------

def _fmt(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "  n/a"


def report(analysis: dict, *, top: int = 25) -> None:
    a = analysis
    mtc = a["mtc"]
    print("\n" + "=" * 116)
    print(f"SWEEP ANALYSIS — {a['sweep_id']}   (honest, OVERFITTING-AWARE ranking)")
    print("=" * 116)
    print(f"Ranked {a['n_total']} valid trials (error IS NULL, IS+OOS Sharpe present) "
          f"by OUT-OF-SAMPLE Sharpe.")
    print("NOTE: ranking by IN-SAMPLE Sharpe is the WRONG thing to do — every config was")
    print("      implicitly selected to maximize it, so the IS leader is a near-certain")
    print("      false discovery. The naive IS pick is shown below to demonstrate its collapse.")

    print("\n" + "-" * 116)
    print("MULTIPLE-TESTING / DEFLATED-SHARPE CORRECTION")
    print("-" * 116)
    print(f"  N trials                 : {mtc['n_trials']}")
    print(f"  OOS periods (n)          : {mtc['n_periods']}   -> Sharpe SE ~= sqrt(1/n) = {_fmt(mtc['sharpe_se'],4)}")
    print(f"  E[max z] of N nulls      : {_fmt(mtc['expected_max_z'],3)}   "
          f"(sqrt(2 ln N) shorthand = {_fmt(mtc['sqrt_2lnN'],3)})")
    print(f"  + {int(mtc['confidence']*100)}% Gumbel-tail cushion : z_gate = {_fmt(mtc['z_gate'],3)}")
    print(f"  => OOS Sharpe HURDLE     : {_fmt(mtc['sharpe_hurdle'],3)}  "
          f"(a config must clear this to beat the luckiest of {mtc['n_trials']} zero-edge trials)")
    print(f"  scipy used               : {mtc['used_scipy']}")
    print(f"  ASSUMPTION: trials treated as INDEPENDENT (optimistic; real sweeps share signals,")
    print(f"  so effective N is smaller and the true hurdle is a touch lower — this gate is")
    print(f"  therefore slightly CONSERVATIVE, the safe direction for a real-money decision).")
    print(f"\n  Winners surviving the correction: {a['n_survive_mtc']} of {a['n_total']} "
          f"({a['n_total'] - a['n_survive_mtc']} are likely NOISE).")
    print(f"  Robust (edge survived the walk-forward): {a['n_robust']} of {a['n_total']}.")

    th = a["thresholds"]
    print(f"\n  robust gate: oos_sharpe>0 & >= {th['robust_min_oos_sharpe']} & beats OOS benchmark "
          f"& gap <= {th['robust_max_gap']} & oos >= {th['robust_retention']}*is_sharpe")

    print("\n" + "-" * 116)
    print(f"TOP {min(top, len(a['top_k']))} BY OOS SHARPE (the honest leaderboard)")
    print("-" * 116)
    print(f"{'#':>3}  {'family':<14} {'oos_shp':>8} {'is_shp':>8} {'gap':>7} "
          f"{'oos_ret':>8} {'oos_mdd':>8} {'bb':>3} {'robust':>7} {'mtc':>4}  params")
    for r in a["top_k"][:top]:
        print(f"{r['oos_rank']:>3}  {r['family']:<14} {_fmt(r['oos_sharpe']):>8} "
              f"{_fmt(r['is_sharpe']):>8} {_fmt(r['is_oos_sharpe_gap']):>7} "
              f"{_fmt(r['oos_ann_return']):>8} {_fmt(r['oos_max_drawdown']):>8} "
              f"{('Y' if r['oos_beats_benchmark'] else '.'):>3} "
              f"{('YES' if r['robust'] else '-'):>7} {('OK' if r['survives_mtc'] else 'x'):>4}  "
              f"{json.dumps(r['params'])[:60]}")

    print("\n" + "-" * 116)
    print("NAIVE PICK CROSS-CHECK (the money-shot: IS->OOS collapse)")
    print("-" * 116)
    naive = a["naive_is_best"]
    best = a["best_oos"]
    if naive:
        print(f"  Naive winner (BEST IN-SAMPLE Sharpe): {naive['family']:<14} "
              f"is_sharpe={_fmt(naive['is_sharpe'])}")
        print(f"    -> OUT OF SAMPLE it scores oos_sharpe={_fmt(naive['oos_sharpe'])} "
              f"(gap={_fmt(naive['is_oos_sharpe_gap'])}), ranking #{naive['oos_rank']} of {a['n_total']} OOS.")
        print(f"    -> robust={naive['robust']}, survives_mtc={naive['survives_mtc']}  "
              f"params={json.dumps(naive['params'])[:80]}")
        if naive['oos_rank'] and naive['oos_rank'] > 1:
            print("    => the in-sample leader is NOT the out-of-sample leader — do not ship it.")
    if best:
        print(f"\n  Honest winner (BEST OUT-OF-SAMPLE Sharpe): {best['family']:<14} "
              f"oos_sharpe={_fmt(best['oos_sharpe'])} (is_sharpe={_fmt(best['is_sharpe'])}, "
              f"gap={_fmt(best['is_oos_sharpe_gap'])})")
        print(f"    robust={best['robust']}, survives_mtc={best['survives_mtc']}  "
              f"params={json.dumps(best['params'])[:80]}")

    print("\n" + "-" * 116)
    print("FAMILY-LEVEL VIEW (stable edge across many configs > one lucky config)")
    print("-" * 116)
    print(f"{'family':<16} {'n':>5} {'med_oos':>8} {'best_oos':>8} {'med_gap':>8} "
          f"{'robust':>7} {'survive':>8}")
    for f in a["family_view"]:
        print(f"{f['family']:<16} {f['n']:>5} {_fmt(f['median_oos_sharpe']):>8} "
              f"{_fmt(f['best_oos_sharpe']):>8} {_fmt(f['median_is_oos_gap']):>8} "
              f"{f['n_robust']:>7} {f['n_survive_mtc']:>8}")
    worst = max(a["family_view"], key=lambda v: (v["median_is_oos_gap"] or -1)) if a["family_view"] else None
    if worst and worst["median_is_oos_gap"] is not None:
        print(f"\n  Worst-overfitting family (largest median IS->OOS gap): "
              f"{worst['family']} (median gap {_fmt(worst['median_is_oos_gap'])}).")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        "sweep_analyze",
        description="Honest, overfitting-aware ranking of a strategy parameter sweep.")
    ap.add_argument("--sweep-id", default=None,
                    help="sweep to analyze (default: latest sweep_runs row)")
    ap.add_argument("--latest", action="store_true",
                    help="analyze the latest sweep (default when --sweep-id omitted)")
    ap.add_argument("--top", type=int, default=25, help="top-K to display + emit as JSON")
    ap.add_argument("--no-persist", action="store_true",
                    help="do not write a sweep_analysis row")
    ap.add_argument("--no-report", action="store_true",
                    help="suppress the human report (emit JSON only)")
    args = ap.parse_args(argv)

    sweep_id = args.sweep_id or latest_sweep_id()
    if not sweep_id:
        print("no sweep found (sweep_runs + sweep_results are empty)", file=sys.stderr)
        return 2

    analysis = analyze(sweep_id, top=args.top)
    if not args.no_report:
        report(analysis, top=args.top)

    analysis_id = None
    if not args.no_persist:
        analysis_id = persist(analysis)
        if not args.no_report:
            print(f"\npersisted sweep_analysis row: {analysis_id}")

    # Emit top-K as JSON to stdout for the orchestrator to consume.
    payload = {
        "analysis_id": analysis_id,
        "sweep_id": sweep_id,
        "n_total": analysis["n_total"],
        "n_robust": analysis["n_robust"],
        "n_survive_mtc": analysis["n_survive_mtc"],
        "sharpe_hurdle": analysis["mtc"]["sharpe_hurdle"],
        "best_oos": analysis["best_oos"],
        "naive_is_best": analysis["naive_is_best"],
        "top_k": analysis["top_k"],
    }
    print("\n===TOP_K_JSON===")
    print(json.dumps(payload, default=str))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    raise SystemExit(main())
