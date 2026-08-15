"""Unit tests for the hardened backtest engine.

These exercise the two correctness fixes WITHOUT the database by calling the
internal helpers (``_apply_weights``, ``_metrics``, ``resolve_cost_bps``,
``eligible_as_of``) against tiny in-memory frames:

  * BUG 1 (turnover cost undercharged): a daily-rebalanced strategy charged 10bps
    yields a materially lower Sharpe than the same strategy charged 2bps, and the
    agents' ``turnover_cost_bps`` key is actually honored (not the old 2bps
    default).
  * BUG 2 (survivorship / look-ahead + ffill-across-gap): a delisted name with
    trailing NaNs is not carried at a stale price / does not fabricate a flat
    return, and is excluded by as-of eligibility.

A DB-backed smoke test is included but SKIPPED automatically when SingleStore is
unreachable, so the suite runs anywhere.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research_fleet.research_agent import backtest as bt


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _daily_turnover_frame(n_days: int = 260, n_names: int = 6, seed: int = 7):
    """Build a returns frame + daily-rebalance weights that FULLY churn each day
    (weights alternate between two disjoint halves), so turnover ~= 2.0/rebalance
    and the cost knob has real bite."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    cols = [f"T{i}" for i in range(n_names)]
    rets = pd.DataFrame(rng.normal(0.0004, 0.01, size=(n_days, n_names)), index=idx, columns=cols)
    half_a = pd.Series(0.0, index=cols); half_a[cols[:n_names // 2]] = 1.0 / (n_names // 2)
    half_b = pd.Series(0.0, index=cols); half_b[cols[n_names // 2:]] = 1.0 / (n_names - n_names // 2)
    weights = {d: (half_a if k % 2 == 0 else half_b) for k, d in enumerate(idx)}
    return rets, weights


# --------------------------------------------------------------------------
# BUG 1 — turnover cost is honored
# --------------------------------------------------------------------------

def test_higher_cost_lowers_sharpe_materially():
    """Same daily-churn strategy: 10bps must yield a MATERIALLY lower Sharpe than 2bps."""
    rets, weights = _daily_turnover_frame()
    p2, t2 = bt._apply_weights(rets, weights, cost_bps=2.0)
    p10, t10 = bt._apply_weights(rets, weights, cost_bps=10.0)
    s2 = bt._metrics(p2, t2)["sharpe"]
    s10 = bt._metrics(p10, t10)["sharpe"]
    # turnover is real (full churn ~ 2.0/day) so cost differs, and 10bps hurts more
    assert t10.mean() > 0.5, "test frame should have real daily turnover"
    assert s10 < s2, "charging 10bps must reduce Sharpe vs 2bps"
    # 'materially': the mean daily drag difference is (10-2)bps * turnover — non-trivial
    drag = (10.0 - 2.0) / 1e4 * t2.mean()
    assert (p2.mean() - p10.mean()) == pytest.approx(drag, rel=1e-6)
    assert (s2 - s10) > 0.05, f"expected a material Sharpe gap, got {s2 - s10:.4f}"


def test_turnover_cost_bps_is_honored_not_default():
    """resolve_cost_bps must PREFER turnover_cost_bps over the tc_bps/default path."""
    # declared turnover_cost_bps=10 -> resolved commission leg is 10, not 2 or 5
    turn, slip, allin = bt.resolve_cost_bps({"turnover_cost_bps": 10})
    assert turn == 10.0
    assert allin == 10.0 + bt.DEFAULT_SLIPPAGE_BPS
    # legacy tc_bps still works as a fallback
    turn2, _, _ = bt.resolve_cost_bps({"tc_bps": 3})
    assert turn2 == 3.0
    # nothing declared -> defensible institutional default (NOT the old 2.0)
    turn3, _, _ = bt.resolve_cost_bps({})
    assert turn3 == bt.DEFAULT_TURNOVER_COST_BPS
    assert bt.DEFAULT_TURNOVER_COST_BPS != 2.0
    # and the honored cost actually changes realized returns end-to-end
    rets, weights = _daily_turnover_frame()
    p_default, _ = bt._apply_weights(rets, weights, cost_bps=bt.resolve_cost_bps({})[2])
    p_declared, _ = bt._apply_weights(rets, weights, cost_bps=bt.resolve_cost_bps({"turnover_cost_bps": 10})[2])
    assert p_declared.mean() < p_default.mean(), "declared 10bps must cost more than the default"


# --------------------------------------------------------------------------
# BUG 2 — no forward-fill-across-gap leakage / survivorship
# --------------------------------------------------------------------------

def test_delisted_name_not_carried_flat():
    """A name that delists (trailing NaN returns) must contribute 0 — never a
    fabricated flat return, and never a jump when/if it reappears."""
    idx = pd.bdate_range("2021-01-01", periods=10)
    # GOOD trades every day; DEAD delists after day 3 (NaN returns thereafter).
    good = np.full(10, 0.01)
    dead = np.array([0.01, 0.01, 0.01, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    rets = pd.DataFrame({"GOOD": good, "DEAD": dead}, index=idx)
    # Hold 50/50 the whole time (single rebalance up front, no churn after).
    w = {idx[0]: pd.Series({"GOOD": 0.5, "DEAD": 0.5})}
    port, turn = bt._apply_weights(rets, w, cost_bps=0.0)
    # On days where DEAD is NaN, its contribution is 0, so the whole-portfolio
    # return is exactly 0.5 * GOOD (idle capital), NOT 0.5*GOOD + 0.5*stale_flat.
    for k in range(3, 10):
        assert port.iloc[k] == pytest.approx(0.5 * 0.01), \
            f"day {k}: dead name must contribute 0, not a fabricated return"
    # sanity: while both alive, both contribute
    assert port.iloc[1] == pytest.approx(0.5 * 0.01 + 0.5 * 0.01)


def test_eligible_as_of_excludes_missing_and_short_history():
    """eligible_as_of must use ONLY trailing info: exclude names with no price at
    t, and names lacking the required trailing history."""
    idx = pd.bdate_range("2021-01-01", periods=12)
    px_full = np.linspace(100, 111, 12)
    px_late = np.array([np.nan] * 8 + list(np.linspace(50, 53, 4)))  # starts late
    px_dead = np.array(list(np.linspace(20, 26, 7)) + [np.nan] * 5)   # delists at day 7
    panel = pd.DataFrame({"FULL": px_full, "LATE": px_late, "DEAD": px_dead}, index=idx)
    lookback = 5
    # As of day 9 (index 9): FULL has full history; LATE has only ~2 obs; DEAD is NaN now.
    elig = bt.eligible_as_of(panel, 9, lookback)
    assert "FULL" in elig
    assert "DEAD" not in elig, "delisted name (NaN at t) must be ineligible"
    assert "LATE" not in elig, "name without >= lookback trailing obs must be ineligible"


def test_load_panel_uses_bounded_ffill_only():
    """The corrected loader forward-fills only within MAX_FFILL_GAP — a long gap
    stays NaN (not carried flat). We assert the constant exists and is small."""
    assert hasattr(bt, "MAX_FFILL_GAP")
    assert 0 < bt.MAX_FFILL_GAP <= 5, "bounded ffill gap must be short + documented"


def test_data_caveats_is_honest_about_pit_membership():
    """The residual limitation must be surfaced and must NOT claim true PIT membership."""
    assert isinstance(bt.DATA_CAVEATS, str) and len(bt.DATA_CAVEATS) > 40
    low = bt.DATA_CAVEATS.lower()
    assert "point-in-time" in low or "point in time" in low
    assert "not reconstructable" in low or "approximat" in low


# --------------------------------------------------------------------------
# DB-backed smoke (skipped when SingleStore is unreachable)
# --------------------------------------------------------------------------

def _db_reachable() -> bool:
    try:
        from research_fleet.research_agent import research_db as rdb
        rdb.query("SELECT 1 AS ok")
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _db_reachable(), reason="SingleStore unreachable")
def test_run_backtest_honors_declared_cost_end_to_end():
    """End-to-end over real prices: a high-turnover daily strategy scored at a
    higher declared cost must not score HIGHER than at a lower cost."""
    from research_fleet.research_agent import backtest as _bt
    lo = _bt.run_backtest("mean_reversion",
                          {"rebalance_days": 1, "reversal_days": 5, "turnover_cost_bps": 1},
                          start="2018-01-01", end="2024-12-31", universe_n=40)
    hi = _bt.run_backtest("mean_reversion",
                          {"rebalance_days": 1, "reversal_days": 5, "turnover_cost_bps": 30},
                          start="2018-01-01", end="2024-12-31", universe_n=40)
    assert lo.get("sharpe") is not None and hi.get("sharpe") is not None
    assert hi["sharpe"] <= lo["sharpe"] + 1e-9, "higher cost must not improve Sharpe"
    assert "data_caveats" in lo and "all_in_cost_bps" in lo
