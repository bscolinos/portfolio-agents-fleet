"""Tests for the overfitting-aware sweep analyzer.

These are DB-backed (they seed a TINY synthetic sweep into ``sweep_results`` under
a throwaway sweep_id, run the analyzer, and assert on the honest ranking), and are
SKIPPED automatically when SingleStore is unreachable so the suite still runs
anywhere. The fixture DELETEs every test row (from sweep_results, sweep_runs, and
sweep_analysis) in teardown.

Coverage:
  * ranking is by OOS Sharpe and excludes error rows;
  * the deflated/MTC hurdle flags obvious-noise rows and passes genuinely-strong ones;
  * the naive IS-best row is identified and its OOS rank computed (the collapse);
  * family aggregation sums to the ranked row count;
  * pure statistics (E[max z], SE, hurdle, closed-form normal ppf/cdf).
"""

from __future__ import annotations

import json
import uuid

import pytest

from research_fleet.research_agent import sweep_analyze as sa


# --------------------------------------------------------------------------
# Pure-statistics tests (no DB)
# --------------------------------------------------------------------------

def test_normal_ppf_cdf_closed_form():
    """The closed-form (scipy-free) normal ppf/cdf must hit textbook values."""
    assert sa._norm_ppf(0.975) == pytest.approx(1.959964, abs=1e-4)
    assert sa._norm_ppf(0.5) == pytest.approx(0.0, abs=1e-6)
    assert sa._norm_cdf(1.959964) == pytest.approx(0.975, abs=1e-4)
    assert sa._norm_cdf(0.0) == pytest.approx(0.5, abs=1e-9)


def test_expected_max_z_grows_with_n():
    """E[max of N nulls] is 0 for N=1 and increases with N (selection bias)."""
    assert sa.expected_max_z(1) == 0.0
    assert sa.expected_max_z(10) < sa.expected_max_z(100) < sa.expected_max_z(10000)
    # tracks the sqrt(2 ln N) order of magnitude (below it, as the refined form should be)
    import math
    assert 0.5 * math.sqrt(2 * math.log(1000)) < sa.expected_max_z(1000) < math.sqrt(2 * math.log(1000))


def test_mtc_hurdle_scales_with_trials_and_periods():
    """More trials -> higher hurdle; more OOS periods -> lower hurdle (tighter SE)."""
    lo = sa.mtc_threshold(30, 252)["sharpe_hurdle"]
    hi = sa.mtc_threshold(5000, 252)["sharpe_hurdle"]
    assert hi > lo > 0
    tight = sa.mtc_threshold(1000, 2520)["sharpe_hurdle"]
    loose = sa.mtc_threshold(1000, 252)["sharpe_hurdle"]
    assert tight < loose  # 10x the OOS periods shrinks the SE, lowering the bar


# --------------------------------------------------------------------------
# DB-backed synthetic sweep
# --------------------------------------------------------------------------

def _db_reachable() -> bool:
    try:
        from research_fleet.research_agent import research_db as rdb
        rdb.query("SELECT 1 AS ok")
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _db_reachable(), reason="SingleStore unreachable")

_INSERT = """INSERT INTO sweep_results
    (result_id, sweep_id, family, params, is_sharpe, is_ann_return, is_ann_vol,
     is_max_drawdown, is_turnover, is_beats_benchmark, oos_sharpe, oos_ann_return,
     oos_ann_vol, oos_max_drawdown, oos_turnover, oos_beats_benchmark,
     is_oos_sharpe_gap, all_in_cost_bps, universe_n, error, created_at)
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, NOW(6))"""


def _row(sweep_id, family, is_s, oos_s, *, beats_oos=1, error=None, i=0):
    gap = None if (is_s is None or oos_s is None) else (is_s - oos_s)
    return (
        f"res-test-{i:03d}-{uuid.uuid4().hex[:6]}", sweep_id, family,
        json.dumps({"family": family, "lookback": 20 + i, "idx": i}),
        is_s, 0.10, 0.15, -0.20, 1.5, 1,
        oos_s, 0.08, 0.14, -0.18, 1.4, beats_oos,
        gap, 10.0, 40, error,
    )


@pytest.fixture()
def seeded_sweep():
    """Seed ~30 synthetic rows: robust winners, overfit collapsers, noise, errors."""
    from research_fleet.research_agent import research_db as rdb
    sweep_id = f"swp-test-{uuid.uuid4().hex[:10]}"
    rows = []
    i = 0

    # 4 genuinely ROBUST rows: strong OOS, beats benchmark, small IS->OOS gap.
    # (oos well above the ~0.24 hurdle; oos >= 0.5*is; gap <= 1.0)
    for oos in (1.6, 1.4, 1.2, 1.0):
        rows.append(_row(sweep_id, "momentum", is_s=oos + 0.4, oos_s=oos, beats_oos=1, i=i)); i += 1

    # 3 more robust-ish in a second family (stable edge across configs)
    for oos in (1.1, 0.9, 0.8):
        rows.append(_row(sweep_id, "risk_parity", is_s=oos + 0.3, oos_s=oos, beats_oos=1, i=i)); i += 1

    # 10 OVERFIT collapsers: huge IS Sharpe, near-zero OOS, does NOT beat benchmark.
    # These are obvious noise: below the MTC hurdle and not robust.
    for k in range(10):
        is_s = 2.8 + 0.12 * k          # 2.8 .. 3.88 (the naive IS leader lives here)
        oos_s = 0.15 - 0.01 * k        # 0.15 .. 0.06  (all below hurdle ~0.24)
        rows.append(_row(sweep_id, "mean_reversion", is_s=is_s, oos_s=oos_s, beats_oos=0, i=i)); i += 1

    # 6 middling noise rows in a third family: modest OOS below the hurdle.
    for k in range(6):
        rows.append(_row(sweep_id, "vol_target", is_s=0.9, oos_s=0.10 + 0.01 * k, beats_oos=0, i=i)); i += 1

    # 5 ERROR rows (must be excluded from ranking entirely) — some with a high
    # oos_sharpe that WOULD top the board if not excluded.
    for k in range(5):
        r = _row(sweep_id, "factor", is_s=3.0, oos_s=9.9, beats_oos=1, error="backtest blew up", i=i); i += 1
        rows.append(r)

    rdb.execute(
        """INSERT INTO sweep_runs (sweep_id, target_n, actual_n, seed, is_start, is_end,
             oos_start, oos_end, families, status, started_at, finished_at, notes)
           VALUES (%s,%s,%s,%s,'2015-01-01','2020-12-31','2021-01-01','2022-12-31',
             %s,'done',NOW(6),NOW(6),'synthetic test sweep')""",
        (sweep_id, len(rows), len(rows), 42,
         "momentum,risk_parity,mean_reversion,vol_target,factor"))
    rdb.executemany(_INSERT, rows)

    yield sweep_id, rows

    # teardown — remove ALL test rows
    rdb.execute("DELETE FROM sweep_results WHERE sweep_id=%s", (sweep_id,))
    rdb.execute("DELETE FROM sweep_runs WHERE sweep_id=%s", (sweep_id,))
    rdb.execute("DELETE FROM sweep_analysis WHERE sweep_id=%s", (sweep_id,))


def test_ranking_is_oos_and_excludes_errors(seeded_sweep):
    sweep_id, rows = seeded_sweep
    a = sa.analyze(sweep_id, top=25)
    # 28 total rows, 5 are error -> 23 ranked
    assert a["n_total"] == 23
    # the excluded error rows had oos_sharpe=9.9; the true OOS leader is 1.6
    assert a["best_oos"]["oos_sharpe"] == pytest.approx(1.6, abs=1e-6)
    assert a["best_oos"]["family"] == "momentum"
    # strictly descending OOS ordering
    oos_seq = [r["oos_sharpe"] for r in a["top_k"]]
    assert oos_seq == sorted(oos_seq, reverse=True)
    # no error row leaked in (none has oos ~ 9.9)
    assert all(r["oos_sharpe"] < 9.0 for r in a["top_k"])


def test_mtc_flags_noise_and_passes_strong(seeded_sweep):
    sweep_id, rows = seeded_sweep
    a = sa.analyze(sweep_id, top=25)
    hurdle = a["mtc"]["sharpe_hurdle"]
    assert 0.1 < hurdle < 1.0  # sane hurdle for ~23 trials over a ~2yr OOS window
    # the 7 strong rows (oos >= 0.8) survive; the ~18 noise rows (oos <= ~0.2) do not
    survivors = [r for r in a["top_k"] if r["survives_mtc"]]
    assert a["n_survive_mtc"] == len(survivors)
    assert a["n_survive_mtc"] == 7, f"expected 7 survivors, got {a['n_survive_mtc']}"
    # every survivor is above the hurdle; every flagged-noise row is below it
    for r in a["top_k"]:
        assert r["survives_mtc"] == (r["oos_sharpe"] >= hurdle)
    # robust winners are a subset of survivors and are the strong momentum/risk_parity rows
    assert a["n_robust"] == 7
    assert all(r["robust"] is False for r in a["top_k"] if r["oos_sharpe"] < 0.5)


def test_naive_is_best_identified_and_collapses(seeded_sweep):
    sweep_id, rows = seeded_sweep
    a = sa.analyze(sweep_id, top=25)
    naive = a["naive_is_best"]
    # the highest IS Sharpe is 3.88 (last mean_reversion overfit row)
    assert naive["is_sharpe"] == pytest.approx(3.88, abs=1e-6)
    assert naive["family"] == "mean_reversion"
    # it collapses OOS: near the very bottom of the 25-row OOS ranking, not robust, not surviving
    assert naive["oos_rank"] >= 20
    assert naive["robust"] is False
    assert naive["survives_mtc"] is False


def test_family_aggregation_sums_to_rowcount(seeded_sweep):
    sweep_id, rows = seeded_sweep
    a = sa.analyze(sweep_id, top=25)
    total = sum(f["n"] for f in a["family_view"])
    assert total == a["n_total"] == 23
    # per-family robust/survivor counts also sum to the global counts
    assert sum(f["n_robust"] for f in a["family_view"]) == a["n_robust"]
    assert sum(f["n_survive_mtc"] for f in a["family_view"]) == a["n_survive_mtc"]
    # error family ('factor') is entirely excluded
    assert "factor" not in {f["family"] for f in a["family_view"]}


def test_persist_writes_one_row_and_cleanup(seeded_sweep):
    sweep_id, rows = seeded_sweep
    from research_fleet.research_agent import research_db as rdb
    a = sa.analyze(sweep_id, top=25)
    aid = sa.persist(a)
    got = rdb.query("SELECT * FROM sweep_analysis WHERE analysis_id=%s", (aid,))
    assert len(got) == 1
    row = got[0]
    assert row["sweep_id"] == sweep_id
    assert row["n_total"] == 23
    assert row["best_oos_family"] == "momentum"
    assert row["naive_is_best_oos_rank"] >= 20
    # summary_json round-trips
    summary = row["summary_json"] if isinstance(row["summary_json"], dict) else json.loads(row["summary_json"])
    assert summary["n_survive_mtc"] == a["n_survive_mtc"]
