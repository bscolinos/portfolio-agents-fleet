"""Tests for the parameter-sweep engine + search space.

Three layers:

  * ``param_space`` (pure, no DB): every family's grid is non-empty; ``sample`` is
    deterministic given a seed and returns n de-duped dicts; ``plan`` returns
    ~target_n UNIQUE (family, params) spread across families.
  * PARITY (the important one, DB-guarded): for a handful of (family, params), the
    engine's cached in-memory evaluation (``sweep._eval_on_panel`` on a panel from
    ``load_price_panel``) yields the SAME Sharpe as ``backtest.run_backtest`` on the
    same window — within 1e-9. This is the whole justification for the cache.
  * END-TO-END (DB-guarded): ``run_sweep`` on a SMALL universe + short windows
    writes N sweep_results rows (IS + OOS populated), flips its sweep_runs row to
    'done', and then CLEANS UP its own rows.

DB-backed tests auto-skip when SingleStore is unreachable so the pure tests run
anywhere.
"""

from __future__ import annotations

import json

import pytest

from research_fleet.research_agent import param_space as ps
from research_fleet.research_agent import sweep as sw
from research_fleet.research_agent import backtest as bt


# --------------------------------------------------------------------------
# param_space — pure, no DB
# --------------------------------------------------------------------------

def test_all_families_have_nonempty_grid():
    assert set(ps.FAMILIES) and len(ps.FAMILIES) == 8
    for fam in ps.FAMILIES:
        g = list(ps.grid(fam))
        assert len(g) > 0, f"{fam} grid empty"
        assert len(g) == ps.grid_size(fam), f"{fam} grid_size mismatch"
        # every produced dict carries the swept knobs and is a plain dict
        assert all(isinstance(p, dict) and "rebalance_days" in p for p in g)


def test_total_grid_size_sums():
    tg = ps.total_grid_size()
    assert set(tg) == set(ps.FAMILIES) | {"overall"}
    assert tg["overall"] == sum(tg[f] for f in ps.FAMILIES)
    assert all(tg[f] > 0 for f in ps.FAMILIES)


def test_sample_is_deterministic_and_deduped():
    for fam in ps.FAMILIES:
        a = ps.sample(fam, 20, seed=7)
        b = ps.sample(fam, 20, seed=7)
        assert a == b, f"{fam}: sample not deterministic for a fixed seed"
        # de-duped
        keys = {json.dumps(p, sort_keys=True) for p in a}
        assert len(keys) == len(a), f"{fam}: sample returned duplicates"
        # bounded by grid size; when grid >= n we get exactly n
        assert len(a) == min(20, ps.grid_size(fam))
    # a different seed generally yields a different draw (for a big grid)
    assert ps.sample("momentum", 20, seed=7) != ps.sample("momentum", 20, seed=8)


def test_plan_is_unique_and_spread():
    target = 200
    pl = ps.plan(target, seed=7)
    # ~target (exact here since overall grid >> target)
    assert abs(len(pl) - target) <= 8
    # unique (family, params)
    keys = {(f, json.dumps(p, sort_keys=True)) for f, p in pl}
    assert len(keys) == len(pl), "plan contains duplicate (family, params)"
    # spread across families — every family represented for a target this size
    fams = {f for f, _ in pl}
    assert fams == set(ps.FAMILIES), f"plan not spread across all families: {fams}"


def test_plan_respects_family_filter():
    pl = ps.plan(60, seed=3, families=["momentum", "regime"])
    fams = {f for f, _ in pl}
    assert fams == {"momentum", "regime"}


# --------------------------------------------------------------------------
# DB guard
# --------------------------------------------------------------------------

def _db_reachable() -> bool:
    try:
        from research_fleet.research_agent import research_db as rdb
        rdb.query("SELECT 1 AS ok")
        return True
    except Exception:
        return False


DB = _db_reachable()


# --------------------------------------------------------------------------
# PARITY — cached in-memory path == run_backtest, within 1e-9
# --------------------------------------------------------------------------

@pytest.mark.skipif(not DB, reason="SingleStore unreachable")
def test_cached_eval_matches_run_backtest():
    """For a spread of families/params, sweep._eval_on_panel on a loaded panel must
    equal backtest.run_backtest on the same window to within 1e-9 (Sharpe + a few
    other metrics). This is the parity guarantee that justifies the cache."""
    start, end, un = "2018-01-01", "2021-12-31", 40
    cases = [
        ("equal_weight", {"rebalance_days": 21}),
        ("momentum", {"lookback_days": 126, "skip_days": 21, "top_n": 10, "rebalance_days": 21}),
        ("mean_reversion", {"lookback_days": 63, "reversal_days": 5, "bottom_n": 10, "rebalance_days": 5}),
        ("low_vol", {"lookback_days": 126, "keep_n": 20, "w_max": 0.10, "rebalance_days": 21}),
        ("risk_parity", {"lookback_days": 126, "w_max": 0.20, "rebalance_days": 63}),
        ("regime", {"lookback_days": 252, "ma_days": 100, "rebalance_days": 21,
                    "turnover_cost_bps": 10}),
    ]
    panel = bt.load_price_panel(start, end, un)
    max_diff = 0.0
    for fam, params in cases:
        ref = bt.run_backtest(fam, params, start=start, end=end, universe_n=un)
        got = sw._eval_on_panel(fam, params, panel)
        # both must be non-error and produce a Sharpe
        assert ref.get("sharpe") is not None, f"{fam}: run_backtest gave no sharpe"
        assert got.get("sharpe") is not None, f"{fam}: cached eval gave no sharpe"
        for key in ("sharpe", "ann_return", "ann_vol", "max_drawdown", "turnover"):
            a, b = ref.get(key), got.get(key)
            if a is None or b is None:
                assert a == b, f"{fam}: {key} None-mismatch {a} vs {b}"
                continue
            d = abs(float(a) - float(b))
            max_diff = max(max_diff, d)
            assert d < 1e-9, f"{fam}: {key} parity break {a} vs {b} (|Δ|={d})"
    # surface the max observed diff for the record
    print(f"\nPARITY max |Δ| across all metrics/cases = {max_diff:.3e}")
    assert max_diff < 1e-9


# --------------------------------------------------------------------------
# END-TO-END — small run writes rows, flips to done, then cleans up
# --------------------------------------------------------------------------

@pytest.mark.skipif(not DB, reason="SingleStore unreachable")
def test_run_sweep_writes_and_finalizes_then_cleanup():
    from research_fleet.research_agent import research_db as rdb
    sweep_id = "swp-test-" + "e2e0"  # stable test id so cleanup is unambiguous
    # ensure a clean slate for this id
    rdb.execute("DELETE FROM sweep_results WHERE sweep_id=%s", (sweep_id,))
    rdb.execute("DELETE FROM sweep_runs WHERE sweep_id=%s", (sweep_id,))
    try:
        s = sw.run_sweep(
            target_n=12, seed=5,
            is_start="2018-01-01", is_end="2019-12-31",
            oos_start="2020-01-01", oos_end="2020-12-31",
            universe_default=30, sweep_id=sweep_id, limit=12)
        assert s["sweep_id"] == sweep_id
        assert s["written"] == 12, f"expected 12 rows, wrote {s['written']}"
        # panel loads should be a small number (one per distinct window x universe),
        # NOT ~24 — proving the cache is used across the 12 configs.
        assert s["panel_loads"] <= 8, f"cache not working, {s['panel_loads']} loads"

        rows = rdb.query(
            "SELECT is_sharpe, oos_sharpe, is_oos_sharpe_gap, family, error "
            "FROM sweep_results WHERE sweep_id=%s", (sweep_id,))
        assert len(rows) == 12
        # at least the non-errored rows have BOTH IS and OOS populated + a gap
        good = [r for r in rows if not r["error"]]
        assert good, "expected at least some non-errored configs"
        for r in good:
            assert r["is_sharpe"] is not None and r["oos_sharpe"] is not None
            gap = float(r["is_sharpe"]) - float(r["oos_sharpe"])
            assert abs(gap - float(r["is_oos_sharpe_gap"])) < 1e-9

        run = rdb.query("SELECT status, actual_n FROM sweep_runs WHERE sweep_id=%s", (sweep_id,))
        assert run and run[0]["status"] == "done"
        assert int(run[0]["actual_n"]) == 12
    finally:
        rdb.execute("DELETE FROM sweep_results WHERE sweep_id=%s", (sweep_id,))
        rdb.execute("DELETE FROM sweep_runs WHERE sweep_id=%s", (sweep_id,))


@pytest.mark.skipif(not DB, reason="SingleStore unreachable")
def test_bad_config_records_error_not_metrics():
    """A config that cannot produce weights (unknown family -> _weights_for returns
    None on every date) must yield an error dict with a null Sharpe, so run_sweep
    records it as an error row rather than fabricating metrics."""
    cache = sw.PanelCache()
    panel = cache.get("2020-01-01", "2020-12-31", 30)
    m = sw._eval_on_panel("no_such_family", {"rebalance_days": 21}, panel)
    assert m.get("error") is not None and m.get("sharpe") is None


def test_eval_on_empty_panel_returns_error(monkeypatch):
    """The insufficient-data guard must return an error dict, never raise — proving
    a bad panel is recorded, not fatal (no DB needed)."""
    import pandas as pd
    m = sw._eval_on_panel("momentum", {"rebalance_days": 21}, pd.DataFrame())
    assert m.get("error") == "insufficient price data" and m.get("sharpe") is None


def test_run_sweep_survives_eval_exception(monkeypatch):
    """If _eval_on_panel RAISES for a config, run_sweep must catch it, record an
    error row, and keep going — one bad config never kills the sweep. Fully mocked
    (no DB): we stub the panel cache, the eval, and the rdb writes."""
    import research_fleet.research_agent.sweep as S

    # deterministic 4-config plan, all one family
    monkeypatch.setattr(S.ps, "plan", lambda *a, **k: [
        ("momentum", {"rebalance_days": 21, "universe_n": 30}),
        ("momentum", {"rebalance_days": 5, "universe_n": 30}),
        ("momentum", {"rebalance_days": 63, "universe_n": 30}),
        ("momentum", {"rebalance_days": 21, "universe_n": 60}),
    ])
    monkeypatch.setattr(S, "ensure_tables", lambda: None)
    monkeypatch.setattr(S.PanelCache, "get", lambda self, s, e, u: "PANEL")

    calls = {"n": 0}

    def _flaky_eval(family, params, panel):
        calls["n"] += 1
        if calls["n"] == 2:  # second config blows up
            raise RuntimeError("boom")
        return {"sharpe": 1.0, "ann_return": 0.1, "ann_vol": 0.1,
                "max_drawdown": -0.05, "turnover": 0.2, "beats_benchmark": True,
                "all_in_cost_bps": 7.0}

    monkeypatch.setattr(S, "_eval_on_panel", _flaky_eval)

    inserted: list = []
    monkeypatch.setattr(S.rdb, "execute", lambda *a, **k: 0)
    monkeypatch.setattr(S.rdb, "executemany", lambda sql, rows: inserted.extend(rows) or len(rows))

    s = S.run_sweep(target_n=4, seed=1, universe_default=30)
    assert s["written"] == 4, "all 4 configs recorded despite one raising"
    assert s["errored"] == 1, "exactly the one raising config is counted as error"
    assert len(inserted) == 4
