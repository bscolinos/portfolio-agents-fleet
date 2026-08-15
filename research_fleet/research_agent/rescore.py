"""Re-score every research_experiment through the CORRECTED backtest engine.

This is the payoff for the backtester hardening. The original engine had two
correctness bugs that mis-selected the winning strategy for a REAL-MONEY
decision:

  1. Turnover cost was undercharged — it read only ``tc_bps`` (defaulting to a
     rosy 2bps) and ignored the agents' declared ``turnover_cost_bps``, so a
     daily-rebalanced strategy that intended 10bps was charged 2bps and its
     Sharpe was inflated.
  2. Universe construction was survivorship-biased + look-ahead — it required a
     name to have data across the FULL future window and forward-filled across
     gaps, systematically overstating returns.

``backtest.py`` now fixes both. This script re-runs each experiment's own params
+ window through the corrected engine and writes the before/after to a NEW table
``experiment_rescores`` (see ``rescore_schema.sql``). It NEVER mutates the
original ``research_experiments`` rows. It then prints a ranked before/after
table and reports how the ranking changed — in particular what the OLD #1 was,
what the NEW #1 is, and where the old "Sharpe 3.66 regime daily" winner lands
once correctly costed.

Run from the demo root:
    python -m research_fleet.research_agent.rescore        # or: python rescore.py
    python -m research_fleet.research_agent.rescore --apply-schema   # create table first
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path

from . import research_db as rdb
from . import backtest as bt


def _uid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _universe_n(universe: str | None, params: dict) -> int:
    """Recover the universe size an experiment ran with (e.g. 'sp500-top60')."""
    if params.get("universe_n"):
        try:
            return int(params["universe_n"]) or 60
        except (TypeError, ValueError):
            pass
    if universe:
        m = re.search(r"(\d+)", universe)
        if m:
            return int(m.group(1))
    return 60


def _as_str_date(d) -> str:
    return d.isoformat() if hasattr(d, "isoformat") else str(d)


def apply_schema() -> None:
    """Create experiment_rescores from rescore_schema.sql (idempotent)."""
    sql = (Path(__file__).resolve().parent / "rescore_schema.sql").read_text()
    for chunk in sql.split(";"):
        # strip full-line SQL comments so a leading comment block doesn't mask the
        # CREATE that follows it in the same split chunk
        body = "\n".join(ln for ln in chunk.splitlines() if not ln.strip().startswith("--")).strip()
        if body:
            rdb.execute(body)


# --------------------------------------------------------------------------
# Re-score core
# --------------------------------------------------------------------------

def rescore_all(*, limit: int | None = None, persist: bool = True) -> list[dict]:
    """Re-run every experiment through the corrected engine; return result rows.

    Panels are cached by (start, end, universe_n) so the ~1.9M-row prices table
    is loaded once per distinct (window, universe) rather than per experiment.
    """
    rows = rdb.query(
        """SELECT experiment_id, agent_id, strategy_family, params, universe,
                  lookback_start, lookback_end, sharpe, turnover, beats_benchmark, status
           FROM research_experiments ORDER BY sharpe DESC""")
    if limit:
        rows = rows[:limit]

    # Cache loaded panels so we don't re-hit the prices table per experiment.
    panel_cache: dict[tuple, object] = {}
    _orig_load = bt.load_price_panel

    def _cached_load(start, end, universe_n=60):
        key = (str(start), str(end), int(universe_n))
        if key not in panel_cache:
            panel_cache[key] = _orig_load(start, end, universe_n)
        return panel_cache[key]

    bt.load_price_panel = _cached_load  # type: ignore[assignment]
    results: list[dict] = []
    try:
        for i, r in enumerate(rows, 1):
            params = r["params"] if isinstance(r["params"], dict) else json.loads(r["params"] or "{}")
            fam = r["strategy_family"] or "equal_weight"
            un = _universe_n(r["universe"], params)
            start = _as_str_date(r["lookback_start"]) if r["lookback_start"] else "2018-01-01"
            end = _as_str_date(r["lookback_end"]) if r["lookback_end"] else "2024-12-31"
            try:
                m = bt.run_backtest(fam, params, start=start, end=end, universe_n=un)
                err = m.get("error")
            except Exception as e:  # keep going; record the failure
                m, err = {"data_caveats": bt.DATA_CAVEATS}, f"rescore exception: {e}"

            old_sharpe = float(r["sharpe"]) if r["sharpe"] is not None else None
            new_sharpe = m.get("sharpe")
            res = {
                "rescore_id": _uid("rsc"),
                "experiment_id": r["experiment_id"],
                "agent_id": r["agent_id"],
                "strategy_family": fam,
                "lookback_start": start,
                "lookback_end": end,
                "universe_n": un,
                "old_sharpe": old_sharpe,
                "new_sharpe": new_sharpe,
                "delta_sharpe": (new_sharpe - old_sharpe)
                                if (new_sharpe is not None and old_sharpe is not None) else None,
                "old_turnover": float(r["turnover"]) if r["turnover"] is not None else None,
                "new_turnover": m.get("turnover"),
                "old_beats_benchmark": int(r["beats_benchmark"]) if r["beats_benchmark"] is not None else None,
                "beats_benchmark_new": 1 if m.get("beats_benchmark") else 0,
                "turnover_cost_bps": m.get("turnover_cost_bps"),
                "slippage_bps": m.get("slippage_bps"),
                "all_in_cost_bps": m.get("all_in_cost_bps"),
                "gross_cost": m.get("gross_cost"),
                "params": params,
                "data_caveats": m.get("data_caveats") or bt.DATA_CAVEATS,
                "status": "failed" if err else "ok",
                "error": err,
            }
            results.append(res)
            if i % 20 == 0:
                print(f"  ... rescored {i}/{len(rows)}", flush=True)
    finally:
        bt.load_price_panel = _orig_load  # type: ignore[assignment]

    if persist:
        _persist(results)
    return results


def _persist(results: list[dict]) -> None:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")
    batch = [(
        r["rescore_id"], r["experiment_id"], r["agent_id"], r["strategy_family"],
        r["lookback_start"], r["lookback_end"], r["universe_n"],
        r["old_sharpe"], r["new_sharpe"], r["delta_sharpe"],
        r["old_turnover"], r["new_turnover"], r["old_beats_benchmark"],
        r["beats_benchmark_new"], r["turnover_cost_bps"], r["slippage_bps"],
        r["all_in_cost_bps"], r["gross_cost"], json.dumps(r["params"]),
        r["data_caveats"], r["status"], r["error"], ts,
    ) for r in results]
    rdb.executemany(
        """INSERT INTO experiment_rescores
           (rescore_id, experiment_id, agent_id, strategy_family, lookback_start,
            lookback_end, universe_n, old_sharpe, new_sharpe, delta_sharpe,
            old_turnover, new_turnover, old_beats_benchmark, beats_benchmark_new,
            turnover_cost_bps, slippage_bps, all_in_cost_bps, gross_cost, params,
            data_caveats, status, error, rescored_at)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
        batch,
    )


def _fmt(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "  n/a"


def report(results: list[dict]) -> None:
    ok = [r for r in results if r["status"] == "ok" and r["new_sharpe"] is not None]
    old_ranked = sorted(
        [r for r in results if r["old_sharpe"] is not None],
        key=lambda r: r["old_sharpe"], reverse=True)
    new_ranked = sorted(ok, key=lambda r: r["new_sharpe"], reverse=True)

    print("\n" + "=" * 108)
    print("RE-SCORE: corrected backtest engine (turnover cost honored + survivorship/look-ahead fix)")
    print("=" * 108)
    print(f"{'rank':>4}  {'family':<14} {'old_shp':>8} {'new_shp':>8} {'Δshp':>8} "
          f"{'old_turn':>9} {'new_turn':>9} {'all_in':>7} {'bb_new':>6}  experiment_id")
    print("-" * 108)
    for rank, r in enumerate(new_ranked[:15], 1):
        print(f"{rank:>4}  {r['strategy_family']:<14} {_fmt(r['old_sharpe']):>8} "
              f"{_fmt(r['new_sharpe']):>8} {_fmt(r['delta_sharpe']):>8} "
              f"{_fmt(r['old_turnover'],4):>9} {_fmt(r['new_turnover'],4):>9} "
              f"{_fmt(r['all_in_cost_bps'],1):>7} {str(bool(r['beats_benchmark_new'])):>6}  "
              f"{r['experiment_id']}")

    print("\n" + "-" * 108)
    print("RANKING CHANGE (the point of the exercise)")
    print("-" * 108)
    if old_ranked:
        o = old_ranked[0]
        print(f"OLD #1 : {o['strategy_family']:<14} old_sharpe={_fmt(o['old_sharpe'])}  "
              f"params={json.dumps(o['params'])[:120]}")
        # where does the OLD #1 land in the NEW ranking?
        pos = next((k for k, r in enumerate(new_ranked, 1)
                    if r["experiment_id"] == o["experiment_id"]), None)
        onew = next((r for r in ok if r["experiment_id"] == o["experiment_id"]), None)
        if onew is not None and pos is not None:
            print(f"         -> under corrected costs: new_sharpe={_fmt(onew['new_sharpe'])} "
                  f"(Δ={_fmt(onew['delta_sharpe'])}), now ranks #{pos} of {len(new_ranked)}.")
        else:
            print("         -> failed / dropped under the corrected engine.")
    if new_ranked:
        n = new_ranked[0]
        print(f"NEW #1 : {n['strategy_family']:<14} new_sharpe={_fmt(n['new_sharpe'])}  "
              f"(old_sharpe={_fmt(n['old_sharpe'])}, all_in={_fmt(n['all_in_cost_bps'],1)}bps)  "
              f"params={json.dumps(n['params'])[:120]}")

    # The specific "3.66 regime daily 10bps" winner
    def _is_the_old_winner(r):
        p = r["params"]
        return (r["strategy_family"] == "regime"
                and int(p.get("rebalance_days", 0) or 0) == 1
                and (p.get("turnover_cost_bps") == 10 or p.get("turnover_cost_bps") == 10.0)
                and r["old_sharpe"] is not None and abs(r["old_sharpe"] - 3.663) < 0.01)
    winners = [r for r in ok if _is_the_old_winner(r)]
    if winners:
        w = sorted(winners, key=lambda r: r["new_sharpe"], reverse=True)[0]
        pos = next((k for k, r in enumerate(new_ranked, 1)
                    if r["experiment_id"] == w["experiment_id"]), None)
        print("\nThe old 'Sharpe 3.66 regime daily (10bps)' winner:")
        print(f"  old_sharpe=3.663 -> new_sharpe={_fmt(w['new_sharpe'])} "
              f"(Δ={_fmt(w['delta_sharpe'])}), all_in={_fmt(w['all_in_cost_bps'],1)}bps, "
              f"now #{pos} of {len(new_ranked)}.")

    n_ok = len(ok)
    flipped = [r for r in ok if r["old_beats_benchmark"] is not None
               and bool(r["old_beats_benchmark"]) != bool(r["beats_benchmark_new"])]
    print(f"\nRe-scored {len(results)} experiments ({n_ok} ok). "
          f"{len(flipped)} flipped beats_benchmark under corrected costs.")
    if ok:
        print(f"data_caveats: {ok[0]['data_caveats']}")


def main(argv=None):
    ap = argparse.ArgumentParser("rescore")
    ap.add_argument("--apply-schema", action="store_true",
                    help="create the experiment_rescores table first")
    ap.add_argument("--limit", type=int, default=0, help="0 = all experiments")
    ap.add_argument("--no-persist", action="store_true", help="do not write experiment_rescores")
    args = ap.parse_args(argv)

    if args.apply_schema:
        print("applying rescore_schema.sql ...", flush=True)
        apply_schema()
    print("re-scoring experiments through corrected engine ...", flush=True)
    results = rescore_all(limit=(args.limit or None), persist=not args.no_persist)
    report(results)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    main()
