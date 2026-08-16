"""Scalable parameter-sweep engine over the CORRECTED strategy backtester.

The single-strategy backtester (``backtest.run_backtest``) is the correct unit of
work; this is the driver that runs it THOUSANDS of times efficiently, with proper
in-sample / out-of-sample discipline, and persists every trial to SingleStore.

--------------------------------------------------------------------------
What it does
--------------------------------------------------------------------------
  * Builds a de-duplicated, cross-family plan of ``(family, params)`` via
    :func:`param_space.plan`.
  * Evaluates EACH config TWICE with the SAME params — once on the IN-SAMPLE
    window and once on the OUT-OF-SAMPLE window — so an overfit config (great IS,
    poor OOS) is visible. The overfitting signal ``is_oos_sharpe_gap = is_sharpe -
    oos_sharpe`` is stored on every row.
  * Persists every trial (both windows) to ``sweep_results`` and tracks the run in
    ``sweep_runs`` (status flips running -> done).

--------------------------------------------------------------------------
Panel caching + PARITY GUARANTEE (the important part)
--------------------------------------------------------------------------
``run_backtest`` re-loads the ~1.9M-row prices panel on EVERY call. Running it
thousands of times would hammer the DB with thousands of identical loads. Instead
we pre-load each distinct ``(start, end, universe_n)`` panel ONCE via
``backtest.load_price_panel`` (same caching idea as ``rescore.py``) and evaluate
each config against the already-loaded, in-memory panel WITHOUT re-querying.

:func:`_eval_on_panel` is a line-for-line replica of ``run_backtest``'s compute
loop (everything AFTER the ``load_price_panel`` call), calling the SAME internal
helpers — ``eligible_as_of`` / ``_weights_for`` / ``_apply_weights`` / ``_metrics``
/ ``resolve_cost_bps`` — in the SAME order on the SAME ``panel.pct_change()``. The
ONLY difference from ``run_backtest`` is that the panel is passed in rather than
loaded. Therefore, for any config, ``_eval_on_panel(fam, params, panel)`` yields
metrics IDENTICAL to ``run_backtest(fam, params, start, end, universe_n)`` when
``panel == load_price_panel(start, end, universe_n)``. The test suite asserts this
parity to within 1e-9 on a sample of configs; keep the two in lockstep if either
changes.

Run from the demo root:
    python -m research_fleet.research_agent.sweep plan --target 2000 --seed 7
    python -m research_fleet.research_agent.sweep run  --target 2000 --seed 7
    python -m research_fleet.research_agent.sweep run  --target 50 --seed 7 --limit 12
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import research_db as rdb
from . import backtest as bt
from . import param_space as ps


# Default IS/OOS split — both windows sit INSIDE the 2005-2024 price history.
# IS trains, OOS validates; a config that only shines IS is overfit.
DEFAULT_IS_START = "2010-01-01"
DEFAULT_IS_END = "2019-12-31"
DEFAULT_OOS_START = "2020-01-01"
DEFAULT_OOS_END = "2024-12-31"
DEFAULT_UNIVERSE_N = 60

INSERT_BATCH = 200          # rows per executemany flush
PROGRESS_EVERY = 100        # print a progress line every N configs


def _uid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


# --------------------------------------------------------------------------
# Schema (create-if-not-exists so the engine + tests run standalone)
# --------------------------------------------------------------------------

# NOTE: these DDLs match, column-for-column, the schema agreed with the agent that
# owns sweep_schema.sql + sweep_analyze.py. Both sides CREATE IF NOT EXISTS with
# identical DDL, so whichever runs first wins and the other is a no-op. Do NOT
# change a column name/type here without changing it there — the analyzer reads
# these rows.
_DDL_SWEEP_RESULTS = """
CREATE TABLE IF NOT EXISTS sweep_results (
    result_id           VARCHAR(64)  NOT NULL,
    sweep_id            VARCHAR(64)  NOT NULL,
    family              VARCHAR(64),
    params              JSON,
    is_sharpe           DOUBLE,
    is_ann_return       DOUBLE,
    is_ann_vol          DOUBLE,
    is_max_drawdown     DOUBLE,
    is_turnover         DOUBLE,
    is_beats_benchmark  TINYINT,
    oos_sharpe          DOUBLE,
    oos_ann_return      DOUBLE,
    oos_ann_vol         DOUBLE,
    oos_max_drawdown    DOUBLE,
    oos_turnover        DOUBLE,
    oos_beats_benchmark TINYINT,
    is_oos_sharpe_gap   DOUBLE,
    all_in_cost_bps     DOUBLE,
    universe_n          INT,
    error               TEXT,
    created_at          DATETIME(6),
    SHARD KEY (result_id),
    SORT KEY (sweep_id, oos_sharpe)
)
"""

_DDL_SWEEP_RUNS = """
CREATE TABLE IF NOT EXISTS sweep_runs (
    sweep_id     VARCHAR(64)  NOT NULL,
    target_n     INT,
    actual_n     INT,
    seed         INT,
    is_start     DATE,
    is_end       DATE,
    oos_start    DATE,
    oos_end      DATE,
    families     TEXT,
    status       VARCHAR(24),
    started_at   DATETIME(6),
    finished_at  DATETIME(6),
    notes        TEXT,
    PRIMARY KEY (sweep_id)
)
"""


def ensure_tables() -> None:
    """Create sweep_results + sweep_runs if absent (idempotent, standalone-safe)."""
    rdb.execute(_DDL_SWEEP_RESULTS)
    rdb.execute(_DDL_SWEEP_RUNS)


# --------------------------------------------------------------------------
# Panel cache
# --------------------------------------------------------------------------

class PanelCache:
    """Loads each distinct (start, end, universe_n) panel ONCE via
    ``backtest.load_price_panel`` and hands back the in-memory DataFrame on repeat
    asks — so a sweep of thousands of configs hits the prices table only once per
    distinct (window, universe), not once per config."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str, int], pd.DataFrame] = {}
        self.loads = 0

    def get(self, start: str, end: str, universe_n: int) -> pd.DataFrame:
        key = (str(start), str(end), int(universe_n))
        if key not in self._cache:
            self._cache[key] = bt.load_price_panel(start, end, int(universe_n))
            self.loads += 1
        return self._cache[key]


# --------------------------------------------------------------------------
# In-memory evaluation — PARITY replica of run_backtest's compute loop
# --------------------------------------------------------------------------

def _eval_on_panel(strategy_family: str, params: dict, panel: pd.DataFrame) -> dict:
    """Evaluate a config against an ALREADY-LOADED panel.

    Byte-for-byte the same computation as ``backtest.run_backtest`` AFTER its
    ``load_price_panel`` call — same helpers, same order, same NaN handling — so
    metrics are IDENTICAL to ``run_backtest(...)`` on the panel's window. See the
    module docstring's PARITY GUARANTEE. The only reason this is duplicated rather
    than calling ``run_backtest`` is to skip the (cached) DB load.
    """
    if panel is None or panel.empty or panel.shape[1] < 5:
        return {"error": "insufficient price data", "sharpe": None,
                "data_caveats": bt.DATA_CAVEATS}

    rets = panel.pct_change()
    tickers = list(panel.columns)
    rebal_freq = max(1, int(params.get("rebalance_days", 21)))
    lookback = int(params.get("lookback_days", 126))
    wmax = float(params.get("w_max", 0.10))
    turn_bps, slip_bps, all_in_bps = bt.resolve_cost_bps(params)

    rebal_idx = list(range(lookback, len(panel), rebal_freq))
    if not rebal_idx:
        rebal_idx = [min(lookback, len(panel) - 1)]

    # benchmark: equal-weight over the as-of-eligible names on the same grid
    bench_w: dict = {}
    for i in rebal_idx:
        elig = bt.eligible_as_of(panel, i, lookback)
        if len(elig) < 5:
            continue
        bench_w[panel.index[i]] = pd.Series(
            1.0 / len(elig), index=elig).reindex(tickers).fillna(0.0)
    bench_ret, _ = bt._apply_weights(rets, bench_w, all_in_bps) if bench_w else (None, None)

    weights: dict = {}
    for i in rebal_idx:
        d = panel.index[i]
        elig = bt.eligible_as_of(panel, i, lookback)
        if len(elig) < 5:
            continue
        lo = max(0, i - lookback)
        window = rets.iloc[lo:i][elig].dropna(axis=1, how="all")
        if len(window) < 10 or window.shape[1] < 5:
            continue
        elig = list(window.columns)
        w = bt._weights_for(strategy_family, params, window,
                            panel.iloc[lo:i + 1][elig], elig, wmax)
        if w is not None:
            weights[d] = w.reindex(tickers).fillna(0.0)

    if not weights:
        return {"error": "no weights produced", "sharpe": None,
                "data_caveats": bt.DATA_CAVEATS}
    port_ret, turn = bt._apply_weights(rets, weights, all_in_bps)
    m = bt._metrics(port_ret, turn, bench_ret)
    m["n_rebalances"] = len(weights)
    m["universe_size"] = len(tickers)
    m["all_in_cost_bps"] = all_in_bps
    m["turnover_cost_bps"] = turn_bps
    m["slippage_bps"] = slip_bps
    m["gross_cost"] = float(turn.sum() * (all_in_bps / 1e4)) if turn is not None else 0.0
    m["data_caveats"] = bt.DATA_CAVEATS
    return m


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

_INSERT_RESULT_SQL = """
INSERT INTO sweep_results
   (result_id, sweep_id, family, params,
    is_sharpe, is_ann_return, is_ann_vol, is_max_drawdown, is_turnover, is_beats_benchmark,
    oos_sharpe, oos_ann_return, oos_ann_vol, oos_max_drawdown, oos_turnover, oos_beats_benchmark,
    is_oos_sharpe_gap, all_in_cost_bps, universe_n, error, created_at)
VALUES (%s,%s,%s,%s, %s,%s,%s,%s,%s,%s, %s,%s,%s,%s,%s,%s, %s,%s,%s,%s,%s)
"""


def _beats(m: dict) -> int | None:
    v = m.get("beats_benchmark")
    return None if v is None else (1 if v else 0)


def _result_row(sweep_id: str, family: str, params: dict,
                is_m: dict, oos_m: dict, universe_n: int,
                error: str | None) -> tuple:
    """Flatten one config's IS + OOS metrics into a sweep_results insert tuple."""
    is_sharpe = is_m.get("sharpe")
    oos_sharpe = oos_m.get("sharpe")
    gap = (is_sharpe - oos_sharpe) if (is_sharpe is not None and oos_sharpe is not None) else None
    # all_in_cost_bps is a property of the params (same both windows); take whichever exists.
    all_in = is_m.get("all_in_cost_bps")
    if all_in is None:
        all_in = oos_m.get("all_in_cost_bps")
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")
    return (
        _uid("swr"), sweep_id, family, json.dumps(params, sort_keys=True),
        is_sharpe, is_m.get("ann_return"), is_m.get("ann_vol"),
        is_m.get("max_drawdown"), is_m.get("turnover"), _beats(is_m),
        oos_sharpe, oos_m.get("ann_return"), oos_m.get("ann_vol"),
        oos_m.get("max_drawdown"), oos_m.get("turnover"), _beats(oos_m),
        gap, all_in, int(universe_n), error, ts,
    )


def _flush(batch: list[tuple]) -> None:
    if batch:
        rdb.executemany(_INSERT_RESULT_SQL, batch)
        batch.clear()


# --------------------------------------------------------------------------
# The sweep
# --------------------------------------------------------------------------

def run_sweep(*, target_n: int, seed: int,
              is_start: str = DEFAULT_IS_START, is_end: str = DEFAULT_IS_END,
              oos_start: str = DEFAULT_OOS_START, oos_end: str = DEFAULT_OOS_END,
              universe_default: int = DEFAULT_UNIVERSE_N,
              sweep_id: str | None = None, families: list[str] | None = None,
              limit: int | None = None, max_workers: int = 1) -> dict:
    """Run the parameter sweep IS + OOS and persist every trial.

    Args:
      target_n: desired number of (family, params) configs (see param_space.plan).
      seed: makes the plan + any random sampling deterministic/reproducible.
      is_start/is_end: in-sample window; oos_start/oos_end: out-of-sample window.
      universe_default: universe_n used for a config that does not carry its own.
      sweep_id: reuse a specific id (else a fresh 'swp-<hex>' is generated) — handy
        for tests that want to clean up their own rows.
      families: restrict the plan to these families (default: all 8).
      limit: cap the plan to the first ``limit`` configs (smoke runs).
      max_workers: reserved; single-process is correct + the DB loads are cached.

    Returns a summary dict (sweep_id, planned/written/errored counts, elapsed, rate,
    panel loads, and the top-3 configs by OOS Sharpe with their IS-OOS gap).
    """
    ensure_tables()
    sweep_id = sweep_id or _uid("swp")
    fam_label = ",".join(families) if families else "all"

    configs = ps.plan(target_n, seed, families=families)
    if limit:
        configs = configs[:int(limit)]
    planned = len(configs)

    # Record the run row up front (status=running).
    rdb.execute(
        """INSERT INTO sweep_runs
             (sweep_id, target_n, actual_n, seed, is_start, is_end, oos_start, oos_end,
              families, status, started_at, notes)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,'running',NOW(6),%s)""",
        (sweep_id, int(target_n), 0, int(seed), is_start, is_end, oos_start, oos_end,
         fam_label, f"universe_default={universe_default}"),
    )

    cache = PanelCache()
    batch: list[tuple] = []
    written = 0
    errored = 0
    top: list[tuple[float, str, dict, float | None]] = []  # (oos_sharpe, family, params, gap)
    t0 = time.time()

    try:
        for k, (family, params) in enumerate(configs, 1):
            un = int(params.get("universe_n", universe_default) or universe_default)
            err = None
            is_m: dict = {}
            oos_m: dict = {}
            try:
                is_panel = cache.get(is_start, is_end, un)
                oos_panel = cache.get(oos_start, oos_end, un)
                is_m = _eval_on_panel(family, params, is_panel)
                oos_m = _eval_on_panel(family, params, oos_panel)
                # a per-window compute error (insufficient data / no weights) is
                # recorded, but does not kill the sweep.
                errs = [m.get("error") for m in (is_m, oos_m) if m.get("error")]
                if errs:
                    err = "; ".join(f"{w}: {e}" for w, e in
                                    zip(("is", "oos"), [is_m.get("error"), oos_m.get("error")]) if e)
            except Exception as e:  # never let one bad config kill the sweep
                err = f"sweep exception: {e}"

            if err:
                errored += 1
            batch.append(_result_row(sweep_id, family, params, is_m, oos_m, un, err))
            written += 1

            oss = oos_m.get("sharpe")
            if oss is not None:
                iss = is_m.get("sharpe")
                gap = (iss - oss) if iss is not None else None
                top.append((float(oss), family, params, gap))

            if len(batch) >= INSERT_BATCH:
                _flush(batch)
            if k % PROGRESS_EVERY == 0:
                el = time.time() - t0
                rate = k / el if el > 0 else 0.0
                print(f"  ... {k}/{planned} configs  ({errored} err)  "
                      f"{el:.1f}s  {rate:.1f}/s  panel_loads={cache.loads}", flush=True)

        _flush(batch)
        status = "done"
    except BaseException:
        _flush(batch)  # persist what we have, then mark the run failed + re-raise
        rdb.execute(
            "UPDATE sweep_runs SET status='failed', actual_n=%s, finished_at=NOW(6) WHERE sweep_id=%s",
            (written, sweep_id))
        raise

    rdb.execute(
        "UPDATE sweep_runs SET status=%s, actual_n=%s, finished_at=NOW(6) WHERE sweep_id=%s",
        (status, written, sweep_id))

    elapsed = time.time() - t0
    top.sort(key=lambda r: r[0], reverse=True)
    summary = {
        "sweep_id": sweep_id,
        "planned": planned,
        "written": written,
        "errored": errored,
        "elapsed_s": round(elapsed, 2),
        "rate_per_s": round(written / elapsed, 2) if elapsed > 0 else None,
        "panel_loads": cache.loads,
        "is_window": [is_start, is_end],
        "oos_window": [oos_start, oos_end],
        "top3_by_oos_sharpe": [
            {"family": f, "oos_sharpe": round(s, 4),
             "is_oos_gap": (round(g, 4) if g is not None else None),
             "params": p}
            for s, f, p, g in top[:3]
        ],
    }
    return summary


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _print_plan(target_n: int, seed: int, families: list[str] | None) -> None:
    from collections import Counter
    configs = ps.plan(target_n, seed, families=families)
    by_fam = Counter(f for f, _ in configs)
    tg = ps.total_grid_size()
    print(f"plan preview: target={target_n} seed={seed} "
          f"families={','.join(families) if families else 'all'}")
    print(f"  planned configs: {len(configs)} (unique, de-duped)")
    print(f"  {'family':<16} {'planned':>8} {'full_grid':>10}")
    print(f"  {'-'*16} {'-'*8} {'-'*10}")
    for fam in ps.FAMILIES:
        if families and fam not in families:
            continue
        print(f"  {fam:<16} {by_fam.get(fam, 0):>8} {tg[fam]:>10}")
    print(f"  {'overall':<16} {len(configs):>8} {tg['overall']:>10}")


def _fmt(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "n/a"


def main(argv=None):
    ap = argparse.ArgumentParser("sweep",
                                 description="Parameter-sweep engine over the backtester (IS/OOS).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def _add_windows(p):
        p.add_argument("--target", type=int, default=2000, help="desired config count")
        p.add_argument("--seed", type=int, default=7, help="deterministic plan seed")
        p.add_argument("--families", type=str, default="",
                       help="comma-separated subset, e.g. momentum,regime")
        p.add_argument("--is-start", default=DEFAULT_IS_START)
        p.add_argument("--is-end", default=DEFAULT_IS_END)
        p.add_argument("--oos-start", default=DEFAULT_OOS_START)
        p.add_argument("--oos-end", default=DEFAULT_OOS_END)

    p_plan = sub.add_parser("plan", help="print plan size/breakdown WITHOUT running")
    p_plan.add_argument("--target", type=int, default=2000)
    p_plan.add_argument("--seed", type=int, default=7)
    p_plan.add_argument("--families", type=str, default="")

    p_run = sub.add_parser("run", help="run the sweep (IS + OOS) and persist")
    _add_windows(p_run)
    p_run.add_argument("--universe-default", type=int, default=DEFAULT_UNIVERSE_N)
    p_run.add_argument("--limit", type=int, default=0, help="cap to first N configs (smoke)")
    p_run.add_argument("--sweep-id", default="", help="reuse a specific sweep_id")

    args = ap.parse_args(argv)
    families = [f.strip() for f in args.families.split(",") if f.strip()] or None

    if args.cmd == "plan":
        _print_plan(args.target, args.seed, families)
        return

    if args.cmd == "run":
        print(f"running sweep: target={args.target} seed={args.seed} "
              f"IS=[{args.is_start}..{args.is_end}] OOS=[{args.oos_start}..{args.oos_end}] "
              f"families={','.join(families) if families else 'all'}"
              f"{' limit='+str(args.limit) if args.limit else ''}", flush=True)
        s = run_sweep(
            target_n=args.target, seed=args.seed,
            is_start=args.is_start, is_end=args.is_end,
            oos_start=args.oos_start, oos_end=args.oos_end,
            universe_default=args.universe_default,
            families=families, limit=(args.limit or None),
            sweep_id=(args.sweep_id or None))
        print("\n" + "=" * 78)
        print(f"SWEEP DONE  sweep_id={s['sweep_id']}")
        print(f"  planned={s['planned']}  written={s['written']}  errored={s['errored']}  "
              f"panel_loads={s['panel_loads']}")
        print(f"  elapsed={s['elapsed_s']}s  rate={s['rate_per_s']}/s")
        print(f"  IS={s['is_window']}  OOS={s['oos_window']}")
        print("  top 3 by OOS Sharpe:")
        for r in s["top3_by_oos_sharpe"]:
            print(f"    {r['family']:<14} oos_sharpe={_fmt(r['oos_sharpe'])} "
                  f"is_oos_gap={_fmt(r['is_oos_gap'])}  params={json.dumps(r['params'])[:110]}")
        print("=" * 78)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    main()
