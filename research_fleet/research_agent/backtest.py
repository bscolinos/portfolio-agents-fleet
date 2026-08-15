"""Lightweight strategy backtester over the sp500 prices already in SingleStore.

The research agents propose strategy parameters (as JSON); this evaluates them
on real daily returns from the ``prices`` table and returns standard metrics
(Sharpe, vol, drawdown, turnover, CVaR) plus a 1/N benchmark comparison. Pure
NumPy/pandas — runs fine on a tiny CPU EC2.

Supported strategy families (dispatched on ``strategy_family``):
  equal_weight, momentum (cross-sectional N-M), mean_reversion (short-term
  reversal), vol_target (overlay), low_vol / factor (low-volatility selection),
  risk_parity (inverse-vol), regime (trend filter on the equal-weight proxy).
The agent can also pass a generic ``weights_rule`` for custom experiments.

--------------------------------------------------------------------------
Cost model (see ``resolve_cost_bps`` / ``_apply_weights``)
--------------------------------------------------------------------------
This is a REAL-MONEY decision engine, so the transaction-cost model mirrors the
live trade path (``pa_agents/trading.py``) rather than a token 2bps placeholder.

The live path charges, per fill, a commission (``$0.005``/share, ``$1`` min) plus
``slippage_bps`` (2bps default) of market impact on the traded notional. The
backtester works in *weight space* (no share counts), so we collapse that into an
all-in round-trip cost in bps charged on the traded notional Σ|Δw| at each
rebalance:

    all_in_cost_bps = turnover_cost_bps (commission proxy) + slippage_bps (impact)

Assumptions / mapping:
  * ``turnover_cost_bps`` is the agent-declared commission/spread estimate. The
    research agents emit this key (NOT ``tc_bps``); the OLD engine only read
    ``tc_bps`` and so silently charged the 2bps default on strategies that
    declared 10bps — materially inflating the Sharpe of high-turnover (e.g.
    daily-rebalanced) strategies. We now prefer ``turnover_cost_bps``, fall back
    to ``tc_bps``, then to a defensible institutional default of 5bps round-trip
    (a commission+spread estimate for large-cap US equities; the old 2.0 was
    optimistic for anything but the most liquid names).
  * ``slippage_bps`` is the modeled market-impact leg (default 2bps, matching
    ``CostModel.slippage_bps`` in the live module). It is charged on turnover too.
  * Per-rebalance cost = turnover * (all_in_cost_bps / 1e4), where turnover is the
    actual Σ|Δw| traded that day — never a constant.

--------------------------------------------------------------------------
Universe construction (see ``load_price_panel``) — bias notes
--------------------------------------------------------------------------
The OLD loader picked "the N most-complete names over [start,end]" then did
``.dropna(thresh=95%).ffill().dropna()``. That is BOTH survivorship-biased (only
names present across the FULL future window were eligible) AND look-ahead
(end-of-window completeness chose the universe as of the start). For a real-money
decision that systematically overstates returns.

The corrected loader:
  * loads every name that has ANY price in [start,end] (no full-window
    completeness gate), so names that later gap-out / delist stay eligible on the
    dates they trade;
  * forward-fills only within a short bounded gap (``MAX_FFILL_GAP`` trading days)
    to bridge sporadic missing prints — it NEVER carries a price across a long
    gap / delisting to fabricate a flat return;
  * defers universe *selection* to each rebalance date, using only trailing
    history available AS OF that date (``eligible_as_of``): a name is tradable
    at ``t`` iff it has a valid price at ``t`` and ≥ ``lookback`` trailing
    observations ending at ``t``. No future information enters selection.

HONESTY / RESIDUAL LIMITATION: we have only a ``prices`` table and a *current*
``securities.is_active`` flag — there is NO historical point-in-time index-
membership table, so a TRUE constituent list is not reconstructable. This engine
implements the best available approximation (trailing-history + as-of
availability). The residual bias is surfaced in the returned metrics as
``data_caveats`` and must not be read as true point-in-time membership.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import research_db as rdb


ANN = 252.0

# Institutional round-trip cost defaults (bps on traded notional Σ|Δw|). These
# back-stop agent-declared params and mirror the live trade module's cost legs.
DEFAULT_TURNOVER_COST_BPS = 5.0   # commission/spread proxy (was an optimistic 2.0)
DEFAULT_SLIPPAGE_BPS = 2.0        # market impact; == CostModel.slippage_bps live

# Max consecutive trading-day gap we will forward-fill to bridge a sporadic
# missing print. Longer absences are treated as not-tradable (delisting/gap-out)
# and left as NaN so the name simply drops out on those dates — we NEVER ffill
# across a long gap to fabricate a flat return.
MAX_FFILL_GAP = 3

# Honest description of the residual bias baked into the universe (no PIT
# index-membership table exists on this data). Surfaced in run_backtest metrics.
DATA_CAVEATS = (
    "Universe approximated from the prices table + a CURRENT securities.is_active "
    "snapshot; no point-in-time index-membership table exists, so true historical "
    "constituents are not reconstructable. Eligibility uses as-of trailing history "
    "(no full-window completeness gate, no ffill across long gaps), which removes "
    "the worst survivorship/look-ahead bias but may still include a name that was "
    "not an index member on a given date or omit one that was."
)


def resolve_cost_bps(params: dict) -> tuple[float, float, float]:
    """Resolve (turnover_cost_bps, slippage_bps, all_in_cost_bps) from params.

    Prefers the agents' declared ``turnover_cost_bps``, falls back to legacy
    ``tc_bps``, then to :data:`DEFAULT_TURNOVER_COST_BPS`. Adds a modeled
    ``slippage_bps`` leg (default :data:`DEFAULT_SLIPPAGE_BPS`) so backtest cost
    ≈ commission + slippage like the live trade path. See module docstring.
    """
    if params.get("turnover_cost_bps") is not None:
        turn_bps = float(params["turnover_cost_bps"])
    elif params.get("tc_bps") is not None:
        turn_bps = float(params["tc_bps"])
    else:
        turn_bps = DEFAULT_TURNOVER_COST_BPS
    slip_bps = float(params.get("slippage_bps", DEFAULT_SLIPPAGE_BPS))
    return turn_bps, slip_bps, turn_bps + slip_bps


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_price_panel(start: str, end: str, universe_n: int = 60) -> pd.DataFrame:
    """Return a (dates x tickers) adj_close panel, survivorship/look-ahead-safe.

    Loads EVERY name with any price in ``[start, end]`` (no full-window
    completeness gate — that was the survivorship + look-ahead bug), capped to the
    ``universe_n`` names with the most trailing history so the panel stays small
    on a tiny EC2. Alignment forward-fills only within a short bounded gap
    (:data:`MAX_FFILL_GAP` trading days) to bridge sporadic missing prints; longer
    absences (delisting / gap-out) are left NaN so the name is simply not tradable
    on those dates — we NEVER ffill across a long gap to fabricate a flat return.

    Point-in-time universe *selection* happens later, per rebalance date, in
    :func:`eligible_as_of` using only history available as of that date. See the
    module docstring for the residual limitation (no PIT membership table).
    """
    # Rank candidates by raw coverage only to CAP panel width — this is a size
    # guard, not a survivorship filter: a name absent late in the window is still
    # loaded and stays tradable on the dates it has data. is_active is a current
    # snapshot (documented caveat), used only as a stable tie-break, never a gate.
    top = rdb.query(
        """SELECT p.ticker, COUNT(*) c FROM prices p
           WHERE p.trade_date BETWEEN %s AND %s
           GROUP BY p.ticker ORDER BY c DESC, p.ticker LIMIT %s""",
        (start, end, int(universe_n)))
    tickers = [r["ticker"] for r in top]
    if not tickers:
        return pd.DataFrame()
    rows = rdb.query(
        "SELECT ticker, trade_date, adj_close FROM prices "
        "WHERE trade_date BETWEEN %s AND %s AND ticker IN (%s) ORDER BY trade_date"
        % ("%s", "%s", ",".join(["%s"] * len(tickers))),
        [start, end, *tickers])
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    panel = df.pivot_table(index="trade_date", columns="ticker", values="adj_close").sort_index()
    # Bounded forward-fill ONLY: bridge gaps up to MAX_FFILL_GAP trading days;
    # leave longer absences NaN (a delisted/gapped name drops out, not carried
    # flat). No trailing .dropna() over columns/rows — that reintroduced
    # survivorship by requiring presence across the whole future window.
    panel = panel.ffill(limit=MAX_FFILL_GAP)
    return panel


def eligible_as_of(panel: pd.DataFrame, i: int, lookback: int) -> list[str]:
    """Names tradable as of ``panel.index[i]`` using ONLY trailing information.

    A name is eligible iff it has a valid price at ``t = index[i]`` AND at least
    ``lookback`` non-NaN observations in the trailing window ending at ``t``. No
    future data enters this decision — this is the survivorship/look-ahead fix.
    """
    if i < 0 or i >= len(panel):
        return []
    lo = max(0, i - lookback)
    window = panel.iloc[lo:i + 1]
    at_t = panel.iloc[i]
    trailing_obs = window.notna().sum()
    eligible = [c for c in panel.columns
                if pd.notna(at_t[c]) and int(trailing_obs[c]) >= min(lookback, len(window))]
    return eligible


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def _metrics(port_ret: pd.Series, turnover_series: pd.Series | None = None,
             bench_ret: pd.Series | None = None) -> dict:
    port_ret = port_ret.dropna()
    if len(port_ret) < 5:
        return {"sharpe": None, "ann_return": None, "ann_vol": None}
    mean, std = port_ret.mean(), port_ret.std()
    ann_ret = float(mean * ANN)
    ann_vol = float(std * np.sqrt(ANN))
    sharpe = float(ann_ret / ann_vol) if ann_vol > 1e-9 else 0.0
    downside = port_ret[port_ret < 0].std()
    sortino = float(ann_ret / (downside * np.sqrt(ANN))) if downside and downside > 1e-9 else None
    curve = (1 + port_ret).cumprod()
    dd = float((curve / curve.cummax() - 1).min())
    var95 = float(np.percentile(port_ret, 5))
    cvar95 = float(port_ret[port_ret <= var95].mean()) if (port_ret <= var95).any() else var95
    win = float((port_ret > 0).mean())
    out = {
        "ann_return": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe, "sortino": sortino,
        "max_drawdown": dd, "cvar_95": cvar95, "win_rate": win,
        "turnover": float(turnover_series.mean()) if turnover_series is not None and len(turnover_series) else 0.0,
        "total_return": float(curve.iloc[-1] - 1),
    }
    if bench_ret is not None and len(bench_ret.dropna()) > 5:
        b = bench_ret.dropna()
        bvol = b.std() * np.sqrt(ANN)
        out["benchmark_sharpe"] = float((b.mean() * ANN) / bvol) if bvol > 1e-9 else 0.0
        out["beats_benchmark"] = bool(sharpe > out["benchmark_sharpe"])
    return out


def _apply_weights(returns: pd.DataFrame, weights_by_date: dict[pd.Timestamp, pd.Series],
                   cost_bps: float = DEFAULT_TURNOVER_COST_BPS + DEFAULT_SLIPPAGE_BPS
                   ) -> tuple[pd.Series, pd.Series]:
    """Given target weights on rebalance dates, compute daily portfolio returns
    (holding weights constant between rebalances) net of transaction cost.

    ``cost_bps`` is the ALL-IN round-trip cost (commission/spread + slippage) in
    bps, charged on the ACTUAL traded notional Σ|Δw| at each rebalance:
    ``tcost = turnover * (cost_bps / 1e4)``. It is never a constant per-day drag —
    only rebalance days incur cost, proportional to that day's turnover.

    A held name whose return is NaN on a date (a delisting / long gap the panel
    left NaN) contributes 0 P&L that day (treated as idle capital) — we do NOT
    fabricate a flat-then-jump return by carrying a stale price across the gap.
    """
    dates = returns.index
    rebal_dates = sorted(weights_by_date)
    cur = pd.Series(0.0, index=returns.columns)
    port, turn = [], []
    ri = 0
    for d in dates:
        if ri < len(rebal_dates) and d >= rebal_dates[ri]:
            new = weights_by_date[rebal_dates[ri]].reindex(returns.columns).fillna(0.0)
            t = float((new - cur).abs().sum())
            turn.append((d, t))
            cur = new
            ri += 1
        else:
            turn.append((d, 0.0))
        # NaN return (untradable name) => 0 contribution, not a fabricated return.
        day_ret = returns.loc[d].fillna(0.0)
        r = float((cur * day_ret).sum())
        # subtract transaction cost on the turnover actually traded this day
        tcost = turn[-1][1] * (cost_bps / 1e4)
        port.append((d, r - tcost))
    ps = pd.Series(dict(port))
    ts = pd.Series(dict(turn))
    return ps, ts


# --------------------------------------------------------------------------
# Strategy dispatch
# --------------------------------------------------------------------------

def run_backtest(strategy_family: str, params: dict, *, start: str, end: str,
                 universe_n: int = 60) -> dict:
    """Evaluate a strategy; returns metrics dict (+ benchmark comparison).

    Cost is resolved from the agents' declared ``turnover_cost_bps`` (falling back
    to legacy ``tc_bps`` then a 5bps default) PLUS a modeled ``slippage_bps`` leg,
    charged on actual turnover — see :func:`resolve_cost_bps`. The universe is
    selected per rebalance from names with valid as-of trailing history
    (:func:`eligible_as_of`), not full-window completeness. Adds ``all_in_cost_bps``,
    ``gross_cost`` and a ``data_caveats`` string to the returned metrics.
    """
    panel = load_price_panel(start, end, universe_n)
    if panel.empty or panel.shape[1] < 5:
        return {"error": "insufficient price data", "sharpe": None, "data_caveats": DATA_CAVEATS}
    # Keep NaNs: a name absent on a date must stay NaN so it contributes nothing
    # (never a fabricated flat return). pct_change() naturally yields NaN across a
    # gap/reappearance; _apply_weights treats NaN as 0 contribution.
    rets = panel.pct_change()
    tickers = list(panel.columns)
    rebal_freq = max(1, int(params.get("rebalance_days", 21)))
    lookback = int(params.get("lookback_days", 126))
    wmax = float(params.get("w_max", 0.10))
    turn_bps, slip_bps, all_in_bps = resolve_cost_bps(params)

    rebal_idx = list(range(lookback, len(panel), rebal_freq))
    if not rebal_idx:
        rebal_idx = [min(lookback, len(panel) - 1)]

    # benchmark: equal-weight over the as-of-eligible names on the same grid
    bench_w: dict = {}
    for i in rebal_idx:
        elig = eligible_as_of(panel, i, lookback)
        if len(elig) < 5:
            continue
        bench_w[panel.index[i]] = pd.Series(1.0 / len(elig), index=elig).reindex(tickers).fillna(0.0)
    bench_ret, _ = _apply_weights(rets, bench_w, all_in_bps) if bench_w else (None, None)

    weights: dict = {}
    for i in rebal_idx:
        d = panel.index[i]
        elig = eligible_as_of(panel, i, lookback)
        if len(elig) < 5:
            continue
        lo = max(0, i - lookback)
        window = rets.iloc[lo:i][elig].dropna(axis=1, how="all")
        if len(window) < 10 or window.shape[1] < 5:
            continue
        elig = list(window.columns)
        w = _weights_for(strategy_family, params, window,
                         panel.iloc[lo:i + 1][elig], elig, wmax)
        if w is not None:
            weights[d] = w.reindex(tickers).fillna(0.0)

    if not weights:
        return {"error": "no weights produced", "sharpe": None, "data_caveats": DATA_CAVEATS}
    port_ret, turn = _apply_weights(rets, weights, all_in_bps)
    m = _metrics(port_ret, turn, bench_ret)
    m["n_rebalances"] = len(weights)
    m["universe_size"] = len(tickers)
    m["all_in_cost_bps"] = all_in_bps
    m["turnover_cost_bps"] = turn_bps
    m["slippage_bps"] = slip_bps
    # gross_cost: total cost drag over the backtest (Σ turnover * cost_bps/1e4)
    m["gross_cost"] = float(turn.sum() * (all_in_bps / 1e4)) if turn is not None else 0.0
    m["data_caveats"] = DATA_CAVEATS
    return m


def _cap_normalize(w: pd.Series, wmax: float) -> pd.Series:
    w = w.clip(lower=0)
    if w.sum() <= 0:
        return pd.Series(1.0 / len(w), index=w.index)
    w = w / w.sum()
    for _ in range(50):
        over = w > wmax
        if not over.any():
            break
        excess = (w[over] - wmax).sum()
        w[over] = wmax
        under = ~over
        if under.any() and w[under].sum() > 0:
            w[under] += excess * (w[under] / w[under].sum())
    return w / w.sum()


def _weights_for(family: str, params: dict, window: pd.DataFrame,
                 window_px: pd.DataFrame, tickers: list[str], wmax: float) -> pd.Series | None:
    n = len(tickers)
    if family in ("equal_weight", "benchmark"):
        return pd.Series(1.0 / n, index=tickers)

    if family == "momentum":
        skip = int(params.get("skip_days", 21))
        mom = window_px.iloc[-1] / window_px.iloc[0] - 1.0 if len(window_px) > skip else window.sum()
        top = int(params.get("top_n", max(5, n // 10)))
        picks = mom.sort_values(ascending=False).head(top).index
        return pd.Series(1.0 / len(picks), index=picks).reindex(tickers).fillna(0.0)

    if family == "mean_reversion":
        recent = window.iloc[-int(params.get("reversal_days", 5)):].sum()
        bottom = int(params.get("bottom_n", max(5, n // 10)))
        picks = recent.sort_values().head(bottom).index  # prior losers
        return pd.Series(1.0 / len(picks), index=picks).reindex(tickers).fillna(0.0)

    if family in ("low_vol", "factor"):
        vol = window.std()
        keep = int(params.get("keep_n", max(10, n // 2)))
        picks = vol.sort_values().head(keep).index  # lowest vol
        w = pd.Series(1.0 / len(picks), index=picks).reindex(tickers).fillna(0.0)
        return _cap_normalize(w, wmax)

    if family == "risk_parity":
        vol = window.std().clip(lower=1e-6)
        w = (1.0 / vol)
        return _cap_normalize(w, wmax)

    if family in ("vol_target", "regime"):
        # base = equal weight; scaling handled at portfolio level via a cash blend
        base = pd.Series(1.0 / n, index=tickers)
        target_vol = float(params.get("target_vol", 0.10))
        realized = float(window.mean(axis=1).std() * np.sqrt(ANN)) or 1e-6
        if family == "vol_target":
            scale = min(1.0, target_vol / realized)
        else:  # regime: invest only if equal-weight proxy above its MA
            proxy = window_px.mean(axis=1)
            ma = proxy.rolling(int(params.get("ma_days", 200)), min_periods=20).mean()
            scale = 1.0 if (len(ma.dropna()) and proxy.iloc[-1] > ma.iloc[-1]) else 0.0
        return base * scale

    return None
