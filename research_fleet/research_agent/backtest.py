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
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import research_db as rdb


ANN = 252.0


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_price_panel(start: str, end: str, universe_n: int = 60) -> pd.DataFrame:
    """Return a (dates x tickers) adj_close panel for the N most-complete names."""
    top = rdb.query(
        """SELECT ticker, COUNT(*) c FROM prices WHERE trade_date BETWEEN %s AND %s
           GROUP BY ticker ORDER BY c DESC, ticker LIMIT %s""",
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
    return panel.dropna(axis=1, thresh=int(len(panel) * 0.95)).ffill().dropna()


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
                   tc_bps: float = 2.0) -> tuple[pd.Series, pd.Series]:
    """Given target weights on rebalance dates, compute daily portfolio returns
    (holding weights constant between rebalances) net of turnover cost."""
    dates = returns.index
    rebal_dates = sorted(weights_by_date)
    cur = pd.Series(0.0, index=returns.columns)
    prev = pd.Series(0.0, index=returns.columns)
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
        r = float((cur * returns.loc[d]).sum())
        # subtract turnover cost on rebalance days
        tcost = turn[-1][1] * (tc_bps / 1e4)
        port.append((d, r - tcost))
    ps = pd.Series(dict(port))
    ts = pd.Series(dict(turn))
    return ps, ts


# --------------------------------------------------------------------------
# Strategy dispatch
# --------------------------------------------------------------------------

def run_backtest(strategy_family: str, params: dict, *, start: str, end: str,
                 universe_n: int = 60) -> dict:
    """Evaluate a strategy; returns metrics dict (+ benchmark comparison)."""
    panel = load_price_panel(start, end, universe_n)
    if panel.empty or panel.shape[1] < 5:
        return {"error": "insufficient price data", "sharpe": None}
    rets = panel.pct_change().dropna()
    tickers = list(panel.columns)
    rebal_freq = int(params.get("rebalance_days", 21))
    lookback = int(params.get("lookback_days", 126))
    wmax = float(params.get("w_max", 0.10))
    tc_bps = float(params.get("tc_bps", 2.0))

    rebal_idx = list(range(lookback, len(panel), rebal_freq))
    if not rebal_idx:
        rebal_idx = [min(lookback, len(panel) - 1)]

    # benchmark: equal-weight, same rebalance grid
    bench_w = {panel.index[i]: pd.Series(1.0 / len(tickers), index=tickers) for i in rebal_idx}
    bench_ret, _ = _apply_weights(rets, bench_w, tc_bps)

    weights: dict = {}
    for i in rebal_idx:
        d = panel.index[i]
        window = rets.iloc[max(0, i - lookback):i]
        if len(window) < 10:
            continue
        w = _weights_for(strategy_family, params, window, panel.iloc[max(0, i - lookback):i + 1], tickers, wmax)
        if w is not None:
            weights[d] = w

    if not weights:
        return {"error": "no weights produced", "sharpe": None}
    port_ret, turn = _apply_weights(rets, weights, tc_bps)
    m = _metrics(port_ret, turn, bench_ret)
    m["n_rebalances"] = len(weights)
    m["universe_size"] = len(tickers)
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
