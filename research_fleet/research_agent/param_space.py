"""Parameter search space for the strategy sweep engine.

Defines, per strategy family, the realistic grid of knobs the sweep explores.
Every knob here is one the CORRECTED backtester actually reads (see
``backtest._weights_for`` + ``run_backtest``): ``lookback_days``, ``skip_days``,
``top_n``, ``bottom_n``, ``reversal_days``, ``keep_n``, ``target_vol``,
``ma_days``, ``rebalance_days``, ``w_max``, ``universe_n``, plus the cost keys
``turnover_cost_bps`` / ``slippage_bps``. Nothing here invents a param the engine
would silently ignore.

Three entry points:

  * :func:`grid(family)`   — full Cartesian product for one family (an iterator).
  * :func:`sample(family, n, seed)` — DETERMINISTIC random search: given the same
    ``seed`` it returns the same ``n`` de-duplicated configs, using a private
    ``random.Random(seed)`` (NEVER numpy global state) so reruns reproduce and it
    does not perturb any other RNG in the process.
  * :func:`plan(target_n, seed)` — a de-duplicated, cross-family list of
    ``(family, params)`` of size ~``target_n``: it takes the FULL grid for families
    whose grid is small, and random-samples families whose grid explodes, so the
    plan is spread across all families instead of being swamped by the one family
    with the biggest Cartesian product.

Cost dimension: ``turnover_cost_bps ∈ {5, 10}`` is swept for every family so cost
sensitivity is visible in the results (a high-turnover config that only looks good
at 5bps is exposed at 10bps). ``slippage_bps`` is left at the engine default.
"""

from __future__ import annotations

import itertools
import json
import random
from typing import Any, Iterator


# The 8 families the backtester dispatches on (backtest._weights_for).
FAMILIES: tuple[str, ...] = (
    "equal_weight", "momentum", "mean_reversion", "vol_target",
    "low_vol", "factor", "risk_parity", "regime",
)

# Dimensions shared by (nearly) every family. rebalance cadence, position cap and
# universe width materially change turnover/cost and breadth, so they are swept
# broadly; the cost knob is swept so cost sensitivity shows up in the sweep.
REBALANCE_DAYS = (5, 21, 63)
W_MAX = (0.05, 0.10, 0.20)
UNIVERSE_N = (30, 60, 100)
TURNOVER_COST_BPS = (5, 10)

# Per-family knob grids. Only the knobs a family actually reads are varied; the
# shared dims above are layered on in _family_axes so each family's grid stays the
# Cartesian product of exactly the params that change its behavior.
_FAMILY_KNOBS: dict[str, dict[str, tuple]] = {
    # equal_weight ignores selection knobs entirely — only cadence/cap/universe
    # and cost move its results, so its grid is deliberately the smallest.
    "equal_weight": {},
    "momentum": {
        "lookback_days": (63, 126, 189, 252),
        "skip_days": (0, 21),
        "top_n": (5, 10, 20, 30),
    },
    "mean_reversion": {
        "lookback_days": (63, 126),
        "reversal_days": (3, 5, 10),
        "bottom_n": (5, 10, 20, 30),
    },
    "vol_target": {
        "lookback_days": (63, 126),
        "target_vol": (0.08, 0.10, 0.12, 0.15),
    },
    "low_vol": {
        "lookback_days": (63, 126, 252),
        "keep_n": (10, 20, 30),
    },
    "factor": {
        "lookback_days": (63, 126, 252),
        "keep_n": (10, 20, 30),
    },
    "risk_parity": {
        "lookback_days": (63, 126, 252),
    },
    "regime": {
        "lookback_days": (126, 252),
        "ma_days": (50, 100, 150, 200),
    },
}


def _family_axes(family: str) -> dict[str, tuple]:
    """The full ordered set of swept axes for a family (family knobs + shared dims).

    ``w_max`` is only meaningful for families that cap+normalize weights
    (low_vol/factor/risk_parity) — for the others it is a no-op in the engine, so
    we do NOT sweep it there (it would only inflate the grid with duplicate-behavior
    configs). Every family sweeps cadence, universe width and the cost knob.
    """
    if family not in _FAMILY_KNOBS:
        raise ValueError(f"unknown family: {family!r}")
    axes: dict[str, tuple] = dict(_FAMILY_KNOBS[family])
    axes["rebalance_days"] = REBALANCE_DAYS
    if family in ("low_vol", "factor", "risk_parity"):
        axes["w_max"] = W_MAX
    axes["universe_n"] = UNIVERSE_N
    axes["turnover_cost_bps"] = TURNOVER_COST_BPS
    return axes


def _canon(params: dict) -> str:
    """Canonical JSON for a params dict — used to hash/de-dup identical configs."""
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def grid(family: str) -> Iterator[dict[str, Any]]:
    """Yield every params dict in the full Cartesian product for ``family``."""
    axes = _family_axes(family)
    keys = list(axes.keys())
    for combo in itertools.product(*(axes[k] for k in keys)):
        yield dict(zip(keys, combo))


def grid_size(family: str) -> int:
    """Size of the full Cartesian product for ``family`` (no enumeration cost)."""
    axes = _family_axes(family)
    n = 1
    for vals in axes.values():
        n *= len(vals)
    return n


def total_grid_size() -> dict[str, int]:
    """Full-grid count per family plus an ``overall`` total (sum across families)."""
    out = {fam: grid_size(fam) for fam in FAMILIES}
    out["overall"] = sum(out[fam] for fam in FAMILIES)
    return out


def sample(family: str, n: int, seed: int) -> list[dict[str, Any]]:
    """Return up to ``n`` DE-DUPLICATED random configs for ``family``.

    Deterministic given ``seed``: uses a private ``random.Random(seed + i)`` per
    draw (varying by index) so the sequence is reproducible AND does not touch the
    numpy/global RNG. If the family's full grid is smaller than ``n`` we just return
    the whole grid (can't draw more unique configs than exist).
    """
    axes = _family_axes(family)
    keys = list(axes.keys())
    full = grid_size(family)
    if n >= full:
        return list(grid(family))

    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    i = 0
    # Draw with a per-index seeded RNG so reruns reproduce exactly; keep drawing
    # (advancing i) until we have n uniques or exhaust a generous attempt budget.
    max_attempts = n * 50 + 1000
    while len(out) < n and i < max_attempts:
        rng = random.Random(f"{seed}:{family}:{i}")
        params = {k: rng.choice(axes[k]) for k in keys}
        key = _canon(params)
        if key not in seen:
            seen.add(key)
            out.append(params)
        i += 1
    return out


def plan(target_n: int, seed: int,
         families: list[str] | None = None) -> list[tuple[str, dict[str, Any]]]:
    """Produce a de-duplicated ``[(family, params), ...]`` list of size ~``target_n``.

    Strategy: split the budget evenly across the requested families. For a family
    whose FULL grid fits in its share, take the whole grid (exhaustive is better
    than random when it's cheap); for a family whose grid explodes, random-sample
    its share via :func:`sample`. Any budget left over by small families (which
    can't fill their share) is redistributed to the families that still have room.
    Every ``(family, params)`` is de-duplicated on canonical JSON, so the returned
    list is unique and spread across families rather than dominated by the one with
    the largest Cartesian product.
    """
    fams = list(families) if families else list(FAMILIES)
    fams = [f for f in fams if f in _FAMILY_KNOBS]
    if not fams or target_n <= 0:
        return []

    # First pass: even share per family, capped at each family's full grid size.
    base = max(1, target_n // len(fams))
    alloc: dict[str, int] = {}
    for f in fams:
        alloc[f] = min(base, grid_size(f))

    # Redistribute the remainder to families that still have grid room, round-robin,
    # so we hit ~target_n even when some families are grid-limited.
    assigned = sum(alloc.values())
    remaining = target_n - assigned
    room = {f: grid_size(f) - alloc[f] for f in fams}
    while remaining > 0 and any(room[f] > 0 for f in fams):
        for f in fams:
            if remaining <= 0:
                break
            if room[f] > 0:
                alloc[f] += 1
                room[f] -= 1
                remaining -= 1

    out: list[tuple[str, dict[str, Any]]] = []
    seen: set[tuple[str, str]] = set()
    for f in fams:
        want = alloc[f]
        if want <= 0:
            continue
        configs = (list(grid(f)) if want >= grid_size(f)
                   else sample(f, want, seed))
        for p in configs:
            key = (f, _canon(p))
            if key not in seen:
                seen.add(key)
                out.append((f, p))
    return out
