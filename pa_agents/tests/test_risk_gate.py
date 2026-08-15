"""Risk-gate tests. Inputs are constructed in-memory; the gate's DB-backed
checks (drawdown/daily-loss) tolerate an EMPTY history for a fresh test agent, so
we pass current_weights + nav explicitly and rely on that empty-history no-op.

Covered:
  (a) a weight >10% is rejected (and clipped+renormalized when configured)
  (b) turnover >cap rejected
  (c) gross >100% / any short rejected when shorts disallowed
  (d) kill-switch engaged -> hard reject regardless of weights
  (e) a clean, small, well-diversified rebalance is APPROVED

All decisions are written to trade_audit + risk_decisions under the test agent;
we clean those rows up afterwards.
"""

from __future__ import annotations

import pytest

from pa_agents import db, kill_switch
from pa_agents.risk_gate import evaluate, RiskLimits

TEST_AGENT = "test-agent-DELETEME"
AS_OF = "2024-12-31"

# A clean, diversified 10-name book at 8% each (gross 80%, max 8%).
CLEAN_WEIGHTS = {f"T{i}": 0.08 for i in range(10)}
PRICES = {f"T{i}": 100.0 for i in range(20)}
NAV = 100_000_000.0


def _codes(dec):
    return {v["code"] for v in dec.violations}


def _cleanup():
    try:
        db.execute("DELETE FROM risk_decisions WHERE agent_id=%s", (TEST_AGENT,))
        db.execute("DELETE FROM trade_audit WHERE agent_id=%s AND event_type='RISK_CHECK'",
                   (TEST_AGENT,))
        db.execute("DELETE FROM kill_switches WHERE scope IN (%s,'global')", (TEST_AGENT,))
    except Exception:
        pass


@pytest.fixture(autouse=True)
def clean_around():
    _cleanup()
    yield
    _cleanup()


def test_clean_rebalance_approved():
    # Already holding the diversified book; a modest drift-correction rebalance:
    # move T0 8%->6% and T1 8%->10%. Turnover = 0.04 (well under the 40% cap).
    target = dict(CLEAN_WEIGHTS)
    target["T0"] = 0.06
    target["T1"] = 0.10
    dec = evaluate(agent_id=TEST_AGENT, as_of_date=AS_OF,
                   target_weights=target, prices=PRICES,
                   nav=NAV, current_weights=CLEAN_WEIGHTS)
    assert dec.approved is True, dec.reason
    assert dec.violations == []
    assert dec.checks["turnover"] == pytest.approx(0.04)
    # notional guard: max ~2% of 100M = 2M per name, under the 25M default cap
    assert dec.checks["gross_exposure"] == pytest.approx(0.80)


def test_overweight_rejected_by_default():
    w = dict(CLEAN_WEIGHTS)
    w["T0"] = 0.25  # 25% single name > 10% cap
    dec = evaluate(agent_id=TEST_AGENT, as_of_date=AS_OF, target_weights=w,
                   prices=PRICES, nav=NAV, current_weights={})
    assert dec.approved is False
    assert "MAX_POSITION_WEIGHT" in _codes(dec)


def test_overweight_clipped_when_configured():
    w = {"T0": 0.30, "T1": 0.30, "T2": 0.20, "T3": 0.20}  # gross 1.0
    # relax turnover so this test isolates the clip+renormalize behavior
    lim = RiskLimits(clip_over_weight=True, min_position_weight=0.0, max_turnover=1.0)
    dec = evaluate(agent_id=TEST_AGENT, as_of_date=AS_OF, target_weights=w,
                   prices=PRICES, nav=NAV, current_weights={}, limits=lim)
    assert dec.approved is True, dec.reason
    assert dec.adjusted_weights is not None
    # every adjusted weight is at/under the 10% cap
    assert max(dec.adjusted_weights.values()) <= lim.max_position_weight + 1e-6


def test_turnover_over_cap_rejected():
    # currently all cash; buying 80% gross => turnover 0.80 > 0.40 cap
    dec = evaluate(agent_id=TEST_AGENT, as_of_date=AS_OF, target_weights=CLEAN_WEIGHTS,
                   prices=PRICES, nav=NAV, current_weights={},
                   limits=RiskLimits(max_turnover=0.40))
    assert dec.approved is False
    assert "MAX_TURNOVER" in _codes(dec)


def test_gross_over_100_rejected():
    w = {f"T{i}": 0.10 for i in range(15)}  # gross 150%
    dec = evaluate(agent_id=TEST_AGENT, as_of_date=AS_OF, target_weights=w,
                   prices=PRICES, nav=NAV, current_weights={})
    assert dec.approved is False
    assert "GROSS_EXPOSURE" in _codes(dec)


def test_short_rejected_when_disallowed():
    w = dict(CLEAN_WEIGHTS)
    w["T0"] = -0.05  # a short
    dec = evaluate(agent_id=TEST_AGENT, as_of_date=AS_OF, target_weights=w,
                   prices=PRICES, nav=NAV, current_weights={})
    assert dec.approved is False
    assert "SHORTS_DISALLOWED" in _codes(dec)


def test_kill_switch_hard_rejects_regardless_of_weights():
    kill_switch.engage("global", reason="test emergency", by="pytest")
    try:
        dec = evaluate(agent_id=TEST_AGENT, as_of_date=AS_OF,
                       target_weights=CLEAN_WEIGHTS, prices=PRICES,
                       nav=NAV, current_weights={})
        assert dec.approved is False
        assert "KILL_SWITCH" in _codes(dec)
        assert "KILL SWITCH ENGAGED" in dec.reason
    finally:
        kill_switch.release("global", by="pytest")


def test_missing_price_rejected():
    w = dict(CLEAN_WEIGHTS)
    w["NOPRICE"] = 0.05
    dec = evaluate(agent_id=TEST_AGENT, as_of_date=AS_OF, target_weights=w,
                   prices=PRICES, nav=NAV, current_weights={})
    assert dec.approved is False
    assert "PRICE_SANITY" in _codes(dec)


def test_missing_nav_fails_closed():
    # nav explicitly 0 => fail-closed, even for otherwise-clean weights
    dec = evaluate(agent_id=TEST_AGENT, as_of_date=AS_OF, target_weights=CLEAN_WEIGHTS,
                   prices=PRICES, nav=0.0, current_weights={})
    assert dec.approved is False
    assert "NAV_MISSING" in _codes(dec)


def test_notional_guard_flags_large_order():
    # one 10% name on a 1B NAV => 100M order > 25M default cap
    w = {"T0": 0.10, "T1": 0.10, "T2": 0.10}
    dec = evaluate(agent_id=TEST_AGENT, as_of_date=AS_OF, target_weights=w,
                   prices=PRICES, nav=1_000_000_000.0, current_weights={},
                   limits=RiskLimits(max_turnover=1.0))  # relax turnover to isolate
    assert dec.approved is False
    assert "MAX_NAME_NOTIONAL" in _codes(dec)


def test_decision_persisted_to_risk_decisions():
    evaluate(agent_id=TEST_AGENT, as_of_date=AS_OF, target_weights=CLEAN_WEIGHTS,
             prices=PRICES, nav=NAV, current_weights={})
    rows = db.query("SELECT approved, reason FROM risk_decisions WHERE agent_id=%s",
                    (TEST_AGENT,))
    assert len(rows) >= 1
