"""Kill-switch tests. Live DB, cleaned up after.

  * engage(global)  -> is_engaged(any agent) True with reason
  * release(global) -> False
  * per-agent engage only trips that agent
"""

from __future__ import annotations

import pytest

from pa_agents import db, kill_switch

TEST_AGENT = "test-agent-DELETEME"
OTHER_AGENT = "other-agent-DELETEME"


def _cleanup():
    """Remove every test-scoped switch + audit row we may have written."""
    try:
        db.execute("DELETE FROM kill_switches WHERE scope IN (%s,%s,'global')",
                   (TEST_AGENT, OTHER_AGENT))
        db.execute("DELETE FROM trade_audit WHERE event_type='KILL_SWITCH' "
                   "AND (agent_id IN (%s,%s) OR actor='pytest')",
                   (TEST_AGENT, OTHER_AGENT))
    except Exception:
        pass


@pytest.fixture(autouse=True)
def clean_around():
    _cleanup()
    yield
    _cleanup()


def test_global_engage_trips_any_agent():
    kill_switch.engage("global", reason="test global trip", by="pytest")
    st = kill_switch.is_engaged(TEST_AGENT)
    assert st["engaged"] is True
    assert st["scope"] == "global"
    assert st["reason"] == "test global trip"
    # a totally unrelated agent is also tripped by the global switch
    assert kill_switch.is_engaged("some-random-agent")["engaged"] is True


def test_global_release_clears():
    kill_switch.engage("global", reason="test", by="pytest")
    assert kill_switch.is_engaged(TEST_AGENT)["engaged"] is True
    kill_switch.release("global", by="pytest")
    st = kill_switch.is_engaged(TEST_AGENT)
    assert st["engaged"] is False
    assert st["scope"] is None


def test_per_agent_engage_scoped():
    kill_switch.engage(TEST_AGENT, reason="misbehaving", by="pytest")
    # only that agent is tripped...
    trip = kill_switch.is_engaged(TEST_AGENT)
    assert trip["engaged"] is True
    assert trip["scope"] == TEST_AGENT
    assert trip["reason"] == "misbehaving"
    # ...a different agent is NOT tripped (no global switch engaged)
    assert kill_switch.is_engaged(OTHER_AGENT)["engaged"] is False
    # and with no agent context at all, only global matters => clear
    assert kill_switch.is_engaged(None)["engaged"] is False


def test_engage_requires_reason():
    with pytest.raises(ValueError):
        kill_switch.engage(TEST_AGENT, reason="", by="pytest")


def test_status_lists_rows():
    kill_switch.engage(TEST_AGENT, reason="r", by="pytest")
    rows = kill_switch.status()
    scopes = {r["scope"] for r in rows}
    assert TEST_AGENT in scopes
