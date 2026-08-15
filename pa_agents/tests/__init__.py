"""Tests for the portfolio-agents safety layer (risk gate + kill switch).

These run against the LIVE SingleStore demo DB (that's fine) but use a
test-scoped agent_id ('test-agent-DELETEME') and clean up all rows they write.
"""
