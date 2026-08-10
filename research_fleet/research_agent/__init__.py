"""Auto-research agent package: OpenClaw-through-NemoClaw agents that research
and test S&P 500 strategies, persisting hypotheses/experiments/findings to
SingleStore over time (with Qwen VECTOR recall) and, when configured, querying
via a real Aura Analyst Portal domain."""

__all__ = ["research_db", "backtest", "analyst", "llm_driver", "agent_loop"]
