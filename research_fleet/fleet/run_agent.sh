#!/usr/bin/env bash
# Launcher for one auto-research agent. Invoked by the systemd unit on the EC2.
# Runs the research loop continuously; the loop drives OpenClaw-through-NemoClaw
# for reasoning and writes results to SingleStore.
set -euo pipefail

AGENT_HOME=/opt/research-agent
cd "$AGENT_HOME"

# shellcheck disable=SC1091
[ -f "$AGENT_HOME/.env" ] && set -a && . "$AGENT_HOME/.env" && set +a

export AGENT_ID="${AGENT_ID:-research-01}"
export AGENT_NAME="${AGENT_NAME:-Researcher ${AGENT_ID}}"
export AGENT_FOCUS="${AGENT_FOCUS:-}"
export AGENT_MODEL="${AGENT_MODEL:-sonnet}"
export PYTHONUNBUFFERED=1

VENV="$AGENT_HOME/venv"
exec "$VENV/bin/python" -m research_agent.agent_loop \
  --agent-id "$AGENT_ID" \
  --display-name "$AGENT_NAME" \
  --focus "$AGENT_FOCUS" \
  --model "$AGENT_MODEL" \
  --idle-sleep 180
