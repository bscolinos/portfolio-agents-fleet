#!/usr/bin/env bash
# Runs ON a research node (via ssh) to onboard the genuine NVIDIA OpenShell
# sandbox running OpenClaw, wired to the local inference shim. Idempotent-ish:
# skips if the sandbox already exists. Arg: sandbox/agent name (e.g. research-03).
set -euxo pipefail
SB="${1:?sandbox name required}"
export PATH="$HOME/.local/bin:$HOME/.npm-global/bin:/usr/bin:$PATH"

# already onboarded? then just ensure it's running.
if nemoclaw list 2>/dev/null | grep -q "$SB"; then
  echo "sandbox $SB already registered"
  nemoclaw "$SB" start 2>/dev/null || true
  nemoclaw "$SB" status 2>&1 | head -8 || true
  exit 0
fi

HAIKU_KEY="$(sudo grep -m1 '^HAIKU_KEY=' /opt/research-agent/.env | cut -d= -f2-)"
export NEMOCLAW_NON_INTERACTIVE=1 NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE=1 \
  NEMOCLAW_AGENT=openclaw NEMOCLAW_SANDBOX_NAME="$SB" NEMOCLAW_PROVIDER=custom \
  NEMOCLAW_PREFERRED_API=openai-completions \
  NEMOCLAW_ENDPOINT_URL="http://host.openshell.internal:11500/v1" \
  NEMOCLAW_MODEL=haiku COMPATIBLE_API_KEY="$HAIKU_KEY" \
  NEMOCLAW_POLICY_TIER=balanced NEMOCLAW_WEB_SEARCH_PROVIDER=none \
  NEMOCLAW_TRUSTED_PRIVATE_HOSTS="host.openshell.internal" NEMOCLAW_SANDBOX_GPU=0

nemoclaw onboard --non-interactive --yes --yes-i-accept-third-party-software \
  --agent openclaw --name "$SB" --no-gpu --no-sandbox-gpu

echo "=== onboarded; status ==="
nemoclaw list 2>&1 | head -8
nemoclaw "$SB" status 2>&1 | head -10
# quick proof turn
nemoclaw "$SB" agent --agent main -m "Reply with exactly: OPENCLAW_LIVE_${SB}" --json 2>&1 | head -20 || echo "proof turn nonzero"
