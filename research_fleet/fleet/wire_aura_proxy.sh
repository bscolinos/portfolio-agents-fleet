#!/usr/bin/env bash
# Wire ONE research node to the hosted Aura proxy:
#   - ship updated analyst.py + agent_loop.py
#   - set ANALYST_PROXY_URL/ANALYST_PROXY_TOKEN in the node .env (drop raw Aura key)
#   - apply the OpenShell aura-proxy egress policy to the sandbox
#   - restart the research loop
# Usage: wire_aura_proxy.sh <ip> <agent_id>
set -euo pipefail
IP="$1"; SB="$2"
PREP=/Users/billscolinos/Documents/code_factory/staging/portfolio-agents-prep
PEM="$PREP/../research-fleet-key.pem"
PROXY_URL="http://172.31.12.154:8799"
TOKEN=$(tr -d '\n' < "$PREP/aura/.proxy_token")
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -i $PEM ubuntu@$IP"
SCP="scp -o StrictHostKeyChecking=no -i $PEM"

echo "== [$SB @ $IP] ship updated analyst.py + agent_loop.py =="
$SCP "$PREP/research_agent/analyst.py"   ubuntu@$IP:/tmp/analyst.py
$SCP "$PREP/research_agent/agent_loop.py" ubuntu@$IP:/tmp/agent_loop.py
$SCP "$PREP/fleet/policy/aura-proxy.yaml" ubuntu@$IP:/tmp/aura-proxy.yaml

$SSH "PROXY_URL='$PROXY_URL' TOKEN='$TOKEN' bash -s" <<'REMOTE'
set -eu
sudo cp /tmp/analyst.py    /opt/research-agent/research_agent/analyst.py
sudo cp /tmp/agent_loop.py /opt/research-agent/research_agent/agent_loop.py
sudo chown ubuntu:ubuntu /opt/research-agent/research_agent/analyst.py /opt/research-agent/research_agent/agent_loop.py
# update .env: drop any raw Aura + old proxy lines, add proxy vars
sudo sed -i '/^ANALYST_API_URL=/d;/^ANALYST_API_KEY=/d;/^ANALYST_PROXY_URL=/d;/^ANALYST_PROXY_TOKEN=/d' /opt/research-agent/.env
printf 'ANALYST_PROXY_URL=%s\nANALYST_PROXY_TOKEN=%s\n' "$PROXY_URL" "$TOKEN" | sudo tee -a /opt/research-agent/.env >/dev/null
sudo systemctl restart research-agent.service
sleep 3
systemctl is-active research-agent.service && echo AGENT_ACTIVE || sudo journalctl -u research-agent.service --no-pager -n 15
REMOTE

echo "== apply OpenShell aura-proxy egress policy to the sandbox =="
$SSH "SB='$SB' bash -s" <<'REMOTE' || echo "  (sandbox policy step nonzero — host loop still reaches proxy directly)"
set -eu
export PATH="$HOME/.local/bin:$HOME/.npm-global/bin:/usr/bin:$PATH"
if nemoclaw list 2>/dev/null | grep -q "$SB"; then
  nemoclaw "$SB" policy add --from-file /tmp/aura-proxy.yaml --trusted-private-host 172.31.12.154 --yes 2>&1 | tail -2 || echo "policy add nonzero"
  nemoclaw "$SB" exec --no-tty -- bash -lc 'curl -sf -m 6 http://172.31.12.154:8799/health >/dev/null && echo SANDBOX_CAN_REACH_PROXY || echo SANDBOX_NO_PROXY' 2>&1 | tail -1
else
  echo "sandbox $SB not registered; skipping policy"
fi
REMOTE
echo "== [$SB @ $IP] wired to proxy =="
