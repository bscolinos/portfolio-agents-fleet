#!/usr/bin/env bash
# Deploy the templated write tool + focus-specific prompts to one research node:
#   - ship the updated research_agent package + tool_server + skill
#   - install/start the tool-server systemd service (:11510)
#   - install the OpenClaw skill into the sandbox so the agent knows the tool
#   - restart the research loop (now prompt-driven + tool-only writes)
#
# Usage: deploy_tool_and_prompts.sh <ip> <agent_id> <focus>
set -euo pipefail
IP="$1"; AGENT_ID="$2"; FOCUS="${3:-}"
PREP=/Users/billscolinos/Documents/code_factory/staging/portfolio-agents-prep
PEM="$PREP/../research-fleet-key.pem"
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -i $PEM ubuntu@$IP"
SCP="scp -o StrictHostKeyChecking=no -i $PEM"

echo "== [$AGENT_ID @ $IP] ship updated package + tool server + skill =="
tar -czf /tmp/ra-$AGENT_ID.tgz -C "$PREP" research_agent
$SCP /tmp/ra-$AGENT_ID.tgz ubuntu@$IP:/tmp/ra.tgz
$SCP "$PREP/fleet/tool_server.py" ubuntu@$IP:/tmp/tool_server.py
$SCP "$PREP/fleet/skill/SKILL.md" ubuntu@$IP:/tmp/SKILL.md
$SCP "$PREP/fleet/policy/research-tool.yaml" ubuntu@$IP:/tmp/research-tool.yaml

$SSH 'bash -s' <<'REMOTE'
set -eux
sudo tar -xzf /tmp/ra.tgz -C /opt/research-agent
sudo cp /tmp/tool_server.py /opt/research-agent/tool_server.py
sudo chown -R ubuntu:ubuntu /opt/research-agent
# deps already present in venv; ensure requests for imds
/opt/research-agent/venv/bin/pip -q install requests >/dev/null 2>&1 || true

# tool-server systemd unit (:11510) — the sandbox-facing uniform write path
sudo tee /etc/systemd/system/research-tool-server.service >/dev/null <<EOF
[Unit]
Description=Research templated write-tool HTTP server
After=network-online.target
Wants=network-online.target
[Service]
User=ubuntu
WorkingDirectory=/opt/research-agent
EnvironmentFile=/opt/research-agent/.env
Environment=AGENT_HOME=/opt/research-agent
Environment=TOOL_PORT=11510
ExecStart=/opt/research-agent/venv/bin/python /opt/research-agent/tool_server.py
Restart=always
RestartSec=5
[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now research-tool-server.service
sleep 3
curl -sf http://127.0.0.1:11510/health && echo " TOOLSERVER_UP" || echo " TOOLSERVER_DOWN"
REMOTE

echo "== apply research-tool egress policy + install skill + verify sandbox reach =="
$SSH "SB='$AGENT_ID' bash -s" <<'REMOTE' || echo "  (sandbox step nonzero — non-fatal; host loop still writes via the tool)"
set -eu
export PATH="$HOME/.local/bin:$HOME/.npm-global/bin:/usr/bin:$PATH"
SB="$SB"
if nemoclaw list 2>/dev/null | grep -q "$SB"; then
  # 1) allow the sandbox to egress to the tool server on :11510
  nemoclaw "$SB" policy add --from-file /tmp/research-tool.yaml --yes 2>&1 | tail -2 || echo "policy add nonzero"
  # 2) install the write-tool skill into the sandbox
  nemoclaw "$SB" exec --no-tty -- bash -lc 'mkdir -p /sandbox/.agents/skills/singlestore-research-writer' 2>/dev/null || true
  nemoclaw "$SB" upload /tmp/SKILL.md /sandbox/.agents/skills/singlestore-research-writer/SKILL.md 2>&1 | tail -1 || echo "skill upload nonzero"
  # 3) verify the sandbox can now reach the tool server
  nemoclaw "$SB" exec --no-tty -- bash -lc 'curl -sf -m 6 http://host.openshell.internal:11510/health >/dev/null && echo SANDBOX_CAN_REACH_TOOL || echo SANDBOX_NO_TOOL' 2>&1 | tail -1
else
  echo "sandbox $SB not registered; skipping policy/skill"
fi
REMOTE

echo "== restart research loop (prompt-driven, tool-only writes) =="
$SSH "sudo systemctl restart research-agent.service; sleep 3; systemctl is-active research-agent.service && echo AGENT_ACTIVE || sudo journalctl -u research-agent.service --no-pager -n 15"
echo "== [$AGENT_ID @ $IP] deploy done =="
