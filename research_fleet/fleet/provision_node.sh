#!/usr/bin/env bash
# Provision ONE already-booted research node over SSH:
#   ship agent package + shim + per-node .env -> venv+deps -> shim systemd ->
#   NemoClaw non-interactive (OpenClaw, custom provider -> shim) -> agent systemd.
#
# Usage: provision_node.sh <public_ip> <agent_id> <agent_name> <focus> <model>
set -euo pipefail
IP="$1"; AGENT_ID="$2"; AGENT_NAME="$3"; FOCUS="${4:-}"; MODEL="${5:-sonnet}"
PREP=/Users/billscolinos/Documents/code_factory/staging/portfolio-agents-prep
PEM="$PREP/../research-fleet-key.pem"
DEMO_ENV=/Users/billscolinos/Documents/code_factory/demos/portfolio-agents/.env
NEMOCLAW_REF="cae757edca1e9996dba57d1d6b85f2d0ab1b23bb"
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -i $PEM ubuntu@$IP"
SCP="scp -o StrictHostKeyChecking=no -i $PEM"

echo "== [$AGENT_ID @ $IP] wait for base bootstrap =="
for i in $(seq 1 40); do
  if $SSH 'test -f /var/log/base-bootstrap.done' 2>/dev/null; then echo "  base ready"; break; fi
  sleep 15
done

echo "== ship agent package + shim + .env =="
tar -czf /tmp/research_agent-$AGENT_ID.tgz -C "$PREP" research_agent
$SCP /tmp/research_agent-$AGENT_ID.tgz ubuntu@$IP:/tmp/research_agent.tgz
$SCP "$PREP/fleet/inference_shim.py" ubuntu@$IP:/tmp/inference_shim.py
# build a per-node .env = demo .env + AGENT_* + optional ANALYST_*
cp "$DEMO_ENV" /tmp/node-$AGENT_ID.env
{
  echo "AGENT_ID=$AGENT_ID"
  echo "AGENT_NAME=$AGENT_NAME"
  echo "AGENT_FOCUS=$FOCUS"
  echo "AGENT_MODEL=$MODEL"
} >> /tmp/node-$AGENT_ID.env
$SCP /tmp/node-$AGENT_ID.env ubuntu@$IP:/tmp/node.env

echo "== lay down files + venv =="
$SSH 'bash -s' <<'REMOTE'
set -eux
sudo mkdir -p /opt/research-agent
sudo tar -xzf /tmp/research_agent.tgz -C /opt/research-agent
sudo cp /tmp/inference_shim.py /opt/research-agent/inference_shim.py
sudo cp /tmp/node.env /opt/research-agent/.env
sudo chmod 600 /opt/research-agent/.env
sudo chown -R ubuntu:ubuntu /opt/research-agent
python3 -m venv /opt/research-agent/venv
/opt/research-agent/venv/bin/pip -q install --upgrade pip
/opt/research-agent/venv/bin/pip -q install singlestoredb openai boto3 botocore pandas numpy requests
REMOTE

echo "== inference shim systemd =="
$SSH 'sudo bash -s' <<'REMOTE'
set -eux
cat > /etc/systemd/system/inference-shim.service <<EOF
[Unit]
Description=OpenAI->Bedrock inference shim
After=network-online.target
Wants=network-online.target
[Service]
User=ubuntu
WorkingDirectory=/opt/research-agent
EnvironmentFile=/opt/research-agent/.env
Environment=SHIM_PORT=11500
ExecStart=/opt/research-agent/venv/bin/python /opt/research-agent/inference_shim.py
Restart=always
RestartSec=5
[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now inference-shim.service
sleep 4
curl -sf http://127.0.0.1:11500/health && echo " SHIM_UP" || echo " SHIM_DOWN"
REMOTE

echo "== NemoClaw non-interactive install (OpenClaw + custom provider -> shim) =="
# Read the HAIKU_KEY from the demo env to pass as COMPATIBLE_API_KEY.
HAIKU_KEY=$(python3 -c "import sys;
[print(l.split('=',1)[1].strip()) for l in open('$DEMO_ENV') if l.startswith('HAIKU_KEY=')]")
$SSH "NEMOCLAW_REF='$NEMOCLAW_REF' AGENT_ID='$AGENT_ID' HAIKU_KEY='$HAIKU_KEY' bash -s" <<'REMOTE' > /tmp/nemoclaw-install-$AGENT_ID.log 2>&1 || echo "  (nemoclaw install returned nonzero; see /tmp/nemoclaw-install-$AGENT_ID.log)"
set -eux
export NEMOCLAW_NON_INTERACTIVE=1 NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE=1 \
  NEMOCLAW_AGENT=openclaw NEMOCLAW_PROVIDER=custom NEMOCLAW_PREFERRED_API=openai-completions \
  NEMOCLAW_ENDPOINT_URL='http://host.openshell.internal:11500/v1' \
  COMPATIBLE_API_KEY="$HAIKU_KEY" \
  NEMOCLAW_SANDBOX_NAME="research-$AGENT_ID" NEMOCLAW_POLICY_TIER=balanced \
  NEMOCLAW_WEB_SEARCH_PROVIDER=none \
  NEMOCLAW_TRUSTED_PRIVATE_HOSTS='host.openshell.internal:11500' \
  NEMOCLAW_INSTALL_REF="$NEMOCLAW_REF"
curl -fsSL "https://raw.githubusercontent.com/NVIDIA/NemoClaw/${NEMOCLAW_REF}/install.sh" | \
  NEMOCLAW_INSTALL_REF="$NEMOCLAW_REF" NEMOCLAW_NON_INTERACTIVE=1 NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE=1 \
  NEMOCLAW_AGENT=openclaw NEMOCLAW_PROVIDER=custom NEMOCLAW_PREFERRED_API=openai-completions \
  NEMOCLAW_ENDPOINT_URL='http://host.openshell.internal:11500/v1' COMPATIBLE_API_KEY="$HAIKU_KEY" \
  NEMOCLAW_SANDBOX_NAME="research-$AGENT_ID" NEMOCLAW_POLICY_TIER=balanced \
  NEMOCLAW_WEB_SEARCH_PROVIDER=none NEMOCLAW_TRUSTED_PRIVATE_HOSTS='host.openshell.internal:11500' \
  NEMOCLAW_INSTALL_REF="$NEMOCLAW_REF" bash
echo NEMOCLAW_INSTALL_DONE
REMOTE
echo "  nemoclaw install log saved: /tmp/nemoclaw-install-$AGENT_ID.log"

echo "== research agent loop systemd =="
$SSH "sudo bash -s '$AGENT_ID' '$AGENT_NAME' '$FOCUS' '$MODEL'" <<'REMOTE'
set -eux
AID="$1"; ANAME="$2"; FOCUS="$3"; MODEL="$4"
cat > /etc/systemd/system/research-agent.service <<EOF
[Unit]
Description=Auto-research agent loop ($AID)
After=network-online.target inference-shim.service
Wants=network-online.target
[Service]
User=ubuntu
WorkingDirectory=/opt/research-agent
EnvironmentFile=/opt/research-agent/.env
Environment=PYTHONUNBUFFERED=1
ExecStart=/opt/research-agent/venv/bin/python -m research_agent.agent_loop --agent-id "$AID" --display-name "$ANAME" --focus "$FOCUS" --model "$MODEL" --idle-sleep 180
Restart=always
RestartSec=10
[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now research-agent.service
sleep 3
systemctl is-active research-agent.service && echo " AGENT_ACTIVE" || (journalctl -u research-agent.service --no-pager -n 20; echo " AGENT_INACTIVE")
REMOTE

echo "== [$AGENT_ID @ $IP] provisioned =="
