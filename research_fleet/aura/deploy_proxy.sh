#!/usr/bin/env bash
# Deploy the Aura proxy service to the dedicated EC2 host.
# Usage: deploy_proxy.sh <public_ip>
set -euo pipefail
IP="$1"
AURA=/Users/billscolinos/Documents/code_factory/staging/portfolio-agents-prep/aura
PEM=/Users/billscolinos/Documents/code_factory/staging/research-fleet-key.pem
DEMO_ENV=/Users/billscolinos/Documents/code_factory/demos/portfolio-agents/.env
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -i $PEM ubuntu@$IP"
SCP="scp -o StrictHostKeyChecking=no -i $PEM"

echo "== wait for base bootstrap =="
for i in $(seq 1 40); do
  $SSH 'test -f /var/log/aura-proxy-bootstrap.done' 2>/dev/null && { echo "  base ready"; break; }
  sleep 15
done

echo "== build production .env (server-side secret) =="
# strong random proxy token generated locally
TOKEN=$(python3 -c "import secrets;print('cf-aura-'+secrets.token_urlsafe(24))")
KEY=$(cat "$AURA/.aura_key_verified")
python3 - "$DEMO_ENV" "$KEY" "$TOKEN" > /tmp/aura-proxy.env <<'PY'
import sys
demo, key, token = sys.argv[1], sys.argv[2], sys.argv[3]
keep = ('SINGLESTORE_HOST','SINGLESTORE_PORT','SINGLESTORE_USER','SINGLESTORE_PASSWORD','SINGLESTORE_DATABASE')
lines = [l for l in open(demo).read().splitlines() if l.startswith(keep)]
lines += [
  "ANALYST_CHAT_URL=https://apps.us-east-1.cloud.singlestore.com/v1/organizations/957c283e-5760-4e9a-b0e7-b077dac9c310/projects/5cc87edb-3e18-48f8-bef9-6097eb8fcab6/analyst/chat",
  f"ANALYST_API_KEY={key}",
  f"PROXY_TOKEN={token}",
  "PROXY_TIMEOUT_S=90","PROXY_MAX_RETRIES=2","PROXY_CACHE_TTL_S=900",
  "PROXY_RATE_PER_MIN=30","CB_FAIL_THRESHOLD=5","CB_RESET_S=30",
]
print("\n".join(lines))
PY
echo "  proxy token: $TOKEN"
echo "$TOKEN" > "$AURA/.proxy_token"; chmod 600 "$AURA/.proxy_token"

echo "== ship proxy + env =="
$SCP "$AURA/aura_proxy.py" ubuntu@$IP:/tmp/aura_proxy.py
$SCP /tmp/aura-proxy.env ubuntu@$IP:/tmp/aura-proxy.env
$SSH 'sudo bash -s' <<'REMOTE'
set -eux
mkdir -p /opt/aura-proxy
cp /tmp/aura_proxy.py /opt/aura-proxy/aura_proxy.py
cp /tmp/aura-proxy.env /opt/aura-proxy/.env
chmod 600 /opt/aura-proxy/.env
chown -R ubuntu:ubuntu /opt/aura-proxy
rm -f /tmp/aura-proxy.env
cat > /etc/systemd/system/aura-proxy.service <<EOF
[Unit]
Description=Aura Analyst hardened proxy
After=network-online.target
Wants=network-online.target
[Service]
User=ubuntu
WorkingDirectory=/opt/aura-proxy
EnvironmentFile=/opt/aura-proxy/.env
ExecStart=/opt/aura-proxy/venv/bin/uvicorn aura_proxy:app --host 0.0.0.0 --port 8799 --workers 2
Restart=always
RestartSec=5
[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now aura-proxy.service
sleep 5
curl -sf http://127.0.0.1:8799/health && echo " PROXY_UP" || (journalctl -u aura-proxy --no-pager -n 20; echo " PROXY_DOWN")
REMOTE
echo "== aura-proxy deployed to $IP =="
