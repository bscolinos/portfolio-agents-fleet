#!/usr/bin/env bash
# Push a filled-in Aura Analyst endpoint+key to every research node and restart
# the agent loop so the fleet begins running NL analysis over SingleStore via the
# REAL Aura Portal domain. Run AFTER the domain is crawled + a key is issued in
# the Portal and ANALYST_API_URL/ANALYST_API_KEY are set in the demo .env.
#
# Usage: fleet/push_aura_key.sh
set -euo pipefail
export AWS_SHARED_CREDENTIALS_FILE=/Users/billscolinos/.aws/credentials_cf AWS_PROFILE=cf_gpu AWS_DEFAULT_REGION=us-east-1
PEM=/Users/billscolinos/Documents/code_factory/staging/research-fleet-key.pem
DEMO_ENV=/Users/billscolinos/Documents/code_factory/demos/portfolio-agents/.env

URL=$(python3 -c "[print(l.split('=',1)[1].strip()) for l in open('$DEMO_ENV') if l.startswith('ANALYST_API_URL=')]")
KEY=$(python3 -c "[print(l.split('=',1)[1].strip()) for l in open('$DEMO_ENV') if l.startswith('ANALYST_API_KEY=')]")
if [ -z "$URL" ] || [ -z "$KEY" ]; then
  echo "ANALYST_API_URL / ANALYST_API_KEY are empty in $DEMO_ENV — fill them first (see fleet/AURA_ANALYST_SETUP.md)." >&2
  exit 1
fi

# smoke-test the endpoint locally before touching the fleet
echo "== smoke-testing Aura endpoint =="
curl -sf -X POST "${URL%/chat}/query" -H "Authorization: Bearer ${KEY}" \
  -H "Content-Type: application/json" \
  -d '{"message":"How many research_experiments are recorded and what is the average sharpe?","output_modes":["sql","data"]}' \
  | head -c 600 || { echo "Aura smoke test FAILED — check URL/key/crawl."; exit 1; }
echo; echo "  smoke OK"

mapfile -t NODES < <(aws ec2 describe-instances \
  --filters Name=tag:fleet,Values=research-agents Name=instance-state-name,Values=running \
  --query 'Reservations[].Instances[].[Tags[?Key==`Name`].Value|[0],PublicIpAddress]' --output text)

for row in "${NODES[@]}"; do
  name=$(echo "$row" | awk '{print $1}'); ip=$(echo "$row" | awk '{print $2}')
  [ -z "$ip" ] && continue
  echo "== $name @ $ip =="
  ssh -o StrictHostKeyChecking=no -i "$PEM" ubuntu@"$ip" "sudo bash -c '
    sed -i \"/^ANALYST_API_URL=/d;/^ANALYST_API_KEY=/d\" /opt/research-agent/.env
    printf \"ANALYST_API_URL=%s\nANALYST_API_KEY=%s\n\" \"$URL\" \"$KEY\" >> /opt/research-agent/.env
    systemctl restart research-agent.service inference-shim.service
    sleep 2; systemctl is-active research-agent.service'" && echo "  updated + restarted"
done
echo "== Aura key pushed to fleet. Agents will now run the Aura analysis phase. =="
