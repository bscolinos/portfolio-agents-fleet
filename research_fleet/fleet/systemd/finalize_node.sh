#!/usr/bin/env bash
# Finalize a fleet node onto Switchyard routing. Run ON the node.
set -e

# 1) switchyard systemd service (binary at ~/.local/bin)
sudo cp /tmp/switchyard-local.service /etc/systemd/system/switchyard.service
sudo systemctl daemon-reload
sudo systemctl enable --now switchyard.service
sleep 5
echo -n "switchyard: "; systemctl is-active switchyard.service
curl -s --max-time 5 http://127.0.0.1:4000/health; echo " <-sy health"

# 2) restart the (already-upgraded) shim so tool-use/structured-output is live
sudo systemctl restart inference-shim.service
sleep 3
echo -n "shim: "; systemctl is-active inference-shim.service

# 3) flip research-agent to switchyard transport (idempotent)
sudo python3 - <<'PYEOF'
import re
p="/etc/systemd/system/research-agent.service"
s=open(p).read()
if "--transport switchyard" not in s:
    s=re.sub(r"(ExecStart=.*agentic_loop)", r"\1 --transport switchyard", s)
if "SWITCHYARD_URL" not in s:
    s=s.replace("[Service]", "[Service]\nEnvironment=SWITCHYARD_URL=http://127.0.0.1:4000/v1/chat/completions\nEnvironment=AGENT_TRANSPORT=switchyard")
s=re.sub(r"^After=.*$", "After=network-online.target inference-shim.service switchyard.service", s, flags=re.M)
open(p,"w").write(s)
PYEOF
sudo systemctl daemon-reload
sudo systemctl restart research-agent.service
sleep 6
echo -n "research-agent: "; systemctl is-active research-agent.service
journalctl -u research-agent.service --no-pager -n 3 2>&1 | grep -iE "registered|cycle" | tail -1
