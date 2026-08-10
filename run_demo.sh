#!/usr/bin/env bash
# Launch the portfolio-agents demo locally (backend + frontend).
# The agent fleet runs on the GPU box (see deploy_fleet.sh); this serves the UI
# over the data the fleet persisted to SingleStore.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV=/Users/billscolinos/Documents/code_factory/.venv

echo "== backend :8210 =="
source "$VENV/bin/activate"
cd "$HERE/backend"
pkill -f "uvicorn main:app --port 8210" 2>/dev/null || true
nohup uvicorn main:app --port 8210 --host 127.0.0.1 > /tmp/portfolio-agents-backend.log 2>&1 &
echo "backend pid $!  (log: /tmp/portfolio-agents-backend.log)"

echo "== frontend :3011 =="
cd "$HERE/frontend"
[ -d node_modules ] || npm install
pkill -f "next dev" 2>/dev/null || true
NEXT_PUBLIC_API_BASE=http://localhost:8210 nohup npm run dev -- --port 3011 > /tmp/portfolio-agents-frontend.log 2>&1 &
echo "frontend pid $!  (log: /tmp/portfolio-agents-frontend.log)"

echo
echo "Open http://localhost:3011"
echo "Stop: pkill -f 'uvicorn main:app --port 8210'; pkill -f 'next dev'"
