"""HTTP tool server exposing the templated SingleStore write tool to the
OpenClaw-in-OpenShell sandbox agent over the trusted host bridge.

The sandbox reaches this at `http://host.openshell.internal:11510` (add
`host.openshell.internal` to the trusted-private-hosts; the same managed bridge
used for the inference shim). Because every write is validated + templated
host-side, the model literally cannot persist a nonconforming row — data stays
uniform across all 5 nodes.

Endpoints:
  GET  /health                      -> {ok}
  GET  /tools                       -> the machine-readable TOOL_SCHEMA (self-describing)
  POST /tool/<name>   body=JSON     -> {ok, id, table, ...}  (the write receipt)
  POST /tool          body={tool, payload}   (alt dispatch form)

Runs as a systemd service on each node next to inference-shim.
"""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# make the research_agent package importable when run as a plain script on the box
sys.path.insert(0, os.environ.get("AGENT_HOME", "/opt/research-agent"))

from research_agent import write_tool as wt  # noqa: E402

PORT = int(os.environ.get("TOOL_PORT", "11510"))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, obj: dict):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass

    def do_GET(self):
        p = self.path.rstrip("/")
        if p in ("", "/health", "/healthz"):
            self._send(200, {"ok": True, "tools": sorted(wt.TOOLS)})
        elif p in ("/tools", "/schema"):
            self._send(200, wt.TOOL_SCHEMA)
        else:
            self._send(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            self._send(400, {"ok": False, "error": f"bad json: {e}"}); return

        parts = self.path.strip("/").split("/")
        if parts[0] != "tool":
            self._send(404, {"ok": False, "error": "use POST /tool/<name>"}); return

        if len(parts) >= 2 and parts[1]:
            name, payload = parts[1], body
        else:  # POST /tool  {tool, payload}
            name, payload = body.get("tool"), body.get("payload", {})

        try:
            receipt = wt.call_tool(name, payload)
            self._send(200, receipt)
        except wt.ToolError as e:
            self._send(422, {"ok": False, "error": str(e)})
        except Exception as e:  # DB/embedding errors
            self._send(502, {"ok": False, "error": f"write failed: {e}"})


def main():
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"research write-tool server on 0.0.0.0:{PORT} (tools: {sorted(wt.TOOLS)})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
