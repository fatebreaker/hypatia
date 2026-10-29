"""GET-only proxy in front of the MedAgentBench FHIR server (started by scripts/medagentbench_server.sh).

Why: in the official MedAgentBench harness a POST is never sent to the server -- it is parsed, answered with "POST
request accepted and executed successfully", and graded later from the conversation. Our agents write Python and could
call requests.post() directly, which WOULD write into the one EHR every concurrent episode reads. So agents are given
this proxy's address: GET/HEAD are forwarded unchanged (with the client's Host header, so HAPI's bundle links point
back here), every other method gets 405 and a message pointing at fhir_post() in data_files/medagentbench_fhir.py.

  venv/bin/python scripts/medagentbench_proxy.py --port 18080 --upstream http://127.0.0.1:18081
"""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
       "transfer-encoding", "upgrade", "content-length", "content-encoding"}
REFUSAL = {
    "resourceType": "OperationOutcome",
    "issue": [{"severity": "error", "code": "forbidden", "diagnostics":
               "This evaluation's EHR is read-only. Record a POST with fhir_post(url, payload) from "
               "data_files/medagentbench_fhir.py -- as in the official MedAgentBench harness, POSTs are accepted and "
               "graded, never executed."}],
}


def make_handler(upstream: str):
    class Proxy(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):     # quiet: one line per request would swamp the log
            pass

        def _forward(self, head_only: bool):
            req = urllib.request.Request(upstream + self.path, method="HEAD" if head_only else "GET")
            for k, v in self.headers.items():
                if k.lower() not in HOP and k.lower() != "accept-encoding":   # identity bodies only
                    req.add_header(k, v)
            try:
                resp = urllib.request.urlopen(req, timeout=300)
                status, headers, body = resp.status, resp.headers, resp.read()
            except urllib.error.HTTPError as e:
                status, headers, body = e.code, e.headers, e.read()
            except Exception as e:                               # noqa: BLE001
                status, headers, body = 502, {}, json.dumps({"error": f"upstream: {e}"}).encode()
            self.send_response(status)
            for k, v in (headers.items() if hasattr(headers, "items") else []):
                if k.lower() not in HOP:
                    self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if not head_only:
                self.wfile.write(body)

        def do_GET(self):
            self._forward(False)

        def do_HEAD(self):
            self._forward(True)

        def _refuse(self):
            n = int(self.headers.get("Content-Length") or 0)
            if n:
                self.rfile.read(n)
            body = json.dumps(REFUSAL).encode()
            self.send_response(405)
            self.send_header("Content-Type", "application/fhir+json")
            self.send_header("Allow", "GET, HEAD")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_POST = do_PUT = do_PATCH = do_DELETE = _refuse

    return Proxy


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--upstream", required=True, help="e.g. http://127.0.0.1:18081 (no trailing slash)")
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.bind, a.port), make_handler(a.upstream.rstrip("/")))
    srv.daemon_threads = True
    srv.serve_forever()
