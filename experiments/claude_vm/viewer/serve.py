"""Local viewer for claude_vm sessions. Standard library only.

Usage: python -I experiments/claude_vm/viewer/serve.py [--port 8765] [--root <log root>]
Then open http://127.0.0.1:8765/

Binds to 127.0.0.1. Serves the page and read-only JSON endpoints; it never serves files by path:
    GET /api/sessions                the session index
    GET /api/session/<session_id>    the merged timeline (rebuilt from the logs on each request)
    GET /api/compare[?study=S&e3=1]  one row per scored run: scores by category, calibration, tool use
    GET /api/diff?a=<id>&b=<id>      item-level differences between two scored sessions
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from urllib.parse import parse_qs, urlsplit  # noqa: E402

from cvm import comparison, timeline  # noqa: E402
from cvm.paths import SESSION_ID_RE, SessionPaths, log_root, read_index  # noqa: E402

PAGE = HERE / "index.html"


def make_handler(root: Path):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj, ensure_ascii=False, default=str).encode(), "application/json")

        def do_GET(self):  # noqa: N802
            parts = urlsplit(self.path)
            path, qs = parts.path, {k: v[-1] for k, v in parse_qs(parts.query).items()}
            if path in ("/", "/index.html"):
                return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            if path == "/api/sessions":
                idx = read_index(root)
                rows = sorted(idx.values(), key=lambda e: e.get("start_time") or "", reverse=True)
                return self._json({"root": str(root), "sessions": rows})
            if path == "/api/compare":
                study = qs.get("study") or None
                return self._json({"rows": comparison.run_rows(root, study, include_e3=qs.get("e3") == "1")})
            if path == "/api/diff":
                a, b = qs.get("a", ""), qs.get("b", "")
                idx = read_index(root)
                for sid in (a, b):
                    if not SESSION_ID_RE.match(sid) or sid not in idx:
                        return self._json({"error": f"unknown session {sid!r}"}, 400)
                    if not SessionPaths(sid, root).items.exists():
                        return self._json({"error": f"session {sid[:8]} has not been scored"}, 400)
                return self._json(comparison.item_diff(root, a, b))
            if path.startswith("/api/session/"):
                sid = path.rsplit("/", 1)[-1]
                if not SESSION_ID_RE.match(sid):
                    return self._json({"error": "bad session id"}, 400)
                entry = read_index(root).get(sid)
                if entry is None:
                    return self._json({"error": "unknown session"}, 404)
                return self._json(timeline.build(SessionPaths(sid, root), entry))
            return self._json({"error": "not found"}, 404)

        def log_message(self, fmt, *args):
            sys.stderr.write("viewer: " + fmt % args + "\n")

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--root", default=None, help="log root (default ~/hbench_raw/claude_vm)")
    a = ap.parse_args()
    root = Path(a.root).expanduser() if a.root else log_root()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(root))
    print(f"claude_vm viewer on http://127.0.0.1:{a.port}/  (log root {root})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
