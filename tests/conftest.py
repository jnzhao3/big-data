import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from hbench.data import REPO_ROOT, Data

RESULTS = REPO_ROOT / "experiments" / "qwen_local" / "results"
FX = Path(__file__).parent / "fixtures"
NATIVE = json.loads((FX / "ollama_native_chat.json").read_text())
V1 = json.loads((FX / "ollama_openai_v1_chat.json").read_text())


@pytest.fixture(scope="session")
def data():
    """Benchmark data ($HBENCH_DATA or <repo>/datasets). Tests that need it are skipped when it is absent."""
    d = Data()
    if not d.exists():
        pytest.skip(f"benchmark data not found at {d.root}; set HBENCH_DATA (see datasets/README.md)")
    return d


@pytest.fixture(scope="session")
def results_dir() -> Path:
    return RESULTS


class MockOllama(BaseHTTPRequestHandler):
    """Answers /api/chat and /v1/chat/completions with real recorded Ollama replies; records what it was sent."""
    seen: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        MockOllama.seen.append((self.path, body))
        if self.path == "/api/chat":
            out = NATIVE["response"]
        elif self.path == "/v1/chat/completions":
            out = V1["response"]
        elif self.path == "/api/generate":
            out = {"done": True}
        else:
            out = {}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        data = json.dumps({"models": []}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    MockOllama.seen = []
    srv = HTTPServer(("127.0.0.1", 0), MockOllama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
