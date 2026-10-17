"""Provider failover, exercised against a real local HTTP server speaking the OpenAI streaming format."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from app.llm import ChatLLM, LLMUnavailable, Provider


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        if self.path.startswith("/limited"):
            self.send_response(429)
            self.end_headers()
            self.wfile.write(b'{"error":"rate limited"}')
            return
        assert self.headers["Authorization"] == "Bearer good-key"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for word in ["He ", "used ", "Azure [1]"]:
            chunk = {"choices": [{"delta": {"content": word}}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


@pytest.fixture(scope="module")
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_streams_tokens(server):
    llm = ChatLLM([Provider("ok", f"{server}/ok", "good-key", "m")])
    assert llm.complete([{"role": "user", "content": "hi"}]) == "He used Azure [1]"


def test_fails_over_on_429(server):
    llm = ChatLLM([Provider("groq", f"{server}/limited", "k", "m"),
                   Provider("gemini", f"{server}/ok", "good-key", "m")])
    assert llm.complete([{"role": "user", "content": "hi"}]).endswith("[1]")
    assert llm.last_provider == "gemini"


def test_all_providers_down_raises(server):
    llm = ChatLLM([Provider("a", f"{server}/limited", "k", "m"),
                   Provider("b", "http://127.0.0.1:1", "k", "m")], timeout=2)
    with pytest.raises(LLMUnavailable):
        llm.complete([{"role": "user", "content": "hi"}])
