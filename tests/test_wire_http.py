"""Round 8 — visitor-curl oracle tests.

The legacy `sgs.wire.playwright.PlaywrightOracle` requires a
persistent Chromium session (`.playwright-data/`) — a heavy
dependency that the user may legitimately not have when they just
want to script the daily GuessWord challenge.

The `xiaoce.fun` GuessWord `/guessV1` endpoint accepts
**anonymous visitor traffic** (Round 8 measured: 200 OK, ~87 ms,
no cookies needed, only the trio of `fun-device: web` +
modern Chrome UA + `Referer: https://xiaoce.fun/`). So we expose a
second wire implementation that uses the `urllib` stdlib only —
no `requests`, no Playwright, no extra deps.

TDD contract for `sgs.wire.http.HttpOracle`:

  * `probe(word)` returns an :class:`OracleResponse`.
  * HTTP 200 + happy path → :class:`parse_response` result.
  * HTTP 200 + rate-limit (`code != 0`) → score=None, rate_limited=True.
  * HTTP 200 + lock (`data:null`, code=0) → score=None, rate_limited=False.
  * HTTP non-200 → raises :class:`HttpError` (caller should
    back off and retry).
  * `close()` is a no-op (no persistent sockets / contexts).
  * `__enter__` / `__exit__` lifecycle mirrors PlaywrightOracle.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import Any, Callable

import pytest

from sgs.oracle import OracleResponse


# ---------------------------------------------------------------------------
# Stub HTTP server — runs in a background thread, no external network
# ---------------------------------------------------------------------------


class _StubHandler(BaseHTTPRequestHandler):
    """Single-purpose handler driven by `_StubServer.state`."""

    # silence stderr noise during pytest
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler)
        # The handler reads response + status from class-level state.
        state = type(self).state  # type: ignore[attr-defined]
        body = json.dumps(state["body"]).encode("utf-8")
        self.send_response(state["status"])
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _StubServer:
    """Spins up an HTTP server bound to 127.0.0.1 with a programmable response."""

    def __init__(self, body: dict[str, Any], status: int = 200) -> None:
        # Install state on the handler class BEFORE serving requests.
        _StubHandler.state = {"body": body, "status": status}
        self._server = HTTPServer(("127.0.0.1", 0), _StubHandler)
        self.port = self._server.server_address[1]
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def shutdown(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def make_stub() -> Callable[..., _StubServer]:
    """Factory for stub servers; caller must shutdown() in teardown."""
    servers: list[_StubServer] = []

    def _factory(*args: Any, **kwargs: Any) -> _StubServer:
        s = _StubServer(*args, **kwargs)
        servers.append(s)
        return s

    yield _factory
    for s in servers:
        s.shutdown()


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_http_oracle_importable():
    from sgs.wire.http import HttpOracle

    assert HttpOracle is not None


def test_http_oracle_happy_path(make_stub: Callable[..., _StubServer]):
    """Real-like happy path: 200 + score/doubleScore/correct."""
    from sgs.wire.http import HttpOracle

    stub = make_stub(
        body={
            "code": 0,
            "msg": "ok",
            "data": {"score": 0.0, "doubleScore": 0.4776, "correct": False},
        },
        status=200,
    )
    oracle = HttpOracle(share_id="abc", base_url=stub.url)
    resp = oracle.probe("测试")
    assert isinstance(resp, OracleResponse)
    assert resp.word == "测试"
    assert resp.correct is False
    assert resp.rate_limited is False
    # doubleScore=0.4776 → double_score field is the *boolean toggle*
    # (true=use double-credit). The actual similarity is in `score`.
    assert resp.score == pytest.approx(0.0)


def test_http_oracle_uses_correct_query_params(make_stub: Callable[..., _StubServer]):
    """`probe()` must URL-encode `word` and append `shareId=` + `skipBusinessErrorToast=true`.

    We inspect the path captured by the stub to ensure the encoding is
    correct for Chinese characters.
    """
    from sgs.wire.http import HttpOracle

    captured: dict[str, str] = {}

    class CaptureHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_GET(self) -> None:  # noqa: N802
            captured["path"] = self.path
            body = json.dumps(
                {"code": 0, "data": {"score": 0.0, "doubleScore": 0.1, "correct": False}}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(("127.0.0.1", 0), CaptureHandler)
    port = server.server_address[1]
    t = Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        from sgs.wire.http import HttpOracle

        oracle = HttpOracle(
            share_id="12345", base_url=f"http://127.0.0.1:{port}"
        )
        oracle.probe("萧山")
    finally:
        server.shutdown()
        server.server_close()

    # word=萧山 URL-encoded, shareId=12345, skipBusinessErrorToast=true
    # 萧 = U+5C0F "small" (not U+8427 "desolate") — %E8%90%A7%E5%B1%B1
    assert "word=%E8%90%A7%E5%B1%B1" in captured["path"]
    assert "shareId=12345" in captured["path"]
    assert "skipBusinessErrorToast=true" in captured["path"]


def test_http_oracle_rate_limit_signal(make_stub: Callable[..., _StubServer]):
    """code=1 with 中文"频率" must surface as `rate_limited=True`."""
    from sgs.wire.http import HttpOracle

    stub = make_stub(
        body={
            "code": 1,
            "msg": "操作过于频繁，请稍后再试",
            "data": None,
        },
        status=200,
    )
    oracle = HttpOracle(share_id="abc", base_url=stub.url)
    resp = oracle.probe("任何")
    assert resp.score is None
    assert resp.rate_limited is True
    assert resp.correct is False


def test_http_oracle_server_lock_signal(make_stub: Callable[..., _StubServer]):
    """data:null + code=0 must NOT be rate-limited; it is server-side lock."""
    from sgs.wire.http import HttpOracle

    stub = make_stub(
        body={"code": 0, "msg": "已猜过", "data": None},
        status=200,
    )
    oracle = HttpOracle(share_id="abc", base_url=stub.url)
    resp = oracle.probe("任何")
    assert resp.score is None
    assert resp.rate_limited is False
    assert resp.correct is False


def test_http_oracle_http_error_raises():
    """HTTP non-200 (e.g. 500) must raise — caller decides back-off."""
    from sgs.wire.http import HttpError, HttpOracle

    class FailHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_GET(self) -> None:  # noqa: N802
            self.send_response(500)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"boom")

    server = HTTPServer(("127.0.0.1", 0), FailHandler)
    port = server.server_address[1]
    t = Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        oracle = HttpOracle(share_id="abc", base_url=f"http://127.0.0.1:{port}")
        with pytest.raises(HttpError) as exc:
            oracle.probe("测试")
        assert "500" in str(exc.value)
    finally:
        server.shutdown()
        server.server_close()


def test_http_oracle_context_manager_is_noop(make_stub: Callable[..., _StubServer]):
    """`with HttpOracle(...) as o: ...` is the recommended idiom."""
    from sgs.wire.http import HttpOracle

    stub = make_stub(
        body={"code": 0, "data": {"score": 0.0, "doubleScore": 0.1, "correct": False}},
        status=200,
    )
    with HttpOracle(share_id="x", base_url=stub.url) as oracle:
        assert isinstance(oracle, HttpOracle)
        oracle.probe("x")
    # No exception on exit; close() is a no-op.
    oracle.close()  # must not raise


def test_http_oracle_close_is_idempotent(make_stub: Callable[..., _StubServer]):
    """`close()` must be safe to call multiple times."""
    from sgs.wire.http import HttpOracle

    stub = make_stub(
        body={"code": 0, "data": {"score": 0.0, "doubleScore": 0.1, "correct": False}},
        status=200,
    )
    oracle = HttpOracle(share_id="x", base_url=stub.url)
    oracle.close()
    oracle.close()  # no-op twice


def test_http_oracle_send_correct_headers(make_stub: Callable[..., _StubServer]):
    """`fun-device: web` + `Referer` headers must reach the server."""
    captured_headers: dict[str, str] = {}

    class HeaderCaptureHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_GET(self) -> None:  # noqa: N802
            for k, v in self.headers.items():
                captured_headers[k.lower()] = v
            body = json.dumps(
                {"code": 0, "data": {"score": 0.0, "doubleScore": 0.1, "correct": False}}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(("127.0.0.1", 0), HeaderCaptureHandler)
    port = server.server_address[1]
    t = Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        from sgs.wire.http import HttpOracle

        oracle = HttpOracle(share_id="x", base_url=f"http://127.0.0.1:{port}")
        oracle.probe("any")
    finally:
        server.shutdown()
        server.server_close()

    assert captured_headers.get("fun-device") == "web"
    assert captured_headers.get("referer") == "https://xiaoce.fun/"


def test_solve_factory_accepts_http_name(make_stub: Callable[..., _StubServer]):
    """`build_oracle(name='http', ...)` must work end-to-end with the new HttpOracle."""
    from sgs.solve import build_oracle

    stub = make_stub(
        body={"code": 0, "data": {"score": 0.0, "doubleScore": 0.1, "correct": False}},
        status=200,
    )
    oracle = build_oracle("http", share_id="x", base_url=stub.url)
    # No need to actually probe — factory wiring is the unit under test.
    assert oracle.__class__.__name__ == "HttpOracle"
    oracle.close()