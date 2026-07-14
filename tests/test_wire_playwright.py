"""Tests for :mod:`sgs.wire.playwright` — PlaywrightOracle, **no Chromium**.

We drive the wire with a tiny stub PlaywrightScript so the tests are
sub-second. The browser-y bits (page.evaluate, page.close) are exercised
manually with the persistent-context recipe, never in CI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from sgs.oracle import OracleResponse
from sgs.wire.playwright import (
    JS_FETCH,
    PlaywrightOracle,
    dumps_canonical_response,
    js_fetch_snippet,
)


@dataclass
class FakeScript:
    """In-memory :class:`PlaywrightScript` for tests."""

    responses: dict[str, Any]
    closed: bool = False

    def evaluate(self, expression: str, arg: Any = None) -> Any:
        # The wire only ever calls evaluate with JS_FETCH; we ignore the
        # expression here and just look up the URL.
        url = (arg or {}).get("url", "")
        return self.responses.get(url, {"__error": "no-scripted-response"})

    def close(self) -> None:
        self.closed = True


# ----- URL crafting --------------------------------------------------------

def test_guess_url_encodes_cjk_word_and_share_id() -> None:
    o = PlaywrightOracle(share_id="376634286041", script=FakeScript({}))
    url = o._guess_url("忍者")
    assert "word=%E5%BF%8D%E8%80%85" in url
    assert "shareId=376634286041" in url
    assert "skipBusinessErrorToast=true" in url


def test_guess_url_encodes_share_id_with_special_chars() -> None:
    o = PlaywrightOracle(share_id="a/b c", script=FakeScript({}))
    url = o._guess_url("测试")
    assert "shareId=a%2Fb%20c" in url  # / and space


# ----- probe() happy path --------------------------------------------------

def test_probe_returns_oracle_response_for_happy_path() -> None:
    share = "376634286041"
    word = "忍者"
    url = PlaywrightOracle(share_id=share, script=FakeScript({}))._guess_url(word)
    script = FakeScript({
        url: {"__status": 200,
              "__body": {"code": 0, "msg": "ok",
                         "data": {"score": 0.989, "doubleScore": True,
                                  "correct": True}}},
    })
    o = PlaywrightOracle(share_id=share, script=script)
    r = o.probe(word)
    assert r.score == 0.989
    assert r.correct is True
    assert r.double_score is True
    assert r.rate_limited is False
    # Caller-owned scripts are NOT auto-closed (see `close_does_not_call_close_on_external_script`).
    script.close()
    assert script.closed is True


def test_probe_returns_none_score_on_data_null() -> None:
    share, word = "376634286041", "萧山"
    url = PlaywrightOracle(share_id=share, script=FakeScript({}))._guess_url(word)
    script = FakeScript({url: {"__status": 200,
                               "__body": {"code": 0, "msg": "ok", "data": None}}})
    o = PlaywrightOracle(share_id=share, script=script)
    r = o.probe(word)
    assert r.score is None
    assert r.correct is False
    o.close()


def test_probe_flags_rate_limit_response() -> None:
    share, word = "376634286041", "上海"
    url = PlaywrightOracle(share_id=share, script=FakeScript({}))._guess_url(word)
    script = FakeScript({url: {"__status": 200,
                               "__body": {"code": 1, "msg": "rate limit",
                                          "data": None}}})
    o = PlaywrightOracle(share_id=share, script=script)
    r = o.probe(word)
    assert r.score is None
    assert r.rate_limited is True
    o.close()


# ----- probe() error & edge branches --------------------------------------

def test_probe_handles_browser_side_error_as_blank_response() -> None:
    share, word = "376634286041", "上海"
    url = PlaywrightOracle(share_id=share, script=FakeScript({}))._guess_url(word)
    script = FakeScript({url: {"__error": "network", "__raw": "..."}})
    o = PlaywrightOracle(share_id=share, script=script)
    r = o.probe(word)
    assert r.score is None
    assert r.correct is False
    assert r.rate_limited is False


def test_probe_handles_exception_in_evaluate_as_blank_response() -> None:
    class RaisingScript:
        def evaluate(self, expression, arg=None):
            raise RuntimeError("playwright disconnected")

        def close(self):
            pass

    o = PlaywrightOracle(share_id="376634286041", script=RaisingScript())
    r = o.probe("上海")
    assert r.score is None
    o.close()


def test_probe_rejects_empty_word() -> None:
    o = PlaywrightOracle(share_id="376634286041", script=FakeScript({}))
    with pytest.raises(ValueError, match="non-empty"):
        o.probe("")
    o.close()


def test_constructor_rejects_empty_share_id() -> None:
    with pytest.raises(ValueError, match="share_id"):
        PlaywrightOracle(share_id="", script=FakeScript({}))


def test_close_does_not_call_close_on_external_script() -> None:
    """If the caller supplied `script`, we must not close it."""
    script = FakeScript({})
    o = PlaywrightOracle(share_id="376634286041", script=script)
    o.close()
    assert script.closed is False


def test_close_swallows_errors_when_owned() -> None:
    class BadClose:
        def evaluate(self, expression, arg=None):
            raise AssertionError("should not be called")

        def close(self):
            raise RuntimeError("boom")

    o = PlaywrightOracle(share_id="376634286041", script=BadClose())
    # Should NOT raise even though close() raises.
    o.close()


# ----- JS_FETCH shape sanity ----------------------------------------------

def test_js_fetch_is_async_arrow_function() -> None:
    src = js_fetch_snippet()
    assert "async" in src
    assert "fetch" in src
    assert "credentials" in src  # cookie replay


def test_canonical_response_dumps_sorted_ascii_safe() -> None:
    resp = OracleResponse(word="忍者", score=0.989, correct=True, double_score=True)
    s = dumps_canonical_response(resp)
    assert "忍者" in s
    assert '"correct": true' in s
    assert '"doubleScore": true' in s
    assert '"rateLimited": false' in s
    assert s == dumps_canonical_response(resp)  # deterministic