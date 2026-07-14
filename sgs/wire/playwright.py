"""PlaywrightOracle: drive xiaoce.fun GuessWord through a real Chromium.

This is the *online* wire for Round 3 — the one that actually talks to
xiaoce.fun. It is **mock-friendly**: we isolate every browser-touching
call behind a thin ``_script`` object so :mod:`unittest.mock` can swap it
out without booting Chromium.

Why Playwright (and not curl, not httpx)
----------------------------------------
The case study showed that ``requests``/``httpx`` get blocked by UA + TLS
fingerprint filters. Playwright rides a real Chromium, so the JS-driven
fetch and the cookie jar match a normal browser exactly. We pay for that
with a 200 ms per-probe startup cost, amortised by reuse.

Persistent context
------------------
``playwright.firefox_persistent_context`` is unavailable; Chromium is what
we have. ``playwright.chromium.launch_persistent_context`` gives us a
long-lived context stored under ``./.playwright-data/`` — the same trick
``playwright codegen`` and ``playwright open-url`` use. Cookies live in
the context, no jar-handling required.

Login once, run many
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # First run only: visit xiaoce.fun, log in via WeChat. The session
    # cookie persists in ``./.playwright-data/`` for future runs.
    python -c "
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            './.playwright-data', headless=False)
        page = ctx.new_page()
        page.goto('https://xiaoce.fun/daily/GuessWord')
        input('Log in, then press Enter to save cookies...')
        ctx.close()
    "

    # Subsequent runs: programmatic.
    python -m sgs.round3 --shareId 376634286041
"""

from __future__ import annotations

import json
import logging
import urllib.parse
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from sgs.oracle import Oracle, OracleResponse
from sgs.wire.base import DEFAULT_HEADERS, GUESS_PATH, parse_response

logger = logging.getLogger("sgs.wire.playwright")

__all__ = [
    "PlaywrightOracle",
    "PlaywrightScript",
    "PersistentContextManager",
    "JS_FETCH",
]


# JavaScript snippet executed inside the browser page (case-5, 376634286041
# phase-1/3 reference). Calls the GuessWord endpoint with `fun-device: web`,
# returns the response JSON as a string so Python-side :func:`json.loads`
# can parse it without DOM round-trips.
JS_FETCH = r"""
async (args) => {
  const { url, headers } = args;
  const resp = await fetch(url, {
    method: 'GET',
    headers,
    credentials: 'include',
    mode: 'cors',
  });
  const text = await resp.text();
  let body = null;
  try { body = JSON.parse(text); } catch (e) {
    return { __error: 'malformed_json', __raw: text.slice(0, 200), __status: resp.status };
  }
  return { __status: resp.status, __body: body };
}
"""


class PlaywrightScript(Protocol):
    """The narrow browser-touching surface we need from Playwright.

    Keeping this a :class:`Protocol` lets us:
    * unit-test the wire logic with a fake (no Chromium install);
    * swap to ``playwright.async_api`` later without breaking callers.
    """

    def evaluate(self, expression: str, arg: Any = None) -> Any: ...
    def close(self) -> None: ...


class PersistentContextManager(Protocol):
    """Factory for the persistent Chromium context."""

    def __call__(self, user_data_dir: Path, headless: bool = True) -> PlaywrightScript: ...


def _default_persistent_factory(
    user_data_dir: Path, headless: bool = True
) -> PlaywrightScript:
    """Boots a real persistent Chromium context. Lazy import keeps tests
    and offline CLI invocations from requiring ``playwright`` at all.
    """
    from playwright.sync_api import sync_playwright  # noqa: WPS433 — lazy

    pw = sync_playwright().start()
    context = pw.chromium.launch_persistent_context(
        user_data_dir=str(user_data_dir),
        headless=headless,
        user_agent=(
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
    )
    # Persistent context ships with at least one page; reuse it.
    page = context.pages[0] if context.pages else context.new_page()

    # Warm up: navigate so the JS domain is allowed to fetch (CORS).
    page.goto("https://xiaoce.fun/", wait_until="domcontentloaded")

    class _Ctx:
        def evaluate(self, expression: str, arg: Any = None) -> Any:
            return page.evaluate(expression, arg)

        def close(self) -> None:
            try:
                context.close()
            finally:
                pw.stop()

    return _Ctx()


class PlaywrightOracle:
    """:class:`Oracle` implementation backed by a real Chromium session.

    Parameters
    ----------
    share_id
        The xiaoce.fun GuessWord share identifier.
    script
        Anything implementing the :class:`PlaywrightScript` protocol.
        Defaults to a fresh persistent Chromium context under
        ``user_data_dir``.
    user_data_dir
        Where Playwright keeps cookies (default ``./.playwright-data``).
    base_url
        Override only when testing/staging.
    page_url
        Page URL — must be inside xiaoce.fun to satisfy CORS for the fetch.
    """

    def __init__(
        self,
        share_id: str,
        *,
        script: PlaywrightScript | None = None,
        user_data_dir: Path | None = None,
        page_url: str = "https://xiaoce.fun/daily/GuessWord",
        base_url: str = "https://xiaoce.fun",
    ) -> None:
        if not share_id or not share_id.strip():
            raise ValueError("share_id required")
        self.share_id = share_id
        self.base_url = base_url
        self.page_url = page_url
        self._user_data_dir = Path(user_data_dir or ".playwright-data")
        self._owns_script = script is None
        self._script: PlaywrightScript = (
            script if script is not None
            else _default_persistent_factory(self._user_data_dir, headless=True)
        )

    def probe(self, word: str) -> OracleResponse:
        if not word:
            raise ValueError("word must be a non-empty str")
        url = self._guess_url(word)
        headers = dict(DEFAULT_HEADERS)
        try:
            raw = self._script.evaluate(JS_FETCH, {"url": url, "headers": headers})
        except Exception as exc:  # noqa: BLE001 — wire boundary, log + flag
            logger.warning("browser fetch raised for %r: %s", word, exc)
            return OracleResponse(
                word=word, score=None, correct=False, rate_limited=False,
            )

        return self._normalise(raw, word)

    def close(self) -> None:
        if self._owns_script:
            try:
                self._script.close()
            except Exception as exc:  # noqa: BLE001 — close is best-effort
                logger.debug("script.close() failed: %s", exc)

    # ---- internals -------------------------------------------------------

    def _guess_url(self, word: str) -> str:
        # `urllib.parse.quote` handles CJK characters safely; pass
        # `safe=""` so / and other URL structural chars get encoded too.
        q = urllib.parse.quote(word, safe="")
        return (
            f"{self.base_url}{GUESS_PATH}"
            f"?word={q}&shareId={urllib.parse.quote(self.share_id, safe='')}"
            f"&skipBusinessErrorToast=true"
        )

    @staticmethod
    def _normalise(raw: Any, word: str) -> OracleResponse:
        """Translate a JS-side fetch result into an :class:`OracleResponse`.

        ``raw`` is either ``{__status, __body}`` (happy path), or
        ``{__error, __raw, __status}`` (browser threw before reaching the
        server), or just an arbitrary value the page handed back.
        """
        if not isinstance(raw, Mapping):
            # JS evaluator returned something exotic — treat as opaque fail.
            return OracleResponse(
                word=word, score=None, correct=False, rate_limited=False,
            )

        # Browser-side error before fetch resolved.
        err = raw.get("__error")
        if err == "malformed_json":
            logger.warning(
                "word=%r malformed JSON, status=%s, preview=%r",
                word, raw.get("__status"), raw.get("__raw"),
            )
            return OracleResponse(
                word=word, score=None, correct=False, rate_limited=False,
            )
        if err is not None:
            logger.warning("word=%r browser error: %s", word, err)
            return OracleResponse(
                word=word, score=None, correct=False, rate_limited=False,
            )

        # Happy path: tunnel through the shared parser.
        body = raw.get("__body")
        if body is None:
            return OracleResponse(
                word=word, score=None, correct=False, rate_limited=False,
            )
        # Re-shape JS dict into the same dict the HTTP wire would emit.
        envelope = {
            "code": body.get("code"),
            "msg": body.get("msg", ""),
            "data": body.get("data"),
        }
        return parse_response(envelope, word)


# Convenience for older callers (e.g. scripts) that want to inspect the JS.
def js_fetch_snippet() -> str:
    """Return the :data:`JS_FETCH` constant — for code generators / docs."""
    return JS_FETCH


def dumps_canonical_response(resp: OracleResponse) -> str:
    """Serialise :class:`OracleResponse` to a canonical JSON string."""
    return json.dumps(resp.to_ndjson(), ensure_ascii=False, sort_keys=True)