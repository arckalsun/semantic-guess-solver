"""sgs.wire.http — visitor-curl oracle (Round 8).

This is the lightweight wire for environments that don't have a
persistent Chromium context. It uses only the `urllib` stdlib — no
`requests`, no `playwright`, no extra deps.

Round 8 measurement (2026-07-14): the `xiaoce.fun` GuessWord
`/guessV1` endpoint accepts anonymous visitor traffic when the
request carries the trio:

  * `fun-device: web`
  * a modern Chrome User-Agent (126+)
  * `Referer: https://xiaoce.fun/`

Response is JSON in the same shape :func:`sgs.wire.base.parse_response`
already understands. We delegate parsing there and never duplicate
business-logic checks (rate-limit heuristic, lock signal, etc.).

Compared with :mod:`sgs.wire.playwright`:

  * No Chromium binary, no `.playwright-data/` directory, no
    human-assisted login — zero setup.
  * No JS execution — we cannot bypass a future bot-check that
    runs in the browser context. If xiaoce.fun tightens to that,
    fall back to PlaywrightOracle.
  * No persistent socket — every ``probe()`` opens a fresh
    connection. For a daily cron at 8am that's fine.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from sgs.oracle import OracleResponse
from sgs.wire.base import (
    DEFAULT_HEADERS,
    GUESS_PATH,
    WireEndpoint,
    parse_response,
)


class HttpError(RuntimeError):
    """Raised when the underlying HTTP request fails.

    Distinct from rate-limit / lock signals — those come back as
    HTTP 200 with a special body. :class:`HttpError` is for
    connection failures and non-200 status codes only.
    """


@dataclass
class HttpOracle:
    """Visitor-curl oracle.

    Each :meth:`probe` call opens a fresh :class:`urllib.request.Request`,
    sends the three required headers, parses the JSON body via
    :func:`sgs.wire.base.parse_response`, and returns the result.
    """

    share_id: str
    base_url: str = "https://xiaoce.fun"
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    )
    timeout_s: float = 8.0

    def __post_init__(self) -> None:
        # Reuse WireEndpoint for URL construction so curl and
        # Playwright can never drift on query-param order.
        self._endpoint = WireEndpoint(share_id=self.share_id, base_url=self.base_url)
        # NOTE: the Referer is hard-coded to the canonical xiaoce.fun
        # origin. Tests override `base_url` only to redirect the GET
        # to a stub server; they do NOT want to lie about the
        # origin to the server under test (the production
        # endpoint's bot-check uses Referer for fingerprinting).
        self._headers = {
            **DEFAULT_HEADERS,
            "User-Agent": self.user_agent,
            "Referer": "https://xiaoce.fun/",
        }

    # ---- context-manager protocol (mirrors PlaywrightOracle) ----

    def __enter__(self) -> "HttpOracle":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """No persistent resources to release. Idempotent."""

    # ---- Oracle protocol ----

    def probe(self, word: str) -> OracleResponse:
        """Fetch a single guess and translate to :class:`OracleResponse`."""
        url = self._guess_url(word)
        req = urllib.request.Request(url, headers=self._headers, method="GET")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                status = resp.status
                raw_bytes = resp.read()
        except urllib.error.HTTPError as e:
            raise HttpError(f"HTTP {e.code} for word={word!r}: {e.reason}") from e
        except urllib.error.URLError as e:
            raise HttpError(f"connection error for word={word!r}: {e.reason}") from e

        if status != 200:
            raise HttpError(f"HTTP {status} for word={word!r}")

        try:
            raw = json.loads(raw_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise HttpError(f"malformed JSON for word={word!r}: {e}") from e

        return parse_response(raw, word)

    # ---- internals ----

    def _guess_url(self, word: str) -> str:
        """Compose the full GET URL with word URL-encoded.

        Mirrors :meth:`sgs.wire.playwright.PlaywrightOracle._guess_url`
        (intentionally duplicated so the two wires can evolve
        independently).
        """
        return (
            f"{self.base_url}{GUESS_PATH}"
            f"?word={urllib.parse.quote(word, safe='')}"
            f"&shareId={urllib.parse.quote(self.share_id, safe='')}"
            f"&skipBusinessErrorToast=true"
        )


__all__ = ["HttpError", "HttpOracle"]