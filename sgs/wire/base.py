"""wire.base: response parsing for the xiaoce.fun GuessWord API.

The wire is a thin wrapper around :class:`sgs.oracle.Oracle` — every
implementation (:mod:`sgs.wire.playwright`, future :mod:`sgs.wire.curl`,
etc.) normalises its raw response into :class:`OracleResponse` here.

Canonical request
-----------------
``GET /api/v0/quiz/daily/GuessWord/guessV1?word=<chinese>&shareId=<id>&skipBusinessErrorToast=true``

Required header: ``fun-device: web``.

Canonical response (case-5, shareId 376634286041)
-------------------------------------------------
.. code-block:: json
    {
        "code": 0,
        "msg": "ok",
        "data": {
            "score": 0.741,
            "doubleScore": false,
            "correct": false
        }
    }

Special cases
-------------
* ``data: null`` — server has **already locked** the word (you've guessed
  it before, or someone else on the same shareId has). The case study
  showed this is benign: treat it as ``OracleResponse(score=None,
  rateLimited=False, correct=None)`` and skip. Use
  :func:`sgs.oracle.already_correct` to build a skip-list upstream.
* ``data.score`` outside ``[0, 1]`` — caller-side bug; we raise.
* ``code != 0`` — business error. We pass the message through to
  ``OracleResponse.error`` so the probe loop can decide (e.g. retry, skip).
* HTTP 429 / ``code = 2xx-with-rate-limit-flag`` — record as
  ``rateLimited=True``; caller applies backoff.

Every branch ends up in :class:`OracleResponse`; nothing leaks.
"""

from __future__ import annotations

import logging
import math
import urllib.parse
from dataclasses import dataclass
from typing import Any, Mapping

from sgs.oracle import OracleResponse

logger = logging.getLogger("sgs.wire.base")

__all__ = [
    "WireEndpoint",
    "parse_response",
    "RateLimitSignal",
    "DEFAULT_HEADERS",
    "GUESS_PATH",
]


GUESS_PATH = "/api/v0/quiz/daily/GuessWord/guessV1"

DEFAULT_HEADERS = {
    # Without `fun-device: web` the API returns 403 (case-5).
    "fun-device": "web",
    # The browser fetch in the case study used `accept: application/json,
    # text/plain, */*`; we mirror it for fidelity.
    "accept": "application/json, text/plain, */*",
}


@dataclass(frozen=True)
class WireEndpoint:
    """The two knobs every wire implementation needs."""

    share_id: str
    base_url: str = "https://xiaoce.fun"

    def guess_url(self, word: str) -> str:
        """Return the full URL to call, with ``word`` URL-encoded.

        Mirrors :meth:`sgs.wire.playwright.PlaywrightOracle._guess_url` —
        the only difference is the template is a method call here so we
        don't need a placeholder format string.
        """
        return (
            f"{self.base_url}{GUESS_PATH}"
            f"?word={urllib.parse.quote(word, safe='')}"
            f"&shareId={urllib.parse.quote(self.share_id, safe='')}"
            f"&skipBusinessErrorToast=true"
        )


@dataclass(frozen=True)
class RateLimitSignal:
    """A 429 / `code=429-style` signal."""

    retry_after_s: float | None  # if server told us; None means "back off yourself"


def parse_response(raw: Mapping[str, Any], word: str) -> OracleResponse:
    """Convert the raw API JSON into :class:`OracleResponse`.

    Never raises for ordinary business outcomes (rate-limit, lock,
    wrong-shareId) — those become flagged :class:`OracleResponse` objects.
    Raises only for caller-side bugs (malformed JSON, score out of range).

    Outcomes
    --------
    * **data: null** — server-side lock. Returned as
      ``OracleResponse(score=None, correct=False, rate_limited=False)``.
      ``score is None`` is the signal to skip this word in the ranker.
    * **code != 0** — business error. Returned as
      ``OracleResponse(score=None, correct=False, rate_limited=<heuristic>)``.
    * **happy path** — score parsed from ``data.score``, correct from
      ``data.correct``, doubleScore from ``data.doubleScore``.
    """
    if not isinstance(raw, Mapping):
        raise ValueError(f"response must be a dict, got {type(raw).__name__}")

    code = raw.get("code")
    msg = raw.get("msg", "")
    data = raw.get("data")

    # --- data is missing entirely (the dict has no `data` key) ---------
    # Be loud: this shouldn't happen in production.
    if "data" not in raw:
        raise ValueError("response missing required 'data' field")

    # --- business error (code != 0) -------------------------------------
    # Note: we check business-error BEFORE the data-is-null shortcut,
    # because a rate-limited response looks like {code: 1, data: null}.
    if code not in (0, None, "0"):
        rate_limited = _looks_like_rate_limit(code, msg)
        if rate_limited:
            logger.debug("word=%r rate-limited (code=%s, msg=%r)", word, code, msg)
        else:
            logger.warning("word=%r business error (code=%s, msg=%r)", word, code, msg)
        return OracleResponse(
            word=word, score=None, correct=False,
            double_score=False, rate_limited=rate_limited,
        )

    # --- data: null with code == 0 — server-side lock -------------------
    # (case-5: guessing the same word again returns data:null)
    if data is None:
        logger.info("word=%r locked by server (data:null, msg=%r)", word, msg)
        return OracleResponse(
            word=word, score=None, correct=False,
            double_score=False, rate_limited=False,
        )

    if not isinstance(data, Mapping):
        raise ValueError(f"data must be a dict or null, got {type(data).__name__}")

    # --- happy path ----------------------------------------------------
    score_raw = data.get("score")
    double = bool(data.get("doubleScore", False))
    correct = bool(data.get("correct", False))

    if score_raw is None:
        # data present but no score — likely a new error shape we haven't
        # seen yet; treat as benign miss but log loudly.
        logger.warning("word=%r missing score field (msg=%r)", word, msg)
        return OracleResponse(
            word=word, score=None, correct=correct,
            double_score=double, rate_limited=False,
        )

    score = float(score_raw)
    if math.isnan(score) or not (0.0 <= score <= 1.0):
        raise ValueError(f"score out of [0,1]: {score!r} (word={word!r})")

    return OracleResponse(
        word=word, score=score, correct=correct,
        double_score=double, rate_limited=False,
    )


def _looks_like_rate_limit(code: Any, msg: str) -> bool:
    """Heuristic: ``code == 1`` or msg mentions 限频 / rate / limit."""
    if code == 1:
        return True
    msg_low = (msg or "").lower()
    return any(tok in msg_low for tok in ("limit", "限频", "频率", "too many"))