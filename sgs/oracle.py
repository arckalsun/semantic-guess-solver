"""Oracle: black-box API contract for the xiaoce.fun GuessWord endpoint.

This module defines the *interface* the solver talks to, not the wire
implementation. The wire lives in ``probe.py`` (Round 2); the policy that
*decides* what to probe lives in ``round1.py`` (Round 1). Keeping them
separate means we can unit-test the policy against a fake oracle that
returns scripted responses — no network required.

Wire contract (canonical reference):
    GET https://xiaoce.fun/api/v0/quiz/daily/GuessWord/guessV1
        ?word=<candidate>&shareId=<id>&skipBusinessErrorToast=true
    Headers:
        fun-device: web
    Response shape:
        { "code": 0,
          "data": { "score": 0.989,
                    "doubleScore": true,
                    "correct": true,
                    "rateLimited": false } }

Three orthogonal signals drive the algorithm:
    * ``data.score``       — continuous similarity 0..1
    * ``data.doubleScore`` — bonus flag (may be True even when score<1)
    * ``data.correct``     — bool, the authoritative stop condition
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = [
    "Oracle",
    "OracleResponse",
    "FakeOracle",
]  # NB: `already_correct` was removed — callers import it from ``probe``.


@dataclass(frozen=True)
class OracleResponse:
    """One oracle (GuessWord server) probe result.

    ``score`` is the only field used by the ranker; ``correct`` is the
    authoritative stop condition; ``double_score`` and ``rate_limited``
    are surfaced as warnings so the operator can see them in the NDJSON.

    ``score`` is ``Optional`` because three legitimate outcomes have no
    numeric score: server-side word lock (data:null), business error
    (code != 0), and "rate-limited" (code == 1). Callers must treat
    ``score is None`` as "skip this word in the ranker".
    """

    word: str
    score: float | None
    correct: bool = False
    double_score: bool = False
    rate_limited: bool = False

    def to_ndjson(self) -> dict:
        """Serialise to the same shape as the canonical wire envelope."""
        return {
            "word": self.word,
            "score": self.score,
            "correct": self.correct,
            "doubleScore": self.double_score,
            "rateLimited": self.rate_limited,
        }


@runtime_checkable
class Oracle(Protocol):
    """Anything the solver can probe.

    Implementations:
        * :class:`FakeOracle` — script-driven, for tests.
        * :class:`HttpOracle` — real fetch via Playwright/browser fetch
          (Round 2 wire; lives in ``probe.py``).
    """

    def probe(self, word: str) -> OracleResponse:
        """Probe one word. Must NOT raise on a rate-limited / captcha /
        network glitch — return an ``OracleResponse(rate_limited=True)``
        instead so the caller can decide whether to back off."""
        ...

    def close(self) -> None:
        """Release any persistent resources (browser context, etc.)."""
        ...


class FakeOracle:
    """An oracle driven by a script of canned responses, plus a learned
    linear model for everything else.

    Used in tests to validate the ranker's reaction to a known oracle
    without a network. Construct one with a ``{word: OracleResponse}``
    map; unrecognised words fall back to ``default_score`` and
    ``default_correct=False``.
    """

    def __init__(
        self,
        script: dict[str, OracleResponse] | None = None,
        default_score: float = 0.0,
    ) -> None:
        self._script: dict[str, OracleResponse] = dict(script or {})
        self._default_score = default_score
        self.calls: list[str] = []  # call log for assertions

    def probe(self, word: str) -> OracleResponse:
        self.calls.append(word)
        if word in self._script:
            return self._script[word]
        return OracleResponse(word=word, score=self._default_score)

    def close(self) -> None:  # noqa: D401 — Protocol contract
        return None

    # ---- test helpers --------------------------------------------------

    def learn(self, word: str, score: float, *, correct: bool = False,
              double_score: bool = False) -> OracleResponse:
        """Inject a fresh response into the script and return it."""
        resp = OracleResponse(
            word=word, score=score, correct=correct,
            double_score=double_score,
        )
        self._script[word] = resp
        return resp

    @property
    def call_count(self) -> int:
        return len(self.calls)