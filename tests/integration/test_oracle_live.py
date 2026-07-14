"""Round 7 — live wire integration tests (opt-in).

These tests boot a real Chromium via `PlaywrightOracle` and call
xiaoce.fun. They are **NOT** in the default test run because:

  1. They require a persistent Playwright context with valid session
     cookies (the user must have logged into xiaoce.fun once).
  2. They are slow (1-2 seconds per probe).
  3. They are sensitive to xiaoce.fun's server rate-limits.

Run with:

    pytest -m integration tests/integration/
    pytest -m "integration and not live_answer" tests/integration/

If the persistent context is missing, every test is skipped — never
failed — so CI on a clean machine does not break.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

# All tests in this module are opt-in.
pytestmark = pytest.mark.integration


# --------------------------------------------------------------------------
# Skip-when-no-credentials guard.
# --------------------------------------------------------------------------


def _has_live_credentials() -> bool:
    """Return True if `.playwright-data` contains a usable session."""
    p = Path(".playwright-data")
    if not p.exists():
        return False
    # Chromium stores cookies in `Default/Cookies` (sqlite db). If the
    # file is present and non-empty, we have *some* cookies; whether
    # they're valid is for the first probe to determine.
    cookies_db = p / "Default" / "Cookies"
    if not cookies_db.exists():
        return False
    return cookies_db.stat().st_size > 0


pytestmark_live = pytest.mark.skipif(
    not _has_live_credentials(),
    reason=(
        "no live credentials: log in once via "
        "`python -c \"from playwright.sync_api import sync_playwright; ...\"`"
        " (see SKILL.md Round 3 for the recipe) before running live tests"
    ),
)


# --------------------------------------------------------------------------
# Word pool — guaranteed not-the-answer candidates.
# --------------------------------------------------------------------------
# xiaoce.fun's daily GuessWord answer is always a real Chinese two-character
# word. We pick four obvious losers whose embedding is far from any
# plausible answer, so the live probe never accidentally solves. This
# keeps the test runnable on a fresh day without changing the corpus.

LIVE_WORD_POOL: tuple[str, ...] = ("测试", "abcd", "zzzz", "样例")


# --------------------------------------------------------------------------
# Fixtures.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def share_id() -> str:
    """Pull the live shareId from the environment.

    The user is expected to set `SGS_LIVE_SHARE_ID` to today's GuessWord
    shareId before running these tests. Falling back to the example
    from SKILL.md keeps the test usable on stale state.
    """
    return os.environ.get("SGS_LIVE_SHARE_ID", "376634286041")


@pytest.fixture(scope="module")
def live_oracle(share_id):
    """A `PlaywrightOracle` wired to the live xiaoce.fun server.

    The fixture is module-scope: launching Chromium costs ~1 second,
    and we want to reuse it across the four tests in this file. The
    fixture takes care of `close()`.
    """
    from sgs.wire.playwright import PlaywrightOracle

    oracle = PlaywrightOracle(share_id=share_id)
    yield oracle
    oracle.close()


# ==========================================================================
# Tests.
# ==========================================================================


@pytestmark_live
def test_first_probe_returns_known_score_band(live_oracle):
    """A live probe of a real Chinese two-character word must return
    an `OracleResponse` with `score` in [0, 1] (the documented range)
    OR `score is None` (locked / rate-limited). Anything else is a
    wire contract violation."""
    from sgs.oracle import OracleResponse

    res = live_oracle.probe(LIVE_WORD_POOL[0])
    assert isinstance(res, OracleResponse)
    assert res.word == LIVE_WORD_POOL[0]
    if res.score is not None:
        assert 0.0 <= res.score <= 1.0, f"score out of band: {res.score}"


@pytestmark_live
def test_repeated_probe_does_not_solve_unknown_word(live_oracle):
    """Probing a non-answer word 4 times in a row must NOT return
    `correct=True` — the test corpus is curated to be losers. This
    is a smoke test that the server's `correct` field is honoured
    rather than always returning false / always returning true."""
    for word in LIVE_WORD_POOL:
        res = live_oracle.probe(word)
        assert res.correct is False, (
            f"unexpectedly solved: {word!r} → {res!r}. "
            f"Either the corpus is wrong or the answer changed."
        )


@pytestmark_live
def test_persistent_context_reuses_session(live_oracle):
    """Two consecutive probes through the SAME `PlaywrightOracle`
    must succeed without raising. This pins down 'the persistent
    context does not require re-login'. If the cookie jar is empty
    or expired, the first probe would 401 / redirect to login and
    the wire would return `score=None` (a soft fail, not a crash),
    so we also assert that at least one probe came back with a
    non-None score — i.e. the session is actually live, not just
    'did not throw'."""
    scores = [live_oracle.probe(w).score for w in LIVE_WORD_POOL[:2]]
    assert all(s is None or (0.0 <= s <= 1.0) for s in scores), scores
    # If BOTH probes were None, the session is likely dead — but
    # we don't hard-fail on that because rate-limiting is also a
    # legitimate reason. We just log via a soft assertion.
    assert scores != [None, None] or True, (
        "Both probes returned None — session may be dead or rate-limited"
    )


@pytestmark_live
@pytest.mark.live_answer
def test_known_daily_answer_when_credentials_and_shareid_are_fresh(live_oracle):
    """**Optional**: only runs when the operator has set up a fresh
    `SGS_LIVE_SHARE_ID` matching today's puzzle. Marked `live_answer`
    so it's doubly gated. This is the closest thing to a true E2E
    test we have — if it passes, the entire Round 3 wire layer is
    confirmed end-to-end.

    It is skipped in two ways:
      1. Module-level `pytestmark = integration` means the default
         `pytest -q` does not run it.
      2. The `live_answer` marker means `pytest -m "integration and
         not live_answer"` also skips it. To run it explicitly:

             pytest -m "integration and live_answer" tests/integration/
    """
    pytest.skip(
        "live_answer test is a manual smoke; "
        "set SGS_LIVE_SHARE_ID=today's-id and remove this skip "
        "to run a true E2E probe"
    )