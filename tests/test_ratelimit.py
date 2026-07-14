"""Tests for ``sgs.ratelimit`` — TokenBucket.

We use a virtual clock so the suite stays sub-second even at rate=0.8.
"""

from __future__ import annotations

import time

import pytest

from sgs.ratelimit import TokenBucket, make_default_bucket


class _Clock:
    """Manual clock for deterministic bucket tests."""

    def __init__(self) -> None:
        self.now = 1000.0  # arbitrary monotonic start

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += dt


def test_rate_must_be_positive() -> None:
    with pytest.raises(ValueError):
        TokenBucket(rate=0)


def test_burst_must_be_at_least_one() -> None:
    with pytest.raises(ValueError):
        TokenBucket(rate=1.0, burst=0)


def test_consume_blocks_until_token_available(monkeypatch) -> None:
    """A burst=1 bucket at rate=1 needs exactly 1 s for the 2nd probe."""
    # Patch time BEFORE constructing the bucket so __init__ captures the
    # virtual clock as its baseline, not real monotonic().
    clock = _Clock()
    monkeypatch.setattr(time, "monotonic", clock)
    bucket = TokenBucket(rate=1.0, burst=1)

    bucket.consume()  # first call: immediate (full bucket)
    # Now bucket is empty. consume must block ~1 s.
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s) or clock.advance(s))
    bucket.consume()
    # The sleep should be ≈ 1.0 (one token of rate=1 per second).
    assert sleeps == pytest.approx([1.0], rel=0.05)


def test_burst_allows_back_to_back_probes() -> None:
    bucket = TokenBucket(rate=0.8, burst=2)
    # Two immediate consumes should NOT block (bucket starts at burst=2).
    bucket.consume()
    bucket.consume()


def test_take_context_manager_consumes_one_token() -> None:
    bucket = TokenBucket(rate=10.0, burst=1)
    with bucket.take():
        assert bucket._tokens == pytest.approx(0.0, abs=1e-9)
    bucket.consume()  # second call should still work after refill


def test_make_default_bucket_uses_documented_params() -> None:
    """Pin the recipe so a docs/code drift fails CI."""
    b = make_default_bucket()
    assert b._rate == 0.8
    assert b._burst == 2