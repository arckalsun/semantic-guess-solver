"""Token-bucket rate limiter — keep batches polite, dodge server bans.

Empirical notes from the case study:

* The GuessWord oracle rejects roughly 12 of every 30 words in a batch
  (rate-limit) when probes land faster than ~120 ms apart.
* A token bucket with ``rate=0.8 tokens/s`` and ``burst=2`` (≈ 800 ms
  per call, but allows two quick back-to-back) hits a clean record.

This implementation is stdlib-only so the library stays 0-dep.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager


class TokenBucket:
    """Simple thread-safe token bucket.

    Parameters
    ----------
    rate
        Tokens added per second. ``rate=0.8`` ⇒ one token every 1.25 s.
    burst
        Maximum bucket capacity. ``burst=2`` allows two fast probes
        followed by a refill gap.
    """

    def __init__(self, rate: float, burst: int = 1) -> None:
        if rate <= 0:
            raise ValueError(f"rate must be positive, got {rate}")
        if burst < 1:
            raise ValueError(f"burst must be ≥1, got {burst}")
        self._rate = rate
        self._burst = burst
        self._tokens = float(burst)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
        self._last = now

    def consume(self, n: int = 1) -> None:
        """Block until ``n`` tokens are available, then take them."""
        if n < 1:
            raise ValueError(f"n must be ≥1, got {n}")
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= n:
                    self._tokens -= n
                    return
                deficit = n - self._tokens
                wait = deficit / self._rate
            time.sleep(wait)

    @contextmanager
    def take(self, n: int = 1) -> Iterator[None]:
        """Context-manager sugar around :meth:`consume`."""
        self.consume(n)
        yield


def make_default_bucket() -> TokenBucket:
    """The canonical GuessWord bucket: rate=0.8, burst=2."""
    return TokenBucket(rate=0.8, burst=2)