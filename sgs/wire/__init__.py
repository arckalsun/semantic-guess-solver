"""sgs.wire: real-world Oracle implementations that talk to xiaoce.fun.

Currently only :mod:`sgs.wire.playwright` (Playwright-driven browser
fetch via a persistent context). The bare-intent is to keep this as the
*only* directory that imports ``playwright``, so the rest of ``sgs`` stays
install-pip-free.

Login flow (first-run)
----------------------
The user must ``python -m playwright open-url https://xiaoce.fun`` once so
the persistent Chromium context under ``./.playwright-data/`` accumulates
the xiaoce session cookie. After that, ``PlaywrightOracle`` reuses the
session across invocations.

Throttling
----------
The wire does NOT rate-limit itself — that's :mod:`sgs.ratelimit`'s job.
A token bucket is composed at the CLI layer (``python -m sgs.round3``).
"""

from __future__ import annotations

# Re-export the runtime helpers that callers actually need.
from sgs.wire.base import (
    DEFAULT_HEADERS,
    GUESS_PATH,
    RateLimitSignal,
    WireEndpoint,
    parse_response,
)

__all__ = [
    "DEFAULT_HEADERS",
    "GUESS_PATH",
    "RateLimitSignal",
    "WireEndpoint",
    "parse_response",
    # Lazy import — only loaded when ``PlaywrightOracle`` is constructed.
    # Imported explicitly in the module body when ``__getattr__`` fires:
    "PlaywrightOracle",
]


def __getattr__(name: str):  # PEP 562 lazy import
    if name == "PlaywrightOracle":
        from sgs.wire.playwright import PlaywrightOracle
        return PlaywrightOracle
    raise AttributeError(f"module 'sgs.wire' has no attribute {name!r}")