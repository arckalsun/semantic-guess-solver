"""semantic-guess-solver (sgs) — Round 1 + Round 2.

Round 1 (offline): pure-numpy ranking + tamper-evident NDJSON replay.
Round 2 (online): API contract adapter + batch probe + TokenBucket.
v0.7.0 (2026-07-15): kernel-ridge-regression predictor (`sgs.krr`) for
embedding-space-misalignment plateaus (case-11 broke the 0.82 plateau
in 9 probes after 632 centroid-stalled brute-force probes).
v0.8.0 (2026-07-15): visitor-accessible daily mode (`sgs.daily_solve`).
The platform daily GuessWord challenge is reachable via
`/guessV1?date=YYYYMMDD` — no shareId, no login. Bypasses the
share/create login wall. case-daily-2026-07-15 (南宁) solved
cleanly via the centroid alone in 150 probes; KRR kicks in past
100 obs or peak≥0.85.

A 0-dependency (numpy only) library for ranking Chinese candidate words
against the xiaoce.fun GuessWord daily oracle.

Submodules:

* :mod:`sgs.replay` — NDJSON read/write + sha256 fingerprint.
* :mod:`sgs.rank` — cosine-similarity ranking against a frozen embedding
  matrix (centroid heuristic) **plus** kernel-ridge-regression ranking
  (`rank_by_predictor`) for plateau cases.
* :mod:`sgs.krr` — Kernel Ridge Regression predictor (RBF kernel,
  closed-form solve). The brain behind ``rank_by_predictor``.
* :mod:`sgs.round1` — CLI entry-point (use as ``python -m sgs.round1``);
  pass ``--predictor`` to switch from centroid to KRR ranking.
* :mod:`sgs.oracle` — ``Oracle`` protocol + ``OracleResponse`` + ``FakeOracle``.
* :mod:`sgs.ratelimit` — ``TokenBucket`` (default rate=0.8/s, burst=2).
* :mod:`sgs.probe` — ``probe_batch`` + ``probe_and_record``.

Tests live in ``tests/`` and run with ``pytest``.

License: MIT.

Note on imports
---------------

``sgs.__init__`` deliberately exposes **only Round 2 symbols** because
they're new and benefit from a stable top-level alias. Round 1 has no
`__init__.py` exports — call its functions via ``from sgs.rank import rank``,
``from sgs.replay import fingerprint``, etc., or invoke the CLI via
``python -m sgs.round1``. Keeping the package surface tiny makes the
public API obvious and prevents accidental export drift.
"""

from __future__ import annotations

__version__ = "0.8.0"

__all__ = [
    # Round 2 — online probe layer
    "Oracle",
    "OracleResponse",
    "FakeOracle",
    "BatchResult",
    "probe_batch",
    "probe_and_record",
    "TokenBucket",
    "make_default_bucket",
]

# Round 2
from sgs.oracle import FakeOracle, Oracle, OracleResponse
from sgs.probe import BatchResult, probe_and_record, probe_batch
from sgs.ratelimit import TokenBucket, make_default_bucket