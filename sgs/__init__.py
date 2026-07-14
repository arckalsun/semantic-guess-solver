"""semantic-guess-solver (sgs) — Round 1 offline ranker.

A pure-numpy, 0-dependency (numpy only) library for ranking Chinese candidate
words against a black-box similarity oracle — built for the xiaoce.fun
GuessWord daily challenge.

Three modules ship in Round 1:

* :mod:`sgs.replay` — load + verify + sha256 fingerprint for NDJSON replay
  files (one JSON object per line: probe → score response pairs).
* :mod:`sgs.rank` — cosine-similarity ranking against a frozen embedding
  matrix (e.g. BGE-zh-base 768-d).
* :mod:`sgs.round1` — entry-point that takes a ``shareId`` plus a candidate
  pool + observed scores and emits the next probe batch (default 30 words).

Tests live in ``tests/`` and run with ``pytest``.

License: MIT.
"""

__version__ = "0.1.0"