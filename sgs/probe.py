"""probe: drive a batch of words through an oracle, write to NDJSON.

This module is the **only place** in the offline package that calls the
oracle. Round-1 (ranker) is pure numpy and never imports probe; probe
imports Round-1 to decide what to ask next, but the round-1 dependency
is the **function**, not the CLI.

Operational notes from the case study:

* **One batch = 30 words** (matches the daily-game UI default).
* **Server-side lock**: once a word has been correctly guessed, the
  oracle returns ``data:null`` for subsequent probes of the same word.
  Probe skips that word in subsequent batches.
* **Rate limit**: ~12/30 words return ``rateLimited=true`` if probes
  land <120 ms apart. Use :class:`TokenBucket`.
* **Stop condition**: the very first response with ``correct=True``
  ends the loop. We do *not* wait for the whole batch.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from sgs.oracle import Oracle, OracleResponse
from sgs.ratelimit import TokenBucket, make_default_bucket


@dataclass
class BatchResult:
    """Summary of a single probe batch.

    Attributes
    ----------
    responses
        All oracle responses, in the order they came back.
    hit
        The response with ``correct=True``, or ``None`` if the batch
        did not surface the answer.
    rate_limited
        Count of probes the oracle flagged ``rateLimited``.
    """

    responses: list[OracleResponse]
    hit: OracleResponse | None
    rate_limited: int

    def to_ndjson(self) -> list[dict]:
        return [r.to_ndjson() for r in self.responses]


def probe_batch(
    words: Sequence[str],
    oracle: Oracle,
    *,
    bucket: TokenBucket | None = None,
    stop_on_correct: bool = True,
) -> BatchResult:
    """Probe each ``word`` in order. Returns a :class:`BatchResult`.

    Parameters
    ----------
    words
        The candidate batch, usually the top-30 from :func:`sgs.rank.rank`.
    oracle
        Any :class:`Oracle` (real or fake). The fake path is what tests use.
    bucket
        Optional :class:`TokenBucket`. Defaults to
        :func:`make_default_bucket` (rate=0.8, burst=2).
    stop_on_correct
        If ``True`` (default), abort the batch as soon as the oracle
        returns ``correct=True``. The remaining words are *not* probed.
        Disable for audits where you want the full batch.
    """
    if not words:
        return BatchResult(responses=[], hit=None, rate_limited=0)
    if bucket is None:
        bucket = make_default_bucket()

    out: list[OracleResponse] = []
    rl = 0
    hit: OracleResponse | None = None
    for w in words:
        with bucket.take():
            resp = oracle.probe(w)
        out.append(resp)
        if resp.rate_limited:
            rl += 1
        if resp.correct:
            hit = resp
            if stop_on_correct:
                break
    return BatchResult(responses=out, hit=hit, rate_limited=rl)


def probe_and_record(
    words: Sequence[str],
    oracle: Oracle,
    replay_path: Path,
    *,
    bucket: TokenBucket | None = None,
    stop_on_correct: bool = True,
) -> BatchResult:
    """Same as :func:`probe_batch` but also **appends** the responses as
    NDJSON to ``replay_path`` (one JSON object per line).

    The file is created if missing. Existing content is **preserved**:
    each batch is appended, never overwritten, so Round 1 → Round 2 → Round 3
    share one continuous replay log per shareId.
    """
    result = probe_batch(
        words, oracle, bucket=bucket, stop_on_correct=stop_on_correct,
    )
    if result.responses:
        with open(replay_path, "a", encoding="utf-8") as fh:
            for r in result.responses:
                fh.write(json.dumps(r.to_ndjson(), ensure_ascii=False))
                fh.write("\n")
    return result


def already_correct(responses: Iterable[OracleResponse]) -> set[str]:
    """Return the subset of words that the oracle has confirmed correct
    across a sequence of past responses. Used to build the skip-list
    for subsequent batches (the server-side ``data:null`` lock)."""
    return {r.word for r in responses if r.correct}