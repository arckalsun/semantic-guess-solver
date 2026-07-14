"""Round-1 entry point: drive one probe batch from current observations.

CLI usage::

    python -m sgs.round1 \
        --replay replay/<shareId>.ndjson \
        --candidates /tmp/cand_words.json \
        --embeddings /tmp/cand_emb.npy \
        --batch-size 30 \
        --out replay/<shareId>-next.ndjson

Behaviour:

1. Load the existing replay (if any) — gives us the observed
   ``(word, score)`` pairs.
2. Build a candidate pool from the JSON + .npy pair (must agree in size).
3. Fit a score-weighted centroid over the observed words.
4. Cosine-rank every un-probed candidate and emit the top-``--batch-size``
   words as the *next probe batch*.
5. Optionally write the batch as an NDJSON file (one record per line,
   ``{"word": ..., "rank": i, "score": sim}``).

The script refuses to run with zero observations — Round 1 is online
learning, you need at least one oracle response first.

Why ``argparse`` and not ``typer`` / ``click``: we promised *0-dependency*
for Round 1. ``argparse`` is stdlib and sufficient for a single CLI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .rank import load_corpus, rank
from .replay import read_replay


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m sgs.round1",
        description="Round-1 cosine-similarity ranker for the GuessWord oracle.",
    )
    p.add_argument(
        "--replay",
        type=Path,
        required=True,
        help="Existing NDJSON replay (may be empty file for round-1 kickoff).",
    )
    p.add_argument(
        "--candidates",
        type=Path,
        required=True,
        help="JSON list[str] of candidate words.",
    )
    p.add_argument(
        "--embeddings",
        type=Path,
        required=True,
        help="(N, D) float32 .npy aligned with --candidates.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=30,
        help="Number of probe words to emit (default 30).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional NDJSON output path for the proposed batch.",
    )
    p.add_argument(
        "--include-correct",
        action="store_true",
        help="Include already-correct guesses in the ranking (for audit).",
    )
    return p.parse_args(argv)


def _load_observations(replay: Path) -> list[tuple[str, float]]:
    if not replay.exists():
        raise FileNotFoundError(f"replay file not found: {replay}")
    if replay.stat().st_size == 0:
        return []
    rows = read_replay(replay)
    obs: list[tuple[str, float]] = []
    for r in rows:
        obs.append((str(r["word"]), float(r["score"])))
    return obs


def _emit_batch(
    ranked: list[tuple[str, float]],
    batch_size: int,
    out: Path | None,
) -> None:
    """Print to stdout and optionally write to ``out`` as NDJSON."""
    sliced = ranked[:batch_size]
    if out is not None:
        with open(out, "w", encoding="utf-8") as fh:
            for i, (w, s) in enumerate(sliced):
                fh.write(
                    json.dumps(
                        {"word": w, "rank": i, "score": round(float(s), 6)},
                        ensure_ascii=False,
                    )
                )
                fh.write("\n")
    for i, (w, s) in enumerate(sliced):
        print(f"{i:>3}  {w}  {s:.4f}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    obs = _load_observations(args.replay)
    if not obs:
        print(
            f"error: replay {args.replay} is empty — Round 1 needs at least "
            "one observation to fit a centroid. Probe the oracle first, "
            "record the (word, score) pair in the replay, then re-run.",
            file=sys.stderr,
        )
        return 2

    words, emb = load_corpus(args.candidates, args.embeddings)
    ranked = rank(
        obs,
        words,
        emb,
        top_k=args.batch_size,
        exclude_observed=not args.include_correct,
    )
    if not ranked:
        print(
            f"warning: every candidate was already probed "
            f"({len(obs)} observations, {len(words)} candidates). "
            "Nothing left to rank.",
            file=sys.stderr,
        )
        return 1

    _emit_batch(ranked, args.batch_size, args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())