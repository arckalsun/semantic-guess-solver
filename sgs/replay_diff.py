"""Round 6 — offline replay regression (`sgs.replay_diff`).

A *replay* is an NDJSON file written by `sgs.solve` (Round 5). One
JSON object per line; each line is the canonical envelope produced by
`OracleResponse.to_ndjson()`:

    {"word": "萧山", "score": 0.95, "correct": false,
     "doubleScore": false, "rateLimited": false}

`sgs.replay_diff` answers the question: "did my new acquisition
strategy regress against last week's run?" WITHOUT re-running the
network. You feed it two replay files; it tells you:

  * How many probes each ran
  * How many words they overlap on
  * The peak score in each (and the delta, signed peak_b - peak_a)
  * Spearman rank correlation on the overlapping scored rows
  * Kendall-tau on the same
  * top-k intersection size (where k = min(n_a, n_b))
  * Probes-to-correct in each (None if neither solved)
  * A `warning` level: "ok" / "warn" / "alarm"

The CLI exposes this:

    python -m sgs.replay_diff a.ndjson b.ndjson [--json] [--threshold 0.02]

Exit codes:
  0  warning=ok
  1  warning=warn (peak regressed beyond threshold)
  2  warning=alarm (no overlap; comparing unrelated sessions)
  3  configuration / IO error

No silent degradation: missing required field → ValueError, missing
file → FileNotFoundError, corrupt JSON → JSONDecodeError. This is a
property of `load_replay` that the test suite pins down.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from sgs.oracle import OracleResponse

__all__ = [
    "ReplayDiffResult",
    "WARNING_OK",
    "WARNING_WARN",
    "WARNING_ALARM",
    "load_replay",
    "compare_runs",
    "cli_main",
]

WARNING_OK: Literal["ok"] = "ok"
WARNING_WARN: Literal["warn"] = "warn"
WARNING_ALARM: Literal["alarm"] = "alarm"

WarningLevel = Literal["ok", "warn", "alarm"]

_REQUIRED_FIELDS = ("word", "score")


# --------------------------------------------------------------------------
# ReplayDiffResult — the public contract.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayDiffResult:
    """One round of `compare_runs`."""

    n_a: int
    n_b: int
    overlap: int
    peak_a: float | None
    peak_b: float | None
    peak_score_delta: float
    spearman: float | None
    kendall: float | None
    topk_intersection: int
    probes_to_correct_a: int | None
    probes_to_correct_b: int | None
    warning: WarningLevel

    def to_dict(self) -> dict:
        return {
            "n_a": self.n_a,
            "n_b": self.n_b,
            "overlap": self.overlap,
            "peak_a": self.peak_a,
            "peak_b": self.peak_b,
            "peak_score_delta": self.peak_score_delta,
            "spearman": self.spearman,
            "kendall": self.kendall,
            "topk_intersection": self.topk_intersection,
            "probes_to_correct_a": self.probes_to_correct_a,
            "probes_to_correct_b": self.probes_to_correct_b,
            "warning": self.warning,
        }


# --------------------------------------------------------------------------
# load_replay — strict NDJSON reader.
# --------------------------------------------------------------------------


def load_replay(path: Path) -> list[OracleResponse]:
    """Read an NDJSON replay file as a list of `OracleResponse`.

    Raises:
        FileNotFoundError: `path` does not exist.
        json.JSONDecodeError: a line is not valid JSON.
        ValueError: a line is missing required fields (`word`, `score`).
            We do NOT silently drop bad rows — a malformed replay is
            a hard schema error.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    out: list[OracleResponse] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                # Re-raise with the line number prepended.
                raise json.JSONDecodeError(
                    f"line {lineno} of {path}: not valid JSON",
                    raw,
                    0,
                )
            for field in _REQUIRED_FIELDS:
                if field not in obj:
                    raise ValueError(
                        f"{path}:{lineno} missing required {field!r} "
                        f"field; got keys {sorted(obj.keys())}"
                    )
            out.append(
                OracleResponse(
                    word=obj["word"],
                    score=obj["score"],
                    correct=bool(obj.get("correct", False)),
                    double_score=bool(obj.get("doubleScore", False)),
                    rate_limited=bool(obj.get("rateLimited", False)),
                )
            )
    return out


# --------------------------------------------------------------------------
# Rank correlation helpers — stdlib-only (no scipy).
# --------------------------------------------------------------------------


def _rank(values: list[float]) -> list[float]:
    """Compute 1-based ranks with mid-rank tie-breaking.

    Pure-Python implementation: O(n²) but we never run on more than
    a few hundred rows so this is fine.
    """
    n = len(values)
    indexed = sorted(enumerate(values), key=lambda iv: iv[1])
    ranks: list[float] = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        # Tie block: indexed[i..j] inclusive. Mid-rank.
        mid = (i + j) / 2.0 + 1.0  # 1-based
        for k in range(i, j + 1):
            orig_idx = indexed[k][0]
            ranks[orig_idx] = mid
        i = j + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0.0 or den_y == 0.0:
        return None
    return num / (den_x * den_y)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman ρ — Pearson on the rank-transformed values."""
    rx = _rank(xs)
    ry = _rank(ys)
    return _pearson(rx, ry)


def kendall_tau(xs: list[float], ys: list[float]) -> float | None:
    """Kendall τ-b (no tie correction). n must be ≥ 2."""
    n = len(xs)
    if n < 2:
        return None
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            if dx == 0.0 or dy == 0.0:
                continue  # tie; not counted in either bucket
            if (dx > 0) == (dy > 0):
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return None
    return (concordant - discordant) / total


# --------------------------------------------------------------------------
# compare_runs — the core.
# --------------------------------------------------------------------------


def _peak(rows: list[OracleResponse]) -> float | None:
    """Best scored row; None if no scored rows (all are locks / rate-limited)."""
    scored = [r.score for r in rows if r.score is not None]
    if not scored:
        return None
    return max(scored)


def _probes_to_correct(rows: list[OracleResponse]) -> int | None:
    """1-based probe index at which `correct=True` was reached; None if never."""
    for i, r in enumerate(rows, start=1):
        if r.correct:
            return i
    return None


def _topk_intersection(
    rows_a: list[OracleResponse],
    rows_b: list[OracleResponse],
    k: int,
) -> int:
    """|top_k(A) ∩ top_k(B)| where top_k is the k best-scored words."""
    def topk(rows: Iterable[OracleResponse], k: int) -> set[str]:
        scored = [r for r in rows if r.score is not None]
        scored.sort(key=lambda r: r.score, reverse=True)
        return {r.word for r in scored[:k]}

    return len(topk(rows_a, k) & topk(rows_b, k))


def compare_runs(
    path_a: Path,
    path_b: Path,
    *,
    threshold: float = 0.02,
) -> ReplayDiffResult:
    """Compare two replay NDJSON files.

    Args:
        path_a: baseline replay (e.g. last week's run).
        path_b: candidate replay (e.g. today's new acquisition).
        threshold: absolute peak_score regression (peak_b - peak_a
            < -threshold) that triggers `warning='warn'`. Pass 0
            to disable numeric-warning (only structural alarms fire).

    Returns:
        A `ReplayDiffResult`. Always returns — even on `warning=alarm`
        — so callers can render the diff.

    Notes:
        * `spearman` / `kendall` are computed on the intersection of
          words both replays scored, in probe order from path_a.
        * `peak_score_delta` is signed `peak_b - peak_a`.
    """
    rows_a = load_replay(path_a)
    rows_b = load_replay(path_b)

    score_a: dict[str, float] = {
        r.word: r.score for r in rows_a if r.score is not None
    }
    score_b: dict[str, float] = {
        r.word: r.score for r in rows_b if r.score is not None
    }

    overlap_words = sorted(set(score_a) & set(score_b))
    overlap = len(overlap_words)

    if overlap == 0:
        # Sentinel: comparing unrelated sessions. We still return
        # the structural metrics so the CLI can render them, but
        # the warning is `alarm`.
        return ReplayDiffResult(
            n_a=len(rows_a),
            n_b=len(rows_b),
            overlap=0,
            peak_a=_peak(rows_a),
            peak_b=_peak(rows_b),
            peak_score_delta=(_peak(rows_b) or 0.0) - (_peak(rows_a) or 0.0),
            spearman=None,
            kendall=None,
            topk_intersection=0,
            probes_to_correct_a=_probes_to_correct(rows_a),
            probes_to_correct_b=_probes_to_correct(rows_b),
            warning=WARNING_ALARM,
        )

    xs = [score_a[w] for w in overlap_words]
    ys = [score_b[w] for w in overlap_words]

    pa = _peak(rows_a) or 0.0
    pb = _peak(rows_b) or 0.0
    delta = pb - pa

    if threshold > 0 and delta < -threshold:
        warning: WarningLevel = WARNING_WARN
    else:
        warning = WARNING_OK

    k = min(len(rows_a), len(rows_b))

    return ReplayDiffResult(
        n_a=len(rows_a),
        n_b=len(rows_b),
        overlap=overlap,
        peak_a=pa if _peak(rows_a) is not None else None,
        peak_b=pb if _peak(rows_b) is not None else None,
        peak_score_delta=delta,
        spearman=spearman(xs, ys),
        kendall=kendall_tau(xs, ys),
        topk_intersection=_topk_intersection(rows_a, rows_b, k),
        probes_to_correct_a=_probes_to_correct(rows_a),
        probes_to_correct_b=_probes_to_correct(rows_b),
        warning=warning,
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sgs.replay_diff",
        description=(
            "Compare two NDJSON replay files written by `sgs.solve`. "
            "Reports rank correlations, peak-score delta, top-k "
            "intersection, and probes-to-correct."
        ),
    )
    parser.add_argument(
        "path_a",
        type=Path,
        help="Baseline replay NDJSON (e.g. last week's run).",
    )
    parser.add_argument(
        "path_b",
        type=Path,
        help="Candidate replay NDJSON (e.g. today's new acquisition).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.02,
        help=(
            "Absolute peak_score regression that triggers warning='warn' "
            "(default 0.02). Pass 0 to disable numeric warnings."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON ReplayDiffResult to stdout.",
    )
    return parser


def _format_summary(res: ReplayDiffResult) -> str:
    def f(v: float | None, fmt: str = "+.4f") -> str:
        return format(v, fmt) if v is not None else "n/a"

    return (
        f"n_a={res.n_a} n_b={res.n_b} overlap={res.overlap} "
        f"peak_a={f(res.peak_a)} peak_b={f(res.peak_b)} "
        f"peak_delta={f(res.peak_score_delta)} "
        f"spearman={f(res.spearman)} kendall={f(res.kendall)} "
        f"topk={res.topk_intersection} "
        f"probes_correct_a={res.probes_to_correct_a or 'n/a'} "
        f"probes_correct_b={res.probes_to_correct_b or 'n/a'} "
        f"warning={res.warning}"
    )


def cli_main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code (0/1/2/3)."""
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        if isinstance(code, int) and code != 0:
            return 3
        raise

    try:
        res = compare_runs(args.path_a, args.path_b, threshold=args.threshold)
    except FileNotFoundError as exc:
        target = exc.filename or str(exc.args[0] if exc.args else exc)
        print(f"ERROR: replay file not found: {target}", file=sys.stderr)
        return 3
    except json.JSONDecodeError as exc:
        print(f"ERROR: {exc.msg}", file=sys.stderr)
        return 3
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    if args.json:
        print(json.dumps(res.to_dict(), ensure_ascii=False))
    else:
        print(_format_summary(res))

    return {
        WARNING_OK: 0,
        WARNING_WARN: 1,
        WARNING_ALARM: 2,
    }[res.warning]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())