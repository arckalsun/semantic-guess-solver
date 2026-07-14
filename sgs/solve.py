"""Round 5 — End-to-end solver orchestrator + CLI.

This module is the **thin glue** that connects three earlier rounds:

  * Round 2  Oracle protocol (`sgs.oracle`) + replay NDJSON
  * Round 3  Wire implementations (`sgs.wire.playwright.PlaywrightOracle`)
  * Round 4  Acquisition strategies (`sgs.learn.{Random,Greedy,
            Uncertainty}Acquisition`) + `active_solve` orchestrator

It does NOT introduce new network or policy logic. Its job is to:

  1. Validate CLI arguments (candidates file exists, acquisition
     name is known, max-probes is positive).
  2. Build an Oracle from a plugin name (`fake` or `playwright`).
     Playwright is imported **lazily**, only when the user actually
     asks for the live wire.
  3. Dispatch the Acquisition strategy by name.
  4. Wrap `active_solve` with plateau detection — if the best score
     observed has not improved for `plateau_window` consecutive
     probes, stop early and report `stop_reason="plateau"`.
  5. Always call `oracle.close()` (P0: never leak browser resources).
  6. Append every probed response to a NDJSON replay log, one JSON
     object per line, in probe order.
  7. Return a `SolveResult` dataclass the CLI can render.

CLI usage (from a shell):

    python -m sgs.solve run \\
        --candidates candidates.txt \\
        --replay out/<shareId>.ndjson \\
        --oracle fake \\
        --acquisition greedy \\
        --max-probes 50 \\
        --plateau-window 5

Exit codes:

    0  solved (correct=True) OR dry-run completed (no correct target)
    2  budget exhausted without `correct=True`
    3  configuration error (missing file, unknown strategy, etc.)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from sgs.learn import (
    Acquisition,
    GreedyAcquisition,
    RandomAcquisition,
    UncertaintyAcquisition,
    active_solve,
)
from sgs.oracle import Oracle, OracleResponse

__all__ = [
    "SolveResult",
    "solve_run",
    "cli_main",
    "build_oracle",
    "build_acquisition",
]

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Result contract
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SolveResult:
    """What `solve_run` returns. Stable contract — the CLI renders
    this as one summary line and (optionally) a JSON blob."""

    solved: bool
    stop_reason: Literal["correct", "budget", "plateau"]
    probes_used: int
    peak_word: str | None
    peak_score: float | None
    responses: tuple[OracleResponse, ...]

    def to_dict(self) -> dict:
        return {
            "solved": self.solved,
            "stop_reason": self.stop_reason,
            "probes_used": self.probes_used,
            "peak_word": self.peak_word,
            "peak_score": self.peak_score,
            "responses": [r.to_ndjson() for r in self.responses],
        }


# --------------------------------------------------------------------------
# Oracle plugin — lazy import for playwright so `import sgs.solve` does
# not pull in the heavy browser stack.
# --------------------------------------------------------------------------


def build_oracle(
    name: str,
    *,
    share_id: str | None = None,
) -> Oracle:
    """Resolve an oracle plugin by name.

    `fake`           — returns a zero-score oracle, useful for dry-runs
                       where you only want to exercise the loop without
                       any network or script.
    `playwright`     — instantiates PlaywrightOracle. REQUIRES that the
                       user is already logged in via a persistent
                       Chromium context (see Round 3 docs). The import
                       happens here, NOT at module load.
    """
    if name == "fake":
        from sgs.oracle import FakeOracle  # local import — keeps oracle tests fast
        return FakeOracle()
    if name == "playwright":
        # Lazy import — keeps `import sgs.solve` light. This is the
        # ONE place playwright is allowed to leak in.
        from sgs.wire.playwright import PlaywrightOracle
        if not share_id:
            raise ValueError(
                "oracle 'playwright' requires --share-id <id> "
                "(the xiaoce.fun GuessWord session id)"
            )
        return PlaywrightOracle(share_id=share_id)
    raise ValueError(
        f"unknown oracle: {name!r}; expected one of: fake, playwright"
    )


# --------------------------------------------------------------------------
# Acquisition dispatch
# --------------------------------------------------------------------------


def build_acquisition(name: str) -> Acquisition:
    if name == "random":
        return RandomAcquisition(seed=random.randint(0, 2**32 - 1))
    if name == "greedy":
        return GreedyAcquisition()
    if name == "uncertainty":
        return UncertaintyAcquisition()
    raise ValueError(
        f"unknown acquisition: {name!r}; "
        f"expected one of: random, greedy, uncertainty"
    )


# --------------------------------------------------------------------------
# Plateau detection — pure helper, easy to unit test if you ever want to
# --------------------------------------------------------------------------


def _plateau_breached(
    history: list[OracleResponse],
    window: int,
    peak: float,
) -> bool:
    """Return True iff the last `window` responses are all strictly
    below `peak` (i.e. we have not seen a new peak in `window` probes)."""
    if window <= 0 or len(history) < window:
        return False
    tail = history[-window:]
    return all(
        (r.score is None) or (r.score < peak) for r in tail
    )


# --------------------------------------------------------------------------
# Replay NDJSON — append-only, one JSON object per line
# --------------------------------------------------------------------------


def _append_replay(
    path: Path | None,
    responses: Iterable[OracleResponse],
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for r in responses:
            fh.write(json.dumps(r.to_ndjson(), ensure_ascii=False))
            fh.write("\n")


# --------------------------------------------------------------------------
# Core orchestrator — callable from a notebook, the CLI, or tests
# --------------------------------------------------------------------------


def solve_run(
    *,
    oracle: Oracle,
    candidates: Iterable[str],
    acquisition_name: str,
    max_probes: int,
    replay_path: Path | None,
    plateau_window: int = 5,
) -> SolveResult:
    """Run the end-to-end solve loop.

    Args:
        oracle: Anything with a `probe(word)` method.
        candidates: Pool of words to probe. Will be de-duplicated
            in probe order.
        acquisition_name: One of "random", "greedy", "uncertainty".
        max_probes: Hard cap on probe count. Must be ≥ 1.
        replay_path: NDJSON file to append to. Created if missing;
            parent directories auto-created.
        plateau_window: Stop after this many consecutive probes with
            no improvement on the best observed score. Must be ≥ 0
            (0 disables plateau detection).

    Returns:
        A `SolveResult` dataclass. The caller is responsible for
        interpreting `solved` and `stop_reason`.

    Raises:
        ValueError on misconfiguration (unknown strategy, empty
            candidates, max_probes ≤ 0, plateau_window < 0).
    """
    # ---- Argument validation BEFORE any probing ----
    if max_probes <= 0:
        raise ValueError(f"max_probes must be ≥ 1, got {max_probes}")
    if plateau_window < 0:
        raise ValueError(
            f"plateau_window must be ≥ 0, got {plateau_window}"
        )
    candidates_list = list(candidates)
    if not candidates_list:
        raise ValueError("candidates must be a non-empty iterable")

    # Resolve acquisition BEFORE we probe so config errors fail fast.
    acquisition = build_acquisition(acquisition_name)

    try:
        history: list[OracleResponse] = []
        peak_word: str | None = None
        peak_score: float | None = None
        stop_reason: str = "budget"

        # We don't use `active_solve` directly here because we need
        # fine-grained plateau detection between probes — and we
        # want to drive the oracle manually so we can call
        # `oracle.close()` on the way out.
        seen: set[str] = set()
        pending: list[str] = list(dict.fromkeys(candidates_list))

        for _ in range(max_probes):
            if not pending:
                break

            # Score all unseen candidates with the acquisition.
            scores = acquisition.score(history, pending)
            assert len(scores) == len(pending), (
                f"acquisition {acquisition_name!r} returned "
                f"{len(scores)} scores for {len(pending)} candidates"
            )

            # Pick the highest-scored pending candidate.
            ranked = sorted(
                zip(pending, scores), key=lambda x: x[1], reverse=True
            )
            next_word = ranked[0][0]

            seen.add(next_word)
            pending = [w for w in pending if w != next_word]

            response = oracle.probe(next_word)
            history.append(response)
            _append_replay(replay_path, [response])

            # Update peak tracking. `score is None` is a lock/rate
            # response — it does NOT improve the peak and it does NOT
            # count toward the plateau window (we filter that case
            # in `_plateau_breached`).
            if response.score is not None and (
                peak_score is None or response.score > peak_score
            ):
                peak_score = response.score
                peak_word = response.word

            if response.correct:
                stop_reason = "correct"
                break

            if plateau_window > 0 and _plateau_breached(
                history, plateau_window, peak_score or float("-inf")
            ):
                stop_reason = "plateau"
                break

        # If we ran out of pending without ever breaking, the loop
        # terminated cleanly via "no more candidates" — treat that
        # as budget (we did not solve).
        if stop_reason == "budget" and not pending:
            stop_reason = "budget"

        solved = stop_reason == "correct"

        return SolveResult(
            solved=solved,
            stop_reason=stop_reason,  # type: ignore[arg-type]
            probes_used=len(history),
            peak_word=peak_word,
            peak_score=peak_score,
            responses=tuple(history),
        )
    finally:
        # P0: never leak browser resources, even on budget exhaust
        # or mid-loop exception.
        try:
            oracle.close()
        except Exception:  # pragma: no cover — best-effort cleanup
            logger.exception("oracle.close() raised; swallowing")


# --------------------------------------------------------------------------
# CLI — `python -m sgs.solve run ...`
# --------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sgs.solve",
        description=(
            "End-to-end GuessWord solver. Wires an Oracle "
            "(Round 2/3) to an Acquisition strategy (Round 4) and "
            "records every probe as NDJSON."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run a solve loop")
    run.add_argument(
        "--candidates",
        type=Path,
        required=True,
        help="UTF-8 file, one candidate per line",
    )
    run.add_argument(
        "--replay",
        type=Path,
        default=None,
        help="NDJSON file to append probe results to (created if missing)",
    )
    run.add_argument(
        "--oracle",
        choices=["fake", "playwright"],
        default="fake",
        help="Oracle backend (default: fake — no network)",
    )
    run.add_argument(
        "--acquisition",
        default="greedy",
        help="Acquisition strategy: one of random/greedy/uncertainty (default: greedy)",
    )
    run.add_argument(
        "--max-probes",
        type=int,
        default=50,
        help="Hard cap on probe count (default: 50)",
    )
    run.add_argument(
        "--plateau-window",
        type=int,
        default=5,
        help=(
            "Stop after this many consecutive probes with no "
            "improvement on the best observed score. 0 disables "
            "(default: 5)"
        ),
    )
    run.add_argument(
        "--share-id",
        type=str,
        default=None,
        help="Required when --oracle playwright",
    )
    run.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON SolveResult to stdout instead of a one-line summary",
    )
    return parser


def _read_candidates(path: Path) -> list[str]:
    """Read candidates file, one per line, ignoring blanks and `#` comments."""
    if not path.exists():
        # We let the CLI surface the actionable error.
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


def cli_main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code (0/2/3)."""
    parser = _build_parser()
    # Translate argparse's default `error: ...` → SystemExit(2) into
    # our exit-3 contract (configuration error). Help (exit 0) and
    # the rare KeyboardInterrupt (exit 130) are left alone.
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        if isinstance(code, int) and code != 0:
            return 3
        raise

    # ---- Configuration errors: exit 3 (distinct from "no answer") ----
    # argparse's own `--help` is still exit 0 (default); any
    # argparse validation failure (unknown --acquisition, bad
    # --max-probes type, missing required flag) is converted to
    # exit 3 so operators can distinguish "I misconfigured" from
    # "the run produced no answer" (exit 2) from "solved" (exit 0).
    if args.command != "run":
        parser.print_help()
        return 3

    try:
        candidates = _read_candidates(args.candidates)
    except FileNotFoundError:
        print(
            f"ERROR: candidates file not found: {args.candidates}",
            file=sys.stderr,
        )
        return 3
    if not candidates:
        print(
            f"ERROR: candidates file is empty: {args.candidates}",
            file=sys.stderr,
        )
        return 3

    # Cross-field validation: --oracle playwright requires --share-id.
    # argparse `choices=` does not catch this; we own it.
    if args.oracle == "playwright" and not args.share_id:
        print(
            "ERROR: --oracle playwright requires --share-id <id>",
            file=sys.stderr,
        )
        return 3

    try:
        oracle = build_oracle(
            args.oracle,
            share_id=args.share_id,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    # ---- Run ----
    try:
        result = solve_run(
            oracle=oracle,
            candidates=candidates,
            acquisition_name=args.acquisition,
            max_probes=args.max_probes,
            replay_path=args.replay,
            plateau_window=args.plateau_window,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    # ---- Render ----
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False))
    else:
        peak = (
            f"{result.peak_score:.4f}" if result.peak_score is not None
            else "n/a"
        )
        print(
            f"solved={result.solved} stop={result.stop_reason} "
            f"probes={result.probes_used} peak={result.peak_word} "
            f"score={peak}"
        )

    # ---- Exit code ----
    if result.solved:
        return 0
    if result.stop_reason == "budget":
        return 2
    # plateau or any other stop reason without `correct=True`
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())