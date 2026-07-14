from __future__ import annotations

import logging
import random
from typing import Iterable, Protocol, runtime_checkable

from sgs.oracle import Oracle, OracleResponse

logger = logging.getLogger("sgs.learn")

__all__ = [
    "Acquisition",
    "RandomAcquisition",
    "GreedyAcquisition",
    "UncertaintyAcquisition",
    "active_solve",
]


@runtime_checkable
class Acquisition(Protocol):
    """An acquisition function scores candidates by expected utility.

    Implementations receive ``history`` (already-probed responses) and
    ``candidates`` (not-yet-probed words). They return a parallel list
    of scores in the same order as ``candidates``. Higher score =
    probe next.

    The simple strategies (``GreedyAcquisition`` etc.) ignore the
    history; an *informed* learner with a predictor (Round 4's
    future fitted model) would mix both.
    """

    def score(
        self,
        history: list[OracleResponse],
        candidates: list[str],
    ) -> list[float]:
        ...


class RandomAcquisition:
    """Uniform random — the no-skill baseline."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def score(
        self, history: list[OracleResponse], candidates: list[str]
    ) -> list[float]:
        return [self._rng.random() for _ in candidates]


class GreedyAcquisition:
    """Pick the highest-known-score candidate next; myopic exploitation.

    For unprobed candidates we score them at 0.0 (worst-known) so the
    loop prefers *already-probed* candidates with higher scores. In a
    Round 4 + predictor setting this would be replaced by an
    embedding-based predictor.
    """

    def score(
        self, history: list[OracleResponse], candidates: list[str]
    ) -> list[float]:
        # index word → score for O(1) lookup
        seen: dict[str, float] = {
            r.word: (r.score if r.score is not None else float("-inf"))
            for r in history
        }
        return [seen.get(w, 0.0) for w in candidates]


class UncertaintyAcquisition:
    """Pick candidates that are *closest to a threshold*.

    For unprobed candidates we don't know their score — by default
    we set the prior to the threshold itself (max uncertainty).
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def score(
        self, history: list[OracleResponse], candidates: list[str]
    ) -> list[float]:
        seen: dict[str, float] = {
            r.word: (r.score if r.score is not None else float("-inf"))
            for r in history
        }
        out: list[float] = []
        for w in candidates:
            s = seen.get(w, self.threshold)
            if s == float("-inf"):
                s = self.threshold
            out.append(1.0 - abs(s - self.threshold))
        return out


# --- orchestrator ---------------------------------------------------------


def active_solve(
    *,
    oracle: Oracle,
    candidates: Iterable[str],
    acquisition: Acquisition,
    budget: int,
    history: list[OracleResponse] | None = None,
) -> list[OracleResponse]:
    """Run an active learning loop until correct or budget runs out.

    Each step:

    1. Score all unseen candidates with ``acquisition``.
    2. Pick the candidate with the highest acquisition score and
       call ``oracle.probe``.
    3. Stop early if the response has ``correct=True``.

    The optional ``history`` argument lets callers seed the loop
    with already-known OracleResponses (e.g. from a fitted
    embedding model) so the acquisition function has signal
    beyond cold-start zeros. Default: empty history.

    Returns the list of responses in probe order (the seed
    history is *not* included in the returned list — only the
    responses probed in *this* call).
    """
    seed = list(history or [])
    # `seed` is the ranker's prior — the *model's predicted scores* for
    # candidates we have *not yet probed in this session*. They affect
    # which candidate gets probed first via the acquisition function
    # but are NOT counted toward ``seen`` (we still probe) and NOT
    # included in the returned list.
    seen: set[str] = set()
    history_out: list[OracleResponse] = []
    pending: list[str] = list(dict.fromkeys(candidates))  # de-dup
    current_history = seed  # acquisition sees seed

    while pending and len(history_out) < budget:
        scores = acquisition.score(current_history, pending)
        assert len(scores) == len(pending)

        # Pick the highest-scored pending candidate.
        ranked = sorted(
            zip(pending, scores), key=lambda x: x[1], reverse=True
        )
        next_word = ranked[0][0]

        seen.add(next_word)
        pending = [w for w in pending if w != next_word]

        response = oracle.probe(next_word)
        history_out.append(response)
        current_history.append(response)
        logger.info(
            "probe %s → score=%s correct=%s",
            next_word,
            response.score,
            response.correct,
        )

        if response.correct:
            logger.info("solved at %s — stopping", next_word)
            break

    return history_out
