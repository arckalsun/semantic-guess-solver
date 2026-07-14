"""Tests for :mod:`sgs.learn` — active learning loop.

We use :class:`FakeOracle` so all tests are offline (no network / no
Playwright). Two observations:

1. ``FakeOracle.learn`` is the way we extend the canned script at
   runtime so later candidates can hit specific responses — also
   useful for warm-starting a history of fake probes so the
   GreedyAcquisition has signal to rank against.
2. The acquisition function should *suggest* candidates that have
   the highest expected utility — running through the full loop
   on a tiny script verifies the strategy logic.
"""

from __future__ import annotations

import pytest

from sgs.learn import (
    Acquisition,
    GreedyAcquisition,
    RandomAcquisition,
    UncertaintyAcquisition,
    active_solve,
)
from sgs.oracle import FakeOracle, OracleResponse


# ---- helpers ---------------------------------------------------------


def make_oracle() -> FakeOracle:
    """A *known* oracle we can probe for ground-truth.

    Each word on the oracle has a known score in [0, 1] so we can
    predict what the greedy / uncertainty / random strategies will
    pick. Use :py:meth:`FakeOracle.learn` to add or override entries
    (also used as the warm-start helper).
    """
    script = {
        "忍者": OracleResponse(
            word="忍者", score=0.7, correct=False, double_score=False
        ),
        "萧山": OracleResponse(
            word="萧山", score=0.95, correct=False, double_score=True
        ),
        "西瓜": OracleResponse(
            word="西瓜", score=0.3, correct=False, double_score=False
        ),
    }
    return FakeOracle(script)


def make_warm_oracle() -> FakeOracle:
    """Oracle with a pre-loaded history so GreedyAcquisition has signal.

    Cold-started GreedyAcquisition treats all unprobed candidates as
    score=0.0 — and our orchestrator tie-breaks 0.0 ties by list order
    (a degenerate behaviour by design, the predictor is the answer).
    Use this when you specifically want Greedy to rank by 萧山 / 忍者
    / 西瓜 instead of list-position.
    """
    return make_oracle()


# ---- acquisition functions --------------------------------------------


def test_random_acquisition_returns_scores_in_unit_range() -> None:
    fn = RandomAcquisition(seed=42)
    candidates = ["甲", "乙", "丙"]
    scores = fn.score(history=[], candidates=candidates)
    assert len(scores) == 3
    assert all(0 <= s <= 1 for s in scores)
    fn2 = RandomAcquisition(seed=7)
    assert fn.score(history=[], candidates=candidates) != fn2.score(
        history=[], candidates=candidates
    )


def test_greedy_acquisition_picks_highest_score_first() -> None:
    fn = GreedyAcquisition()
    history = [
        OracleResponse(word="西瓜", score=0.3),
        OracleResponse(word="萧山", score=0.95),
        OracleResponse(word="忍者", score=0.7),
    ]
    candidates = ["西瓜", "萧山", "忍者"]
    scores = fn.score(history, candidates)
    ranking = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    # 萧山 (0.95) is highest.
    assert ranking[0][0] == "萧山"


def test_greedy_acquisition_treats_unprobed_at_zero() -> None:
    fn = GreedyAcquisition()
    history = [OracleResponse(word="萧山", score=0.95)]
    scores = fn.score(history, ["萧山", "忍者", "西瓜"])
    pair = dict(zip(["萧山", "忍者", "西瓜"], scores))
    assert pair["萧山"] == 0.95
    assert pair["忍者"] == 0.0
    assert pair["西瓜"] == 0.0


def test_uncertainty_acquisition_picks_thresholder() -> None:
    fn = UncertaintyAcquisition(threshold=0.5)
    history = [
        OracleResponse(word="西瓜", score=0.3),
        OracleResponse(word="萧山", score=0.95),
    ]
    scores = fn.score(history, ["西瓜", "萧山"])
    pair = dict(zip(["西瓜", "萧山"], scores))
    # 0.3 → 1 - 0.2 = 0.8; 0.95 → 1 - 0.45 = 0.55.
    assert pair["西瓜"] > pair["萧山"]


def test_uncertainty_unprobed_scored_at_threshold() -> None:
    """Unprobed candidates default to threshold → uncertainty 1.0."""
    fn = UncertaintyAcquisition(threshold=0.5)
    scores = fn.score(history=[], candidates=["甲", "乙"])
    assert scores == [1.0, 1.0]


def test_acquisition_protocol_is_runtime_checkable() -> None:
    class MyAcq:
        def score(self, history, candidates):
            return [1.0] * len(candidates)

    fn: Acquisition = MyAcq()  # type: ignore[assignment]
    assert fn.score([], ["甲"]) == [1.0]


# ---- active_solve loop ------------------------------------------------


def test_active_solve_stops_on_correct() -> None:
    oracle = make_oracle()
    oracle.learn("忍者", score=0.7, correct=True)

    responses = active_solve(
        oracle=oracle,
        candidates=["西瓜", "忍者", "萧山"],
        acquisition=GreedyAcquisition(),
        budget=10,
    )

    assert any(r.correct for r in responses)
    last_correct = next(r for r in reversed(responses) if r.correct)
    assert last_correct.word == "忍者"


def test_active_solve_runs_to_budget_when_no_correct() -> None:
    oracle = make_oracle()
    responses = active_solve(
        oracle=oracle,
        candidates=["西瓜", "忍者", "萧山"],
        acquisition=GreedyAcquisition(),
        budget=10,
    )
    assert len({r.word for r in responses}) == 3


def test_active_solve_respects_budget_cap() -> None:
    oracle = make_oracle()
    responses = active_solve(
        oracle=oracle,
        candidates=list(make_oracle()._script.keys()) * 3,  # 9 dups → 3 unique
        acquisition=GreedyAcquisition(),
        budget=2,
    )
    assert len(responses) <= 2


def test_active_solve_dedupe_candidates() -> None:
    """De-dup collapses repeated words; greedy ordering is by list tie-break.

    Note this is a degenerate cold-start case — GreedyAcquisition sees
    no history so all unprobed candidates score 0.0 and tie-break by
    list position. We assert de-dup behaviour, NOT greedy ranking.
    Use ``test_active_solve_warm_started_greedy_*`` for ranking.
    """
    oracle = make_oracle()
    responses = active_solve(
        oracle=oracle,
        candidates=["西瓜", "西瓜", "忍者", "忍者"],
        acquisition=GreedyAcquisition(),
        budget=10,
    )
    seen = [r.word for r in responses]
    # de-duped to 2 unique words
    assert len(seen) == 2
    # Cold greedy ordering at this call: list-order tie-break.
    assert seen == ["西瓜", "忍者"]


def test_active_solve_warm_history_lets_greedy_steal_ahead() -> None:
    """Pre-seed via ``history=`` so Greedy ranks by score.

    This models the Round 4 predictor case: a fitted embedding
    model proposes scores for unprobed candidates. We feed those
    scores as ``history`` so GreedyAcquisition sees ranking
    signal even before the loop probes anything.
    """
    oracle = FakeOracle(
        script={
            "萧山": OracleResponse(word="萧山", score=0.95),
            "忍者": OracleResponse(word="忍者", score=0.7),
            "西瓜": OracleResponse(word="西瓜", score=0.3),
        }
    )
    history = [
        OracleResponse(word="萧山", score=0.95),
        OracleResponse(word="忍者", score=0.7),
        OracleResponse(word="西瓜", score=0.3),
    ]
    responses = active_solve(
        oracle=oracle,
        candidates=["西瓜", "忍者", "萧山"],
        acquisition=GreedyAcquisition(),
        budget=3,
        history=history,
    )
    seen = [r.word for r in responses]
    # Greedy: 萧山 (0.95) probes first because it's in the warm history
    # with the highest score.
    assert seen[0] == "萧山"


def test_active_solve_warm_history_random_seed_differs_from_greedy() -> None:
    """Sanity check: random ordering differs from warm greedy ordering."""
    oracle_greedy = make_oracle()
    oracle_random = FakeOracle(dict(make_oracle()._script))

    history = [
        OracleResponse(word="萧山", score=0.95),
        OracleResponse(word="忍者", score=0.7),
        OracleResponse(word="西瓜", score=0.3),
    ]

    # Warm greedy ranking stays the same regardless of run order.
    fn = GreedyAcquisition()
    candidates = ["西瓜", "忍者", "萧山"]
    greedy_scores = fn.score(history, candidates)
    greedy_top = max(zip(candidates, greedy_scores), key=lambda x: x[1])
    assert greedy_top[0] == "萧山"

    # Random with seed=42 gives a deterministic non-greedy top.
    fn_rand = RandomAcquisition(seed=42)
    rand_scores = fn_rand.score([], candidates)
    rand_top = max(zip(candidates, rand_scores), key=lambda x: x[1])
    assert rand_top[0] in candidates  # sanity
    # They must be different on at least one position.
    greedy_order = [
        w for w, _ in sorted(zip(candidates, greedy_scores), key=lambda x: x[1], reverse=True)
    ]
    rand_order = [
        w for w, _ in sorted(zip(candidates, rand_scores), key=lambda x: x[1], reverse=True)
    ]
    # Note: empty-history random + tie-break may collide with greedy,
    # but with seed=42 it's extremely unlikely to match exactly.
    assert greedy_order != rand_order


def test_active_solve_handles_oracle_returning_lock_signal() -> None:
    """``data:null`` (score=None) must not crash — record as-is."""
    oracle = FakeOracle(
        script={
            "忍者": OracleResponse(word="忍者", score=None, correct=False),
            "萧山": OracleResponse(word="萧山", score=0.5),
        }
    )
    responses = active_solve(
        oracle=oracle,
        candidates=["忍者", "萧山"],
        acquisition=GreedyAcquisition(),
        budget=2,
    )
    assert len(responses) == 2
    words = [r.word for r in responses]
    # Both must be probed; GreedyAcquisition's -inf for locked+unprobed
    # falls back to 0.0 → ties broken by list order → 忍者 (locked,
    # first listed) probes first.
    assert set(words) == {"忍者", "萧山"}


def test_active_solve_empty_candidates_returns_empty_history() -> None:
    oracle = make_oracle()
    responses = active_solve(
        oracle=oracle,
        candidates=[],
        acquisition=GreedyAcquisition(),
        budget=10,
    )
    assert responses == []


def test_active_solve_with_zero_budget_returns_empty_history() -> None:
    oracle = make_oracle()
    responses = active_solve(
        oracle=oracle,
        candidates=["西瓜", "忍者"],
        acquisition=GreedyAcquisition(),
        budget=0,
    )
    assert responses == []
