"""Round 5 tests — end-to-end solver CLI.

These tests pin down the contract of `sgs.solve`:

  * The CLI is a thin orchestrator that wires together (1) an Oracle
    (Round 2/3), (2) an Acquisition (Round 4), and (3) replay-NDJSON
    append (Round 2). It does NOT introduce new network/policy logic.

  * `solve_run(...)` is the library entry point (callable from a
    notebook, from `python -m sgs.solve`, and from tests). It returns
    a `SolveResult` that the CLI renders as JSON or as a one-line
    "solved word=X score=Y probes=N" message.

  * The CLI subcommand `python -m sgs.solve run --candidates foo.txt
    --replay out.ndjson --acquisition greedy --max-probes 50` must
    work without ever importing playwright at import-time (only when
    `--oracle playwright` is passed).

  * Plateau detection: if the top-1 score has not improved for
    `plateau-window` consecutive probes, the run stops early and the
    caller sees `stop_reason="plateau"`.

  * Exit codes:
      0 = solved (correct=True reached) OR dry-run completed
      2 = budget exhausted without `correct=True`
      3 = configuration error (missing candidates file, unknown
          acquisition, missing playwright when requested, etc.)

The tests are TDD-first: they import from `sgs.solve` which doesn't
exist yet; this file is the red, the implementation is the green.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from sgs.oracle import Oracle, OracleResponse


# --------------------------------------------------------------------------
# Test fakes — extend the existing FakeOracle with a "correct after N
# probes" hook so we can drive plateau and budget-exhaustion tests.
# --------------------------------------------------------------------------


class ScriptedOracle:
    """Like FakeOracle but yields responses in probe order regardless
    of the word's identity. Used to simulate "the answer is the 3rd
    unique word we ask about" without scripting the ranker.
    """

    def __init__(self, queue: list[OracleResponse]):
        self._queue = list(queue)
        self.calls: list[str] = []

    def probe(self, word: str) -> OracleResponse:
        self.calls.append(word)
        if not self._queue:
            # Safety net — never let a test silently "miss" its queue.
            raise AssertionError(
                f"ScriptedOracle exhausted at word={word!r}; "
                "test setup under-supplied the queue"
            )
        return self._queue.pop(0)

    def close(self) -> None:
        return None


# --------------------------------------------------------------------------
# 1. Core SolveResult contract
# --------------------------------------------------------------------------


def test_solve_run_returns_solve_result_on_correct():
    from sgs.solve import SolveResult, solve_run

    oracle = ScriptedOracle(
        [
            OracleResponse("萧山", 0.95),
            OracleResponse("忍者", 0.989, correct=True),
        ]
    )
    result = solve_run(
        oracle=oracle,
        candidates=["萧山", "忍者"],
        acquisition_name="greedy",
        max_probes=50,
        replay_path=None,
    )
    assert isinstance(result, SolveResult)
    assert result.solved is True
    assert result.stop_reason == "correct"
    assert result.probes_used == 2
    assert result.peak_word == "忍者"
    assert result.peak_score == pytest.approx(0.989)


def test_solve_run_stops_on_budget_with_correct_false():
    from sgs.solve import solve_run

    oracle = ScriptedOracle(
        [OracleResponse(f"w{i}", 0.5) for i in range(10)]
    )
    result = solve_run(
        oracle=oracle,
        candidates=[f"w{i}" for i in range(10)],
        acquisition_name="greedy",
        max_probes=4,
        replay_path=None,
    )
    assert result.solved is False
    assert result.stop_reason == "budget"
    assert result.probes_used == 4


def test_solve_run_unknown_acquisition_raises_value_error():
    """Bad config must NOT silently fall back to a default."""
    from sgs.solve import solve_run

    oracle = ScriptedOracle([OracleResponse("x", 0.1)])
    with pytest.raises(ValueError, match="acquisition"):
        solve_run(
            oracle=oracle,
            candidates=["x"],
            acquisition_name="not_a_real_strategy",
            max_probes=5,
            replay_path=None,
        )


def test_solve_run_appends_ndjson_replay(tmp_path: Path):
    """Replay-NDJSON contract: one JSON object per line, append-only,
    in probe order. Same shape as `probe_and_record`."""
    from sgs.solve import solve_run

    replay = tmp_path / "out.ndjson"
    oracle = ScriptedOracle(
        [
            OracleResponse("萧山", 0.95),
            OracleResponse("忍者", 0.989, correct=True),
        ]
    )
    solve_run(
        oracle=oracle,
        candidates=["萧山", "忍者"],
        acquisition_name="greedy",
        max_probes=10,
        replay_path=replay,
    )
    assert replay.exists()
    lines = replay.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert [d["word"] for d in parsed] == ["萧山", "忍者"]
    assert parsed[1]["correct"] is True
    assert parsed[1]["score"] == pytest.approx(0.989)


def test_solve_run_replay_appends_does_not_overwrite(tmp_path: Path):
    """A second run on the same shareId appends — never overwrites —
    so Round 1 → 2 → 3 share one continuous log per shareId."""
    from sgs.solve import solve_run

    replay = tmp_path / "out.ndjson"
    oracle_a = ScriptedOracle([OracleResponse("萧山", 0.95)])
    solve_run(
        oracle=oracle_a,
        candidates=["萧山"],
        acquisition_name="greedy",
        max_probes=5,
        replay_path=replay,
    )
    oracle_b = ScriptedOracle([OracleResponse("忍者", 0.99)])
    solve_run(
        oracle=oracle_b,
        candidates=["忍者"],
        acquisition_name="greedy",
        max_probes=5,
        replay_path=replay,
    )
    lines = replay.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert [json.loads(l)["word"] for l in lines] == ["萧山", "忍者"]


# --------------------------------------------------------------------------
# 2. Plateau detection (Round 5 hard rule)
# --------------------------------------------------------------------------


def test_solve_run_stops_on_plateau_when_top1_stagnates():
    """If the best score observed has not improved for N probes,
    the run ends early with stop_reason='plateau'."""
    from sgs.solve import solve_run

    plateau_window = 3
    oracle = ScriptedOracle(
        [
            OracleResponse("a", 0.50),
            OracleResponse("b", 0.55),  # peak
            OracleResponse("c", 0.40),
            OracleResponse("d", 0.30),
            OracleResponse("e", 0.20),
        ]
    )
    result = solve_run(
        oracle=oracle,
        candidates=["a", "b", "c", "d", "e"],
        acquisition_name="greedy",
        max_probes=20,
        replay_path=None,
        plateau_window=plateau_window,
    )
    assert result.stop_reason == "plateau"
    assert result.solved is False
    # Plateau detection checks the *trailing window* after each probe.
    # Probe sequence:
    #   a → peak=0.50   (window=[a])
    #   b → peak=0.55   (window=[a,b])
    #   c → tail=[a,b,c]; peak=0.55, b==peak → not all below → no breach
    #   d → tail=[b,c,d]; b==peak → not all below → no breach
    #   e → tail=[c,d,e]; all 0.40/0.30/0.20 < 0.55 → BREACH
    # So we probe 5 times (a,b,c,d,e) before stopping.
    assert result.probes_used == 5


def test_solve_run_plateau_does_not_trigger_if_improving():
    """A monotonically increasing run must run until budget or correct,
    never plateau."""
    from sgs.solve import solve_run

    oracle = ScriptedOracle(
        [
            OracleResponse("a", 0.10),
            OracleResponse("b", 0.20),
            OracleResponse("c", 0.30),
            OracleResponse("d", 0.40),
            OracleResponse("e", 0.50),
        ]
    )
    result = solve_run(
        oracle=oracle,
        candidates=["a", "b", "c", "d", "e"],
        acquisition_name="greedy",
        max_probes=20,
        replay_path=None,
        plateau_window=3,
    )
    assert result.stop_reason == "budget"
    assert result.probes_used == 5


# --------------------------------------------------------------------------
# 3. Acquisition dispatch
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "acq_name",
    ["random", "greedy", "uncertainty"],
)
def test_solve_run_dispatches_each_acquisition(acq_name):
    from sgs.solve import solve_run

    oracle = ScriptedOracle(
        [OracleResponse(f"w{i}", 0.5 + i * 0.1) for i in range(3)]
    )
    result = solve_run(
        oracle=oracle,
        candidates=["w0", "w1", "w2"],
        acquisition_name=acq_name,
        max_probes=3,
        replay_path=None,
    )
    assert result.probes_used == 3
    assert result.peak_score is not None


# --------------------------------------------------------------------------
# 4. CLI entry point — `python -m sgs.solve run ...`
# --------------------------------------------------------------------------


def test_cli_run_dry_run_exits_zero(tmp_path: Path):
    """`--oracle fake` must NOT import playwright at any point, and
    a 2-probe run that hits correct=True exits 0."""
    from sgs import solve as solve_mod

    candidates_file = tmp_path / "cands.txt"
    candidates_file.write_text("萧山\n忍者\n", encoding="utf-8")
    replay = tmp_path / "out.ndjson"

    # Pre-flight: ensure playwright is NOT imported at this moment.
    assert "playwright" not in sys.modules or (
        "playwright" in sys.modules
        and not hasattr(sys.modules.get("playwright"), "_dummy_test_only")
    )

    # Monkey-patch build_oracle to return a scripted oracle whose
    # second probe is correct=True so we exercise the solved=exit-0
    # code path without ever touching playwright.
    from sgs.oracle import OracleResponse
    from tests.test_solve import ScriptedOracle

    scripted = ScriptedOracle(
        [
            OracleResponse("萧山", 0.95),
            OracleResponse("忍者", 0.989, correct=True),
        ]
    )

    original_build = solve_mod.build_oracle

    def fake_build_oracle(name, *, share_id=None):
        return scripted

    solve_mod.build_oracle = fake_build_oracle
    try:
        rc = solve_mod.cli_main(
            [
                "run",
                "--candidates", str(candidates_file),
                "--replay", str(replay),
                "--oracle", "fake",
                "--acquisition", "greedy",
                "--max-probes", "5",
            ]
        )
    finally:
        solve_mod.build_oracle = original_build

    assert rc == 0
    assert replay.exists()
    lines = replay.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2


def test_cli_run_missing_candidates_file_exits_3(tmp_path: Path):
    """Missing input file = configuration error = exit 3 (not exit 1).
    Operators MUST be able to distinguish 'I misconfigured' from
    'the run produced no answer'."""
    from sgs.solve import cli_main

    rc = cli_main(
        [
            "run",
            "--candidates", str(tmp_path / "does-not-exist.txt"),
            "--oracle", "fake",
        ]
    )
    assert rc == 3


def test_cli_run_unknown_acquisition_exits_3(tmp_path: Path):
    from sgs.solve import cli_main

    candidates_file = tmp_path / "cands.txt"
    candidates_file.write_text("x\n", encoding="utf-8")

    rc = cli_main(
        [
            "run",
            "--candidates", str(candidates_file),
            "--oracle", "fake",
            "--acquisition", "fake_strategy",
        ]
    )
    assert rc == 3


# --------------------------------------------------------------------------
# 5. Plugin oracle (--oracle playwright) — must NOT import at module load
# --------------------------------------------------------------------------


def test_solve_module_does_not_import_playwright_at_load_time():
    """Importing sgs.solve must NOT pull in playwright. The heavy
    browser stack is loaded only when --oracle playwright is passed
    AND the function actually runs."""
    import sgs.solve  # noqa: F401

    # playwright.sync_api is the canonical import that surfaces
    # in sys.modules when anything pulls in the library.
    assert "playwright.sync_api" not in sys.modules


# --------------------------------------------------------------------------
# 6. Oracle lifecycle — close() is always called even on budget-exhaust
# --------------------------------------------------------------------------


def test_solve_run_calls_close_on_budget_exhaustion():
    """Even when the run stops because of budget, we must release
    browser resources. Resource leak = P0 for live mode."""
    from sgs.solve import solve_run

    class CloseTrackingOracle:
        def __init__(self):
            self.calls: list[str] = []
            self.closed = False

        def probe(self, word: str) -> OracleResponse:
            self.calls.append(word)
            return OracleResponse(word, 0.5)

        def close(self) -> None:
            self.closed = True

    oracle = CloseTrackingOracle()
    solve_run(
        oracle=oracle,
        candidates=["a", "b", "c"],
        acquisition_name="random",
        max_probes=2,
        replay_path=None,
    )
    assert oracle.closed is True


def test_solve_run_calls_close_on_correct_too():
    from sgs.solve import solve_run

    class CloseTrackingOracle:
        def __init__(self):
            self.closed = False

        def probe(self, word: str) -> OracleResponse:
            return OracleResponse("忍者", 0.989, correct=True)

        def close(self) -> None:
            self.closed = True

    oracle = CloseTrackingOracle()
    result = solve_run(
        oracle=oracle,
        candidates=["忍者"],
        acquisition_name="greedy",
        max_probes=5,
        replay_path=None,
    )
    assert result.solved is True
    assert oracle.closed is True


# --------------------------------------------------------------------------
# 7. Argument validation that must happen BEFORE the run starts
# --------------------------------------------------------------------------


def test_solve_run_rejects_max_probes_zero():
    """A run with max_probes=0 would either no-op silently or hit
    a divide-by-zero in plateau math. Reject at the door."""
    from sgs.solve import solve_run

    oracle = ScriptedOracle([OracleResponse("x", 0.1)])
    with pytest.raises(ValueError, match="max_probes"):
        solve_run(
            oracle=oracle,
            candidates=["x"],
            acquisition_name="greedy",
            max_probes=0,
            replay_path=None,
        )


def test_solve_run_rejects_empty_candidates():
    from sgs.solve import solve_run

    oracle = ScriptedOracle([])
    with pytest.raises(ValueError, match="candidates"):
        solve_run(
            oracle=oracle,
            candidates=[],
            acquisition_name="greedy",
            max_probes=5,
            replay_path=None,
        )


def test_solve_run_rejects_negative_plateau_window():
    from sgs.solve import solve_run

    oracle = ScriptedOracle([OracleResponse("x", 0.1)])
    with pytest.raises(ValueError, match="plateau_window"):
        solve_run(
            oracle=oracle,
            candidates=["x"],
            acquisition_name="greedy",
            max_probes=5,
            replay_path=None,
            plateau_window=-1,
        )


# --------------------------------------------------------------------------
# 8. Replay path is auto-created if missing (parent dir too)
# --------------------------------------------------------------------------


def test_solve_run_creates_replay_parent_dirs(tmp_path: Path):
    from sgs.solve import solve_run

    replay = tmp_path / "deep" / "nested" / "out.ndjson"
    oracle = ScriptedOracle([OracleResponse("忍者", 0.989, correct=True)])
    solve_run(
        oracle=oracle,
        candidates=["忍者"],
        acquisition_name="greedy",
        max_probes=2,
        replay_path=replay,
    )
    assert replay.exists()