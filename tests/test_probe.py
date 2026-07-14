"""Tests for ``sgs.probe`` — batch probing + NDJSON append + stop condition.

All tests use a :class:`FakeOracle` — no network. The point is to validate
the *behaviour* of the probe loop (rate-limit accounting, stop-on-correct,
NDJSON append), independent of any real HTTP stack.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sgs.oracle import FakeOracle, OracleResponse
from sgs.probe import (
    already_correct,
    probe_and_record,
    probe_batch,
)
from sgs.ratelimit import TokenBucket


# --- probe_batch ------------------------------------------------------


def test_empty_batch_returns_empty_result() -> None:
    r = probe_batch([], FakeOracle(), bucket=TokenBucket(rate=1000, burst=10))
    assert r.responses == []
    assert r.hit is None
    assert r.rate_limited == 0


def test_batch_probes_each_word_in_order() -> None:
    fo = FakeOracle()
    r = probe_batch(
        ["忍者", "武士", "剑客"], fo,
        bucket=TokenBucket(rate=1000, burst=10),
    )
    assert fo.calls == ["忍者", "武士", "剑客"]
    assert [resp.word for resp in r.responses] == ["忍者", "武士", "剑客"]


def test_batch_stops_on_first_correct(monkeypatch) -> None:
    """Server lock: once an answer is found, don't waste tokens on the rest."""
    fo = FakeOracle({"萧山": OracleResponse("萧山", 0.989,
                                             correct=True, double_score=True)})
    r = probe_batch(
        ["上海", "浙江", "萧山", "南京", "杭州"],
        fo,
        bucket=TokenBucket(rate=1000, burst=10),
    )
    # Stops at 萧山 — the next two are never probed.
    assert [resp.word for resp in r.responses] == ["上海", "浙江", "萧山"]
    assert r.hit is not None
    assert r.hit.word == "萧山"
    assert r.hit.correct is True


def test_batch_continue_on_correct_when_disabled() -> None:
    """Auditing mode: --no-stop-on-correct keeps the whole batch."""
    fo = FakeOracle({"萧山": OracleResponse("萧山", 0.989, correct=True)})
    r = probe_batch(
        ["上海", "萧山", "南京"],
        fo,
        bucket=TokenBucket(rate=1000, burst=10),
        stop_on_correct=False,
    )
    assert len(r.responses) == 3
    assert r.hit is not None  # still records the hit


def test_batch_counts_rate_limited() -> None:
    fo = FakeOracle({
        "A": OracleResponse("A", 0.0, rate_limited=True),
        "B": OracleResponse("B", 0.5),
        "C": OracleResponse("C", 0.0, rate_limited=True),
    })
    r = probe_batch(
        ["A", "B", "C", "D"], fo,
        bucket=TokenBucket(rate=1000, burst=10),
    )
    assert r.rate_limited == 2


def test_batch_hit_none_when_nothing_correct() -> None:
    fo = FakeOracle(default_score=0.4)
    r = probe_batch(["x", "y", "z"], fo,
                    bucket=TokenBucket(rate=1000, burst=10))
    assert r.hit is None
    assert len(r.responses) == 3


# --- probe_and_record ------------------------------------------------


def test_probe_and_record_appends_ndjson(tmp_path: Path) -> None:
    fo = FakeOracle({"萧山": OracleResponse("萧山", 0.989,
                                            correct=True, double_score=True)})
    nd = tmp_path / "session.ndjson"
    r = probe_and_record(
        ["上海", "浙江", "萧山"], fo, nd,
        bucket=TokenBucket(rate=1000, burst=10),
    )
    assert r.hit is not None
    # File exists, is NDJSON, one record per response.
    lines = [json.loads(l) for l in nd.read_text().splitlines() if l.strip()]
    assert [d["word"] for d in lines] == ["上海", "浙江", "萧山"]
    assert lines[-1]["correct"] is True
    assert lines[-1]["doubleScore"] is True


def test_probe_and_record_preserves_existing_content(tmp_path: Path) -> None:
    """Round-1 and Round-2 both write to the same replay file."""
    nd = tmp_path / "session.ndjson"
    nd.write_text(json.dumps({"word": "phase1", "score": 0.5,
                              "correct": False, "doubleScore": False,
                              "rateLimited": False}) + "\n")
    fo = FakeOracle()
    probe_and_record(["phase2"], fo, nd,
                     bucket=TokenBucket(rate=1000, burst=10))
    lines = [json.loads(l) for l in nd.read_text().splitlines() if l.strip()]
    assert [d["word"] for d in lines] == ["phase1", "phase2"]


def test_probe_and_record_noop_on_empty_batch(tmp_path: Path) -> None:
    nd = tmp_path / "session.ndjson"
    fo = FakeOracle()
    r = probe_and_record([], fo, nd,
                         bucket=TokenBucket(rate=1000, burst=10))
    assert r.responses == []
    assert not nd.exists()  # nothing to write


# --- already_correct -------------------------------------------------


def test_already_correct_collects_correct_words() -> None:
    responses = [
        OracleResponse("A", 0.5),
        OracleResponse("B", 0.989, correct=True),
        OracleResponse("C", 0.7),
        OracleResponse("D", 0.989, correct=True, double_score=True),
    ]
    assert already_correct(responses) == {"B", "D"}


def test_already_correct_empty_when_none_correct() -> None:
    assert already_correct([]) == set()
    assert already_correct([OracleResponse("X", 0.0)]) == set()