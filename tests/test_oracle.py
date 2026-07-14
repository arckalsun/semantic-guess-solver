"""Tests for ``sgs.oracle`` — OracleResponse + FakeOracle.

Reference case: case-5 (shareId 376634286041, answer=萧山), the cleanest
3-phase trajectory in the case study.
"""

from __future__ import annotations

import json

import pytest

from sgs.oracle import FakeOracle, Oracle, OracleResponse


# --- OracleResponse --------------------------------------------------


def test_oracle_response_to_ndjson_uses_camel_case_keys() -> None:
    """Wire format is camelCase, snake_case Python fields."""
    r = OracleResponse(
        word="忍者", score=0.989, correct=True,
        double_score=True, rate_limited=False,
    )
    d = r.to_ndjson()
    assert d == {
        "word": "忍者",
        "score": 0.989,
        "correct": True,
        "doubleScore": True,
        "rateLimited": False,
    }
    # Round-trips through JSON without surprises.
    s = json.dumps(d)
    assert "doubleScore" in s
    assert "rateLimited" in s


def test_oracle_response_is_frozen() -> None:
    r = OracleResponse(word="x", score=0.5)
    with pytest.raises((AttributeError, Exception)):
        r.score = 0.9  # type: ignore[misc]


# --- FakeOracle -------------------------------------------------------


def test_fake_oracle_returns_scripted_response() -> None:
    fo = FakeOracle({"忍者": OracleResponse(word="忍者", score=0.989,
                                            correct=True, double_score=True)})
    r = fo.probe("忍者")
    assert r.score == pytest.approx(0.989)
    assert r.correct is True
    assert r.double_score is True


def test_fake_oracle_falls_back_to_default_score() -> None:
    fo = FakeOracle(default_score=0.0)
    r = fo.probe("不在脚本里的词")
    assert r.word == "不在脚本里的词"
    assert r.score == 0.0
    assert r.correct is False


def test_fake_oracle_records_every_call() -> None:
    fo = FakeOracle()
    for w in ["忍者", "武士", "剑客", "忍者"]:  # 忍者 probed twice
        fo.probe(w)
    assert fo.calls == ["忍者", "武士", "剑客", "忍者"]
    assert fo.call_count == 4


def test_fake_oracle_learn_injects_fresh_response() -> None:
    fo = FakeOracle()
    fo.learn("萧山", 0.989, correct=True, double_score=True)
    r = fo.probe("萧山")
    assert r.correct is True
    assert r.double_score is True


def test_fake_oracle_is_protocol_compatible() -> None:
    """Anything we use as an Oracle must satisfy the runtime protocol."""
    fo = FakeOracle()
    assert isinstance(fo, Oracle)  # runtime_checkable Protocol


def test_fake_oracle_close_is_noop() -> None:
    fo = FakeOracle()
    assert fo.close() is None