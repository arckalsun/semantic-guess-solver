"""Tests for :mod:`sgs.daily_solve` — the visitor-accessible daily mode.

Discovered 2026-07-15: ``/api/v0/quiz/daily/GuessWord/guessV1?date=YYYYMMDD``
is unauthenticated. This test suite covers:

1. ``DailyOracle.probe`` URL shape (no shareId, includes date).
2. ``DailyProbeResult`` schema — every field the ranker needs.
3. ``main()`` flow: seed sweep → centroid → KRR plateau escape.
4. NDJSON round-trip: results persist + reload.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from sgs.daily_solve import (
    DailyOracle,
    DailyProbeResult,
    _load_observations,
    _record,
    _seed_sweep,
    main,
)


# ----- DailyOracle URL construction (smoke test, no network) -----


def test_daily_oracle_url_shape() -> None:
    """Daily probe URL uses date= not shareId=."""
    captured: dict[str, str] = {}

    def fake_open(req, *args, **kwargs):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.headers)

        class FakeResp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self):
                return json.dumps({
                    "success": True,
                    "data": {"score": 10000, "doubleScore": 0.95, "correct": False},
                }).encode("utf-8")
        return FakeResp()

    oracle = DailyOracle(date="20260715")
    with mock.patch("urllib.request.urlopen", fake_open):
        result = oracle.probe("南宁")
    assert "date=20260715" in captured["url"]
    assert "shareId=" not in captured["url"]
    assert "skipBusinessErrorToast=true" in captured["url"]
    # urllib normalises header names (Fun-device, User-agent, Accept)
    # — check case-insensitively.
    headers_lower = {k.lower(): v for k, v in captured["headers"].items()}
    assert "fun-device" in headers_lower
    assert headers_lower["fun-device"] == "web"
    assert "referer" in headers_lower


def test_daily_oracle_parses_correct_true() -> None:
    """correct=true is surfaced through DailyProbeResult."""

    def fake_open(req, *a, **kw):
        class R:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self):
                return json.dumps({
                    "success": True,
                    "data": {"score": 10000, "doubleScore": 1.0, "correct": True},
                }).encode()
        return R()

    oracle = DailyOracle(date="20260715")
    with mock.patch("urllib.request.urlopen", fake_open):
        result = oracle.probe("南宁")
    assert result.correct is True
    assert result.score == 1.0


def test_daily_oracle_handles_rate_limit() -> None:
    """Rate-limit response captured as err_code, score=None."""

    def fake_open(req, *a, **kw):
        class R:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self):
                return json.dumps({
                    "success": False,
                    "data": None,
                    "errorCode": "rate_limit_exceed",
                    "errorMessage": "访问过快, 请稍后访问",
                }).encode()
        return R()

    oracle = DailyOracle(date="20260715")
    with mock.patch("urllib.request.urlopen", fake_open):
        result = oracle.probe("南宁")
    assert result.score is None
    assert result.correct is False
    assert result.err_code == "rate_limit_exceed"


def test_daily_oracle_handles_network_exception() -> None:
    """Network error captured cleanly — solver continues."""
    oracle = DailyOracle(date="20260715")
    with mock.patch("urllib.request.urlopen", side_effect=ConnectionError("timeout")):
        result = oracle.probe("南宁")
    assert result.score is None
    assert result.err_code == "NET"
    assert result.correct is False


# ----- DailyProbeResult schema -----


def test_probe_result_to_ndjson_round_trips() -> None:
    """NDJSON output preserves all fields the ranker needs."""
    r = DailyProbeResult(
        word="南宁", score=1.0, correct=True,
        raw_score=10000, err_code=None, err_msg=None,
    )
    d = r.to_ndjson()
    # Required keys
    for k in ("word", "score", "correct", "doubleScore"):
        assert k in d
    assert d["word"] == "南宁"
    assert d["score"] == 1.0
    assert d["correct"] is True
    # Roundtrip via JSON
    blob = json.dumps(d, ensure_ascii=False)
    d2 = json.loads(blob)
    assert d2 == d


# ----- _seed_sweep cluster coverage -----


def test_seed_sweep_includes_high_frequency_clusters() -> None:
    """Seed list should bias toward cities/places/foods — the clusters
    that historically give BGE-zh-base high cosine to today's answer
    (see case-daily-2026-07-15: top 30 of first round all cities)."""
    seed = _seed_sweep()
    assert isinstance(seed, list)
    assert len(seed) >= 25  # at least 25 seed words
    # Cities should dominate (>=10)
    city_keywords = ("广州", "上海", "北京", "成都", "西安", "武汉")
    city_count = sum(1 for w in seed if w in city_keywords)
    assert city_count >= 5


# ----- _load_observations — corpus filter -----


def test_load_observations_drops_words_not_in_corpus(tmp_path: Path) -> None:
    """Words outside the candidate corpus are silently filtered out
    — they came from the seed sweep and can never be ranked."""
    log = tmp_path / "replay.ndjson"
    log.write_text(
        json.dumps({"word": "教育", "score": 0.5}) + "\n"  # 教育 not in corpus
        + json.dumps({"word": "南宁", "score": 0.95}) + "\n"
        + json.dumps({"word": "上海", "score": None}) + "\n"  # rate-limited, score None
    )
    corpus = {"南宁", "上海", "学校"}
    obs = _load_observations(log, corpus)
    assert obs == [("南宁", 0.95)]


def test_load_observations_empty_file_returns_empty_list(tmp_path: Path) -> None:
    log = tmp_path / "replay.ndjson"
    log.write_text("")
    assert _load_observations(log, set()) == []


def test_load_observations_missing_file_returns_empty_list(tmp_path: Path) -> None:
    log = tmp_path / "nonexistent.ndjson"
    assert _load_observations(log, set()) == []


# ----- _record appends correctly -----


def test_record_appends_to_log(tmp_path: Path) -> None:
    log = tmp_path / "replay.ndjson"
    _record(
        DailyProbeResult(word="南宁", score=0.95),
        log,
    )
    lines = log.read_text().splitlines()
    assert len(lines) == 1
    d = json.loads(lines[0])
    assert d["word"] == "南宁"
    assert d["score"] == 0.95


# ----- main() integration — minimal stub corpus -----


def _make_stub_corpus(tmp_path: Path, words: list[str], dim: int = 16):
    """Helper: write a tiny words.json + emb.npy for a faux corpus."""
    import numpy as np

    json_path = tmp_path / "cand.json"
    emb_path = tmp_path / "emb.npy"
    json_path.write_text(json.dumps(words, ensure_ascii=False))
    # Random unit-norm 16-dim embeddings
    emb = np.random.RandomState(42).randn(len(words), dim).astype("float32")
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    np.save(emb_path, emb)
    return json_path, emb_path


def test_main_finds_correct_word_via_seed(tmp_path: Path, capsys) -> None:
    """main() must surface the first correct=true probe and exit 0."""
    words = ["南宁", "广州", "上海", "北京", "成都", "学校", "医院",
             "火车", "水果", "海洋", "通过", "继续"]
    json_path, emb_path = _make_stub_corpus(tmp_path, words)
    out_path = tmp_path / "replay.ndjson"
    # Put 南宁 first in seed by passing custom seed
    seed = ",".join(words)
    # Stub: every probe returns correct=True on the second word — no,
    # simpler: stub all to fail, except 南宁 which returns correct=True
    seen = []

    def fake_probe(self, word):
        # Simulate the daily-2026-07-15 ranking: cities all score high,
        # 南宁 is the correct answer.
        if word == "南宁":
            return DailyProbeResult(word=word, score=1.0, correct=True,
                                     raw_score=10000)
        if word in {"广州", "上海"}:
            return DailyProbeResult(word=word, score=0.7 + len(seen)*0.01,
                                     raw_score=10000)
        return DailyProbeResult(word=word, score=0.3, raw_score=10000)

    with mock.patch.object(DailyOracle, "probe", new=fake_probe):
        rc = main([
            "--date", "20260715",
            "--candidates", str(json_path),
            "--embeddings", str(emb_path),
            "--out", str(out_path),
            "--seed", seed,
            "--batch-size", "5",
            "--rounds", "3",
        ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "CORRECT" in out
    assert "南宁" in out


def test_main_resumes_from_existing_log(tmp_path: Path, capsys) -> None:
    """If replay file already has observations, skip the seed sweep."""
    words = ["南宁", "广州", "上海", "北京", "成都"]
    json_path, emb_path = _make_stub_corpus(tmp_path, words)
    out_path = tmp_path / "replay.ndjson"
    # Pre-populate with one valid observation
    out_path.write_text(json.dumps({"word": "广州", "score": 0.83}) + "\n")

    probe_count = [0]

    def fake_probe(self, word):
        probe_count[0] += 1
        if word == "南宁":
            return DailyProbeResult(word=word, score=1.0, correct=True)
        return DailyProbeResult(word=word, score=0.5)

    with mock.patch.object(DailyOracle, "probe", new=fake_probe):
        rc = main([
            "--date", "20260715",
            "--candidates", str(json_path),
            "--embeddings", str(emb_path),
            "--out", str(out_path),
            "--batch-size", "5",
            "--rounds", "1",
        ])
    # Should still find 南宁
    assert rc == 0
    # Should NOT do the seed sweep — would be wasteful, takes 30+ probes
    # (the corpus is small so KRR/nearest-neighbor should home in)
    assert probe_count[0] <= 5


def test_main_returns_nonzero_when_unable_to_solve(tmp_path: Path, capsys) -> None:
    """No correct=true reachable → main returns 1 with diagnostic top-1."""
    words = ["南宁", "广州", "上海"]
    json_path, emb_path = _make_stub_corpus(tmp_path, words)

    def fake_probe(self, word):
        # 0.5 plateau — never correct
        return DailyProbeResult(word=word, score=0.5)

    out_path = tmp_path / "replay.ndjson"
    with mock.patch.object(DailyOracle, "probe", new=fake_probe):
        rc = main([
            "--date", "20260715",
            "--candidates", str(json_path),
            "--embeddings", str(emb_path),
            "--out", str(out_path),
            "--batch-size", "3",
            "--rounds", "1",
            "--seed", "广州,上海,南宁",
        ])
    assert rc == 1
    out = capsys.readouterr().out
    assert "Best score" in out
    assert "南宁" in out  # top-1 reported
