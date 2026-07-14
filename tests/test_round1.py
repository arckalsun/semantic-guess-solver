"""Tests for ``sgs.round1`` — CLI entry-point behaviour.

Uses :mod:`subprocess` against the real ``python -m sgs.round1`` invocation
on a tiny corpus so we exercise argparse + I/O end-to-end.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sgs.replay import write_replay


@pytest.fixture()
def project_root() -> Path:
    """Repo root = parent of the tests/ dir."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture()
def mini_corpus(tmp_path: Path) -> tuple[Path, Path]:
    words = ["忍者", "剑客", "武士", "浪人", "刺客", "弓手"]
    emb = np.eye(6, dtype=np.float32)  # each word is its own axis
    w_path = tmp_path / "words.json"
    e_path = tmp_path / "emb.npy"
    w_path.write_text(json.dumps(words), encoding="utf-8")
    np.save(e_path, emb)
    return w_path, e_path


def _run_round1(
    project_root: Path,
    replay: Path,
    candidates: Path,
    embeddings: Path,
    *extra: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "sgs.round1",
         "--replay", str(replay),
         "--candidates", str(candidates),
         "--embeddings", str(embeddings),
         *extra],
        capture_output=True,
        text=True,
        cwd=project_root,
        timeout=20,
    )


def test_empty_replay_errors_with_actionable_message(
    project_root: Path, mini_corpus: Path, tmp_path: Path
) -> None:
    w_path, e_path = mini_corpus
    replay = tmp_path / "empty.ndjson"
    replay.write_text("")
    r = _run_round1(project_root, replay, w_path, e_path)
    assert r.returncode == 2
    assert "needs at least one observation" in r.stderr


def test_missing_replay_file_errors(
    project_root: Path, mini_corpus: Path, tmp_path: Path
) -> None:
    w_path, e_path = mini_corpus
    r = _run_round1(project_root, tmp_path / "ghost.ndjson", w_path, e_path)
    assert r.returncode != 0
    assert "not found" in r.stderr


def test_emits_top_n_words_with_scores(
    project_root: Path, mini_corpus: Path, tmp_path: Path
) -> None:
    w_path, e_path = mini_corpus
    replay = tmp_path / "obs.ndjson"
    write_replay(
        replay,
        [
            {"word": "忍者", "score": 0.9, "ts": "t"},
            {"word": "剑客", "score": 0.1, "ts": "t"},
        ],
    )
    r = _run_round1(project_root, replay, w_path, e_path, "--batch-size", "3")
    assert r.returncode == 0
    lines = [l for l in r.stdout.splitlines() if l.strip()]
    assert len(lines) == 3
    # Observed words (忍者, 剑客) should not appear.
    text = r.stdout
    assert "忍者" not in text
    assert "剑客" not in text
    # Each line has rank, word, score.
    parts = lines[0].split()
    assert len(parts) == 3
    assert parts[0] == "0"


def test_out_file_is_ndjson_with_rank_score(
    project_root: Path, mini_corpus: Path, tmp_path: Path
) -> None:
    w_path, e_path = mini_corpus
    replay = tmp_path / "obs.ndjson"
    out = tmp_path / "next.ndjson"
    write_replay(replay, [{"word": "忍者", "score": 0.9, "ts": "t"}])
    r = _run_round1(
        project_root, replay, w_path, e_path,
        "--batch-size", "2", "--out", str(out),
    )
    assert r.returncode == 0
    records = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l]
    assert len(records) == 2
    assert {r["rank"] for r in records} == {0, 1}
    for rec in records:
        assert "word" in rec
        assert "score" in rec
        assert 0.0 <= rec["score"] <= 1.0


def test_include_correct_keeps_observed(
    project_root: Path, mini_corpus: Path, tmp_path: Path
) -> None:
    w_path, e_path = mini_corpus
    replay = tmp_path / "obs.ndjson"
    write_replay(replay, [{"word": "忍者", "score": 1.0, "ts": "t"}])
    r = _run_round1(
        project_root, replay, w_path, e_path,
        "--batch-size", "6", "--include-correct",
    )
    assert r.returncode == 0
    # 忍者 is on its own axis (eye matrix), so it must rank #1 with sim=1.0.
    first_line = r.stdout.splitlines()[0]
    assert "忍者" in first_line
    # Sim should be 1.0.
    assert "1.0000" in first_line