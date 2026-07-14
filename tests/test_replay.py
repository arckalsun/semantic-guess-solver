"""Tests for ``sgs.replay`` — NDJSON round-trip + sha256 fingerprint.

Reference cases:
* case-1 (shareId 375865943437, answer=忍者) — first NDJSON prototype.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sgs.replay import (
    fingerprint,
    read_replay,
    stream_replay,
    verify_fingerprint,
    write_replay,
)


def _sample_lines() -> list[dict]:
    return [
        {
            "word": "忍者",
            "score": 0.989,
            "ts": "2026-07-14T08:11:32Z",
            "correct": True,
            "doubleScore": False,
        },
        {
            "word": "剑客",
            "score": 0.412,
            "ts": "2026-07-14T08:10:11Z",
        },
    ]


def test_write_then_read_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "case-1.ndjson"
    lines = _sample_lines()
    n = write_replay(p, lines)
    assert n == 2
    out = read_replay(p)
    assert out == lines


def test_chinese_chars_survive_unicode(tmp_path: Path) -> None:
    """``ensure_ascii=False`` keeps Chinese readable on disk."""
    p = tmp_path / "unicode.ndjson"
    write_replay(p, [{"word": "忍者", "score": 0.5, "ts": "t"}])
    text = p.read_text(encoding="utf-8")
    assert "忍者" in text  # not escaped
    assert "\\u" not in text


def test_empty_lines_are_skipped(tmp_path: Path) -> None:
    p = tmp_path / "blanks.ndjson"
    p.write_text(
        '{"word": "x", "score": 0.1, "ts": "t"}\n'
        "\n"
        '   \n'
        '{"word": "y", "score": 0.2, "ts": "t"}\n',
        encoding="utf-8",
    )
    assert [r["word"] for r in read_replay(p)] == ["x", "y"]


def test_missing_required_key_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.ndjson"
    p.write_text('{"word": "x", "score": 0.1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="missing required key"):
        read_replay(p)


def test_malformed_json_raises(tmp_path: Path) -> None:
    p = tmp_path / "broken.ndjson"
    p.write_text('{"word": "x", "score": 0.1, "ts": "t"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        read_replay(p)


def test_non_object_line_raises(tmp_path: Path) -> None:
    p = tmp_path / "list.ndjson"
    p.write_text('[1, 2, 3]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="expected object"):
        read_replay(p)


def test_stream_matches_read(tmp_path: Path) -> None:
    p = tmp_path / "case-1.ndjson"
    write_replay(p, _sample_lines())
    streamed = list(stream_replay(p))
    assert streamed == read_replay(p)


def test_fingerprint_is_deterministic(tmp_path: Path) -> None:
    p = tmp_path / "case-1.ndjson"
    write_replay(p, _sample_lines())
    assert fingerprint(p) == fingerprint(p)
    # 64 hex chars (sha256)
    assert len(fingerprint(p)) == 64


def test_fingerprint_changes_on_byte_edit(tmp_path: Path) -> None:
    p = tmp_path / "case-1.ndjson"
    write_replay(p, _sample_lines())
    before = fingerprint(p)
    # tamper one byte
    raw = p.read_bytes()
    p.write_bytes(raw.replace(b"0.989", b"0.990"))
    after = fingerprint(p)
    assert before != after


def test_verify_fingerprint_case_insensitive(tmp_path: Path) -> None:
    p = tmp_path / "case-1.ndjson"
    write_replay(p, _sample_lines())
    fp = fingerprint(p)
    assert verify_fingerprint(p, fp)
    assert verify_fingerprint(p, fp.upper())  # case-insensitive
    assert not verify_fingerprint(p, "0" * 64)