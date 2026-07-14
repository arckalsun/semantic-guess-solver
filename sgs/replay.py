"""NDJSON replay file reader/writer + sha256 integrity check.

A replay file records a single probing session against the black-box oracle.
One JSON object per line; each object carries at minimum:

* ``word`` — the candidate we submitted (``str``).
* ``score`` — the server-reported similarity score (``float`` ∈ [0, 1]).
* ``ts`` — ISO-8601 UTC timestamp when the probe returned.
* ``correct`` — ``True`` iff the probe was the answer (default ``False``).
* ``doubleScore`` — server-reported bonus flag (``bool``, default ``False``).

Extra keys (e.g. ``correct``, ``wrong``, ``rate_limited``) are tolerated and
preserved verbatim; this lets downstream tooling pass-through error flags
without breaking the parser.

Reference: case-1 (shareId 375865943437, answer = 忍者) first NDJSON format
prototype; case-2/3/4/5 confirmed the same envelope.

Example
-------
>>> lines = [
...     {"word": "忍者", "score": 0.989, "ts": "2026-07-14T08:11:32Z",
...      "correct": True, "doubleScore": False},
...     {"word": "剑客", "score": 0.412, "ts": "2026-07-14T08:10:11Z"},
... ]
>>> from pathlib import Path
>>> p = Path("/tmp/example.ndjson")
>>> write_replay(p, lines)
>>> read_replay(p) == lines
True
>>> fingerprint(p)
'a1...e7'
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator, Mapping

# Canonical envelope — keys we promise to honour when reading.
# Other keys are preserved verbatim.
REQUIRED_KEYS = ("word", "score", "ts")
OPTIONAL_KEYS = ("correct", "doubleScore")


def write_replay(path: Path, lines: Iterable[Mapping[str, object]]) -> int:
    """Write NDJSON ``lines`` to ``path``; return count written.

    Uses ``ensure_ascii=False`` so Chinese words stay readable on disk.
    Each line ends with ``\\n``; trailing newline is also present after the
    last record (POSIX-friendly, ``cat`` friendly).
    """
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for obj in lines:
            fh.write(json.dumps(obj, ensure_ascii=False))
            fh.write("\n")
            n += 1
    return n


def read_replay(path: Path) -> list[dict[str, object]]:
    """Parse an NDJSON file into a list of dicts.

    Empty lines are skipped. Malformed JSON or missing required keys raise
    :class:`ValueError` with the line number for easy debugging.

    The result preserves the insertion order of keys inside each record.
    """
    out: list[dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            # Strip trailing newline AND surrounding whitespace so blank /
            # whitespace-only lines are skipped (matching the recorded NDJSON
            # convention used in case-1..5 replays).
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{lineno}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(obj, dict):
                raise ValueError(
                    f"{path}:{lineno}: expected object, got {type(obj).__name__}"
                )
            for k in REQUIRED_KEYS:
                if k not in obj:
                    raise ValueError(
                        f"{path}:{lineno}: missing required key {k!r}"
                    )
            out.append(obj)
    return out


def stream_replay(path: Path) -> Iterator[dict[str, object]]:
    """Yield replay records one at a time — memory-friendly for huge files."""
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            for k in REQUIRED_KEYS:
                if k not in obj:
                    raise ValueError(
                        f"{path}:{lineno}: missing required key {k!r}"
                    )
            yield obj


def fingerprint(path: Path) -> str:
    """Return the hex sha256 of the file's bytes.

    Computed by streaming the file in 64 KiB chunks. Use this to detect
    tampering or accidental re-export of a replay file.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_fingerprint(path: Path, expected: str) -> bool:
    """Return ``True`` iff ``fingerprint(path) == expected`` (case-insensitive)."""
    return fingerprint(path).lower() == expected.lower()


__all__ = [
    "REQUIRED_KEYS",
    "OPTIONAL_KEYS",
    "write_replay",
    "read_replay",
    "stream_replay",
    "fingerprint",
    "verify_fingerprint",
]