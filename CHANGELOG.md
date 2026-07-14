# Changelog

All notable changes to `semantic-guess-solver` are documented here. The
project follows [semver](https://semver.org/) and the **Keep a Changelog**
format.

## [0.2.0] — 2026-07-14

### Added — Round 2: online probe layer

- `sgs.oracle` — `Oracle` runtime-checkable Protocol, `OracleResponse`
  dataclass with the canonical wire envelope (score / doubleScore /
  correct / rateLimited), and a script-driven `FakeOracle` for tests.
- `sgs.ratelimit` — thread-safe `TokenBucket` (rate=0.8/s, burst=2) with
  `take()` context manager; `make_default_bucket()` pinned to the
  case-study-tested recipe.
- `sgs.probe` — `probe_batch` and `probe_and_record` with stop-on-correct,
  rate-limit accounting, and NDJSON **append** (so Round 1 → 2 → 3 share
  one continuous replay log per shareId).
- `sgs.already_correct` — server-lock helper: returns the set of words the
  oracle has confirmed correct (used to build skip-lists for subsequent
  batches; the case study showed the server returns `data:null` for
  re-probes of locked words).
- 25 new tests (`test_oracle`, `test_ratelimit`, `test_probe`); total
  now **52 passed in 0.79s**.
- README "Roadmap" section + module table.
- `__version__` bumped to `0.2.0`; `__all__` re-exports the new symbols.

### Validation

All 52 tests pass with **zero** runtime dependencies beyond numpy (PEP 561
marker preserved).

## [0.1.0] — 2026-07-14

### Added

- `sgs.replay` — NDJSON read/write + sha256 fingerprint with verified
  required-keys envelope (`word`, `score`, `ts`); optional `correct` /
  `doubleScore` pass-through; streaming iterator.
- `sgs.rank` — embedding-corpus loader (auto-L2-normalises if needed),
  score-weighted centroid fitting, cosine-similarity ranking with
  configurable `exclude_observed`.
- `sgs.round1` — `python -m sgs.round1` CLI: drives one probe batch from
  an existing NDJSON replay. Refuses to run on empty replay with an
  actionable error.
- 27-test pytest suite covering all three modules, including subprocess
  integration tests for the CLI.
- `pyproject.toml` (MIT, `numpy`-only runtime, `pytest` dev-only), MIT
  `LICENSE`, `CONTRIBUTING.md`, this `CHANGELOG.md`, `.gitignore`.

### Validated against

- **case-1** (shareId `375865943437`, answer = 忍者, score 0.989)
- **case-5** (shareId `376634286041`, answer = 萧山, score 0.989, 3-phase
  cluster narrowing).

### Known limitations

- Library is **offline-only**; the browser probe loop is out of scope for
  Round 1 and will ship as a separate module in `0.2.0`.
- BGE embeddings are not bundled (≈120 MB); users supply their own or
  download from [BAAI/bge-base-zh-v1.5](https://huggingface.co/BAAI/bge-base-zh-v1.5).

### Internal design notes (for future contributors)

- The original `rank()` API took an `exclude: set[str]` parameter; this
  was refactored to `exclude_observed: bool` because the boolean
  semantics were clearer than "empty-set means default". See
  `tests/test_rank.py::test_rank_include_correct_keeps_observed`.
- `read_replay` / `stream_replay` use `str.strip()` (not `rstrip("\n")`)
  so whitespace-only lines in concatenated replays are tolerated.