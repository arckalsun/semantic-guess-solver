# Changelog

All notable changes to `semantic-guess-solver` are documented here. The
project follows [semver](https://semver.org/) and the **Keep a Changelog**
format.

## [0.3.0] — 2026-07-14

### Added — Round 3: real wire + single-instance lock

- `sgs.flock` — `SingleInstanceLock` (fcntl.flock + banner w/ pid) and the
  `single_instance` decorator. Prevents two solvers racing for the
  same shareId in the same bandit's rate-limit quota. `AlreadyRunningError`
  is the typed signal — note that **mp fork preserves class identity**, so
  using `importlib.reload(sgs.flock)` after `from sgs.flock import
  AlreadyRunningError` will silently desync the two references and break
  `except` clauses inside `mp.Process` children (debugged during v0.3.0).
- `sgs.wire`:
  - `base.WireEndpoint.guess_url(word)` — produces the URL-encoded
    `https://xiaoce.fun/api/v0/quiz/daily/GuessWord/guessV1?word=...`
    with `safe=""` so CJK characters are properly encoded.
  - `base.parse_response(raw, word)` — three-way classifier that
    recognises (a) `data:null` + Chinese rate-limit message → marked
    `rate_limited=True` (NOT a lock!), (b) `data:null` without rate-limit
    message → `score=None` (lock), (c) normal envelope →
    `score/doubleScore/correct`.
  - `playwright.PlaywrightOracle(share_id, script=...)` — protocol-driven
    so tests inject a `FakeScript` and never boot Chrome. Real script
    builds headers (`fun-device: web`), fetches via `page.evaluate`,
    calls `close()` only when the script is owned.
- 32 new tests (`test_flock`, `test_wire_base`, `test_wire_playwright`).

### Added — Round 4: active learning loop

- `sgs.learn` — `Acquisition` Protocol + three strategies:
  - `RandomAcquisition(seed)` — uniform random baseline.
  - `GreedyAcquisition` — picks highest-known-score next; treats
    unprobed candidates as 0.0 (cold-start) so warm-ups must inject
    a `history=` prior when you want Greedy to dominate.
  - `UncertaintyAcquisition(threshold)` — picks candidates closest to
    a threshold (default 0.5, the bandit mid-band) for max info.
- `active_solve(*, oracle, candidates, acquisition, budget, history=None)`
  orchestrator. Optional `history` is the ranker's *prior* — the fitted
  model's predicted scores for candidates we haven't yet probed in this
  session. Affects ranking via the acquisition function but is **not**
  counted as already-seen and **not** returned in the probe list.
- 15 new tests (`test_learn`).

### Validation

All 103 tests pass 5× stable (2.0-2.2s wall per run).

### Internal design notes

- `OracleResponse.score` was `float` in Round 1; v0.3.0 promotes it to
  `Optional[float]` so the wire layer can express "server-side lock
  (data:null, no rate-limit msg)" without sentinel magic numbers.
- `WireEndpoint.guess_url` was a `property` in an early draft;
  switched to a method because callers don't have a `@cached_property`
  boundary — `urllib.parse.quote(safe="")` must run *per-call* in case
  the caller passes CJK that hasn't been encoded.
- `active_solve` `history` semantics: the seed list is the *ranker's
  prior*, not "words already probed". The `seen` set starts empty;
  each round picks one fresh word via the acquisition function.

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