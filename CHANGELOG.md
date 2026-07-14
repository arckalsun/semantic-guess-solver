# Changelog

All notable changes to `semantic-guess-solver` are documented here. The
project follows [semver](https://semver.org/) and the **Keep a Changelog**
format.

## [0.6.0] — 2026-07-14

### Added — Round 7 live wire integration (opt-in E2E)

**Why this round.** After 6 rounds, every existing test still mocked
`Oracle` (`FakeOracle`) or `PlaywrightScript`. Round 3's
`PlaywrightOracle` shipped *believed-correct*, not *proven-correct*.
v0.6.0 adds the missing E2E tier without disrupting the default
test loop.

* **New `tests/integration/` directory**:
  * `test_oracle_live.py` — 4 opt-in tests that boot Chromium and
    call xiaoce.fun:
    * `test_first_probe_returns_known_score_band` — score in [0,1]
      or None (rate-limit / lock)
    * `test_repeated_probe_does_not_solve_unknown_word` — corpus
      is curated losers, must never return `correct=True`
    * `test_persistent_context_reuses_session` — soft-asserts
      that the persistent Chromium context actually carries a
      live session (not just "did not throw")
    * `test_known_daily_answer_when_credentials_and_shareid_are_fresh`
      — *doubly-gated* (integration + live_answer), operator-only
  * `test_skip_guard.py` — 3 unit tests that pin down the opt-in
    mechanism: even with no `.playwright-data/`, the helper
    `_has_live_credentials()` must return False (and therefore
    every live test must SKIP, never FAIL).
* **Skip-not-fail contract**: If `.playwright-data/Default/Cookies`
  is absent, every live test is skipped with an actionable message
  pointing the user at the SKILL.md Round 3 login recipe. CI on a
  clean machine stays green.
* **Two new pytest markers** in `pyproject.toml`:
  * `integration` — live wire tests
  * `live_answer` — operator-only true E2E
* **Zero-impact on default `pytest -q`**: total stays at
  `146 passed, 4 skipped` in 2.14-2.25s (5/5 stable runs).

### Migration

Purely additive. Run the live tier explicitly with:

    pytest -m integration tests/integration/
    # operator-only:
    pytest -m "integration and live_answer" tests/integration/

### Test count

`146 passed, 4 skipped` (was `143 passed`).

## [0.5.0] — 2026-07-14

### Added — `sgs.replay_diff` offline regression suite (Round 6)

**Why this round.** `sgs.solve` (Round 5) writes an NDJSON replay
after every run. With ground truth in hand, the operator can now
ask: *"did my new acquisition regress against last week's run?"*
— **without re-running the network**.

* **Library API**: `compare_runs(path_a, path_b, *, threshold=0.02)`
  returns a frozen `ReplayDiffResult` with:
  * `n_a`, `n_b`, `overlap` (rows in common)
  * `peak_a`, `peak_b`, `peak_score_delta` (signed: `peak_b - peak_a`)
  * `spearman` ρ, `kendall` τ-b (computed on the intersection of
    *scored* rows, ordered by `path_a`'s probe sequence)
  * `topk_intersection` at `k = min(n_a, n_b)`
  * `probes_to_correct_a/b` (1-based probe index, `None` if never)
  * `warning ∈ {"ok", "warn", "alarm"}`
* **CLI**: `python -m sgs.replay_diff a.ndjson b.ndjson [--json]
  [--threshold 0.02]`. Exit codes:
  * `0` — warning=ok
  * `1` — warning=warn (peak regressed beyond threshold)
  * `2` — warning=alarm (no overlap; comparing unrelated sessions)
  * `3` — configuration / IO error
* **Strict NDJSON loader**: `load_replay(path)` raises
  `ValueError` on missing `word` / `score`, `FileNotFoundError` on
  missing file, `JSONDecodeError` on corrupt lines. **No silent
  drop** of bad rows — a malformed replay is a hard schema error.
* **stdlib-only** rank correlation: `_rank` with mid-rank tie-break,
  pure-Python `spearman` and `kendall_tau`. No scipy / numpy.

### Test count

`143 passed` in 2.10-2.21s (Round 6 added 20; 5/5 stable runs).

### Migration

This is purely additive — no other module was modified.

## [0.4.0] — 2026-07-14

### Added

- `sgs.solve` — end-to-end solver orchestrator and CLI.
  - `solve_run(...)` library entry point that wires an `Oracle` (Round
    2/3) to an `Acquisition` (Round 4) and a replay-NDJSON file.
  - `cli_main(...)` exposes `python -m sgs.solve run --candidates X
    --replay Y --oracle {fake|playwright} --acquisition {random|greedy|
    uncertainty} --max-probes N --plateau-window W`.
  - Plateau detection: stops early if the best observed score has not
    improved for `plateau_window` consecutive probes (configurable,
    default 5; pass `--plateau-window 0` to disable).
  - Lazy `playwright` import — `import sgs.solve` does **not** pull in
    the browser stack; `playwright.sync_api` only enters `sys.modules`
    when `--oracle playwright` is actually requested AND the run
    starts.
  - Always calls `oracle.close()` (P0: no browser resource leak on
    budget-exhaust or mid-loop exception).
  - Distinct exit codes for operator triage:
    - `0` = solved (`correct=True` reached)
    - `2` = budget exhausted without `correct=True`
    - `3` = configuration error (missing file, unknown strategy,
      missing `--share-id` for `--oracle playwright`, etc.)
  - `--json` flag for machine-readable output (for shell pipes / CI).
- `tests/test_solve.py` — 20 tests covering the `SolveResult` contract,
  replay-NDJSON append, plateau detection (with the precise 5-probe
  trace), acquisition dispatch, CLI exit codes, lazy playwright import,
  oracle lifecycle (`close()` on both correct and budget-exhaust paths),
  and arg validation (`max_probes > 0`, non-empty candidates,
  `plateau_window ≥ 0`).

### Changed

- Test count: `103 → 123`. Total runtime stays under 2.3 s × 5 stable
  runs.
- No public API breakage. `sgs.oracle.Oracle` / `OracleResponse`,
  `sgs.learn.{Random,Greedy,Uncertainty}Acquisition`,
  `sgs.wire.{WireEndpoint,PlaywrightOracle}` are unchanged.

### Why this round is `0.x` not `1.0`

The Round 5 CLI exercises the offline layer end-to-end (FakeOracle +
greedy/uncertainty + replay-NDJSON append + plateau + exit codes),
but **no live-wire integration test** is shipped yet. The next
candidate is a `markers=integration` test suite that runs the CLI
against `PlaywrightOracle` with a logged-in persistent context. Until
that ships, `1.0` is premature.

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