# Contributing

Thanks for your interest in `semantic-guess-solver`! This is a small,
research-grade project. The contribution bar is **TDD-first, zero-runtime-
deps beyond numpy, and Round 1 stays offline-only**.

## Quick rules

1. **Open an issue first** for any non-trivial change. We discuss
   architecture before code.
2. **Tests come first.** Write a failing test (`pytest` should go red),
   then make it green, then refactor.
3. **No new runtime dependencies.** `numpy` is the only one. `pytest` is
   allowed for dev.
4. **Round 1 must stay offline.** No network calls, no browser code in
   `sgs/` proper. The browser probe (Round 2) lives in a separate module
   or repo.
5. **No silent fallback.** Errors should raise with actionable messages,
   not degrade to defaults.

## Local setup

```bash
git clone https://github.com/arckalsun/semantic-guess-solver.git
cd semantic-guess-solver
pip install -e .[dev]
pytest                          # should pass in <1s
```

## Code layout

```text
sgs/
  replay.py    # NDJSON I/O + sha256
  rank.py      # embedding centroid + cosine ranking
  round1.py    # CLI entrypoint (python -m sgs.round1)
tests/         # one test file per module
```

Add a new module? Mirror the layout: `sgs/newmod.py` + `tests/test_newmod.py`.

## Commit & PR

- One logical change per commit.
- Commit message: `<scope>: <imperative summary>` — e.g.
  `rank: drop empty exclude set, use bool flag`.
- PR description should mention the case-study reference (case-1..case-5)
  if the change came from a real puzzle observation.

## Versioning

We follow [semver](https://semver.org/) but aggressively stay on `0.x.y`
until the library has been validated against ≥10 documented puzzles.
Round 1 is `0.1.0` — the moment a Round 2 module lands and we ship a
working probe loop, we'll cut `0.2.0`.

## License

By contributing, you agree your contributions are MIT-licensed, matching
the project license.