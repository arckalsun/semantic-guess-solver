# semantic-guess-solver

> Reverse-engineering the **xiaoce.fun GuessWord daily oracle** with
> semantic similarity (BGE embedding + black-box scoring) and an offline
> Round-1 cosine ranker. Pure numpy, NDJSON replay format, TDD-tested.

[![tests](https://img.shields.io/badge/tests-52%20passed-brightgreen)]()
[![python](https://img.shields.io/badge/python-≥3.10-blue)]()
[![deps](https://img.shields.io/badge/runtime%20deps-numpy%20only-orange)]()
[![license](https://img.shields.io/badge/license-MIT-lightgrey)]()

---

## What this is

A research-grade solver for the **GuessWord** daily puzzle on
[xiaoce.fun](https://xiaoce.fun). The game shows you 30 candidates per round
and returns a black-box similarity score for each guess (range ≈ 0.4–1.0).
The answer is one 2-character Chinese word.

This library ships **Round 1 (offline ranker) + Round 2 (online probe)**:

| Round | Module(s) | Network? |
| --- | --- | --- |
| **Round 1** — pure-numpy cosine ranker | `sgs.replay`, `sgs.rank`, `sgs.round1` | no |
| **Round 2** — oracle contract + batch probe + rate-limit | `sgs.oracle`, `sgs.ratelimit`, `sgs.probe` | yes (or fake) |

The browser wire adapter (Playwright/Chromium CDP) and the active-learning
loop live in **Round 3+** — see [Roadmap](#roadmap).

---

## Why Round 1 works

Hypothesis validated against 5 documented puzzles (cases 1–5, see
[`references/`](references/)):

1. **BGE-zh-base 768-d embeddings** cluster semantically similar Chinese
   tokens in cosine space.
2. **The oracle score is approximately monotonic** in cosine similarity to
   the unknown answer.
3. **Even 3-5 noisy observations** shift a score-weighted centroid toward
   the correct semantic cluster — that's the active-learning signal.

A 2-stage cluster-narrowing + 1-day-endpoint test (case-5) hit `0.989`
on the answer ("萧山") with a 100-word seed → 20-word pivot →
single-word answer pass.

---

## Installation

```bash
pip install -e .[dev]
```

Tested on Python 3.10 – 3.12. **Runtime dependency: numpy only.**

---

## Quick start

```bash
# 1. Probe the oracle with a few words (browser-side; not in scope here).
# 2. Record observations as NDJSON — one record per line:
cat > replay/376634286041.ndjson <<'EOF'
{"word": "剑客", "score": 0.398, "ts": "2026-07-14T07:55:00Z"}
{"word": "武士", "score": 0.481, "ts": "2026-07-14T07:55:08Z"}
{"word": "忍者", "score": 0.612, "ts": "2026-07-14T07:55:17Z"}
{"word": "浪人", "score": 0.527, "ts": "2026-07-14T07:55:25Z"}
EOF

# 3. Run Round 1 — get the next 30 words to probe.
python -m sgs.round1 \
    --replay     replay/376634286041.ndjson \
    --candidates /path/to/cand_words.json \
    --embeddings /path/to/cand_emb.npy \
    --batch-size 30 \
    --out        replay/376634286041-next.ndjson
```

The CLI prints `rank  word  cosine` per line and optionally writes an
NDJSON file with `{"word", "rank", "score"}` records.

---

## API

### `sgs.replay` — NDJSON replay I/O + sha256 fingerprint

```python
from sgs.replay import write_replay, read_replay, stream_replay, fingerprint

write_replay(Path("obs.ndjson"), [
    {"word": "忍者", "score": 0.989, "ts": "2026-07-14T08:11:32Z",
     "correct": True, "doubleScore": False},
])
records = read_replay(Path("obs.ndjson"))            # list[dict]
for rec in stream_replay(Path("big.ndjson")): ...    # memory-friendly
sha = fingerprint(Path("obs.ndjson"))                # hex sha256
```

Required keys: `word`, `score`, `ts`.
Optional keys preserved verbatim: `correct`, `doubleScore`, …

### `sgs.rank` — embedding centroid + cosine ranking

```python
from sgs.rank import load_corpus, fit_centroid, rank

words, emb = load_corpus("cand_words.json", "cand_emb.npy")
top30 = rank(
    observations=[("忍者", 0.612), ("剑客", 0.398), ("武士", 0.481)],
    words=words,
    emb=emb,
    top_k=30,                    # default 30
    exclude_observed=True,       # set False to audit how the answer ranks
)
# top30 == [(word, cosine), ...] sorted descending
```

### `sgs.round1` — CLI

```text
usage: python -m sgs.round1 [-h] --replay REPLAY --candidates CANDIDATES
                            --embeddings EMBEDDINGS [--batch-size 30]
                            [--out OUT] [--include-correct]
```

---

## Development

### Run the tests

```bash
pytest                       # 27 tests, ~0.7s, 0 network
```

### Layout

```text
sgs/
  __init__.py     # __version__
  replay.py       # NDJSON + sha256
  rank.py         # centroid + cosine
  round1.py       # CLI
tests/
  test_replay.py  # 10 tests
  test_rank.py    # 12 tests
  test_round1.py  # 5 subprocess tests
```

### Design principles

1. **Round 1 is pure numpy** — zero business logic, zero network code,
   zero dependency on the browser probe (Round 2).
2. **TDD**: tests came first; failures drove the API surface (e.g.
   `exclude_observed: bool` replaced an awkward `exclude: set` parameter).
3. **NDJSON + sha256** — replay files are tamper-evident and
   memory-streamable for big sessions.
4. **No silent fallback** — every corpus / score error raises immediately.

---

## Caveats

- This library does **not** probe the oracle. Pair it with a Playwright
  / browser-fetch script (Round 2, WIP).
- BGE embeddings are large (~120 MB for 38930 Chinese words). They are
  **not** committed; download separately or generate from
  [BAAI/bge-base-zh-v1.5](https://huggingface.co/BAAI/bge-base-zh-v1.5).
- The 2-character answer constraint is *not* enforced by the math — feed
  in a corpus of any word length. Cases 1–5 happened to be 2-character
  answers; the API is length-agnostic.

---

## Roadmap

| Round | Status | What lands |
| --- | --- | --- |
| **1. Offline ranker** | ✅ `v0.1.0` | numpy cosine + NDJSON replay + sha256 (27 tests) |
| **2. Online probe** | ✅ `v0.2.0` | `Oracle` protocol + `TokenBucket` + batch probe + stop-on-correct (52 tests) |
| **3. Browser wire** | planned | Playwright/Chromium CDP, persistent context, login-by-human-once, `fcntl.flock` against double-launch |
| **4. Active learning** | planned | `U = α·pred + β·uncert + γ·diversity`, multi-round convergence |
| **5. End-to-end** | planned | `dry-run` / `assisted` / `supervised` / `live` modes; gates: `--max-probes`, `--max-domain-switches`, `--stop-on-plateau` |
| **6. Replay regression** | planned | NDJSON-driven offline regression with golden diff |

---

## License

MIT. See [`LICENSE`](LICENSE).

---

## References

- Case study (1–5 puzzles documented): `references/xiaoce-fun-case-study.md`
- Companion skill (in Hermes Agent): `xiaoce-fun-case-study-375865943437`