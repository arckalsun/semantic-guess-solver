# Case study — xiaoce.fun GuessWord (cases 1–5)

> Companion skill (Hermes Agent):
> [`xiaoce-fun-case-study-375865943437`](../)
> — see the skill body for the full reverse-engineering timeline.

## What this documents

The 5 documented daily-puzzle sessions that drove the design of Round 1
of `semantic-guess-solver`. Each case is one `shareId` from
[xiaoce.fun](https://xiaoce.fun)'s **GuessWord** mini-game, with the
oracle's response envelope and the cluster-narrowing trajectory that
hit the answer.

## The oracle envelope

Every probe call hits the same endpoint:

```
GET https://xiaoce.fun/api/v0/quiz/daily/GuessWord/guessV1
    ?word=<candidate>&shareId=<id>&skipBusinessErrorToast=true
Headers:
  fun-device: web
```

…and returns a JSON envelope of the shape:

```json
{
  "code": 0,
  "data": {
    "score":          0.989,
    "doubleScore":    true,
    "correct":        true,
    "rateLimited":    false
  }
}
```

Three signals — `data.score` (continuous 0–1), `data.doubleScore` (bool
bonus), `data.correct` (bool, the answer found) — drive every algorithm
in this repo.

## Case 1 — shareId `375865943437`, answer = **忍者** (0.989)

- 30-word seed batch probed against a BGE-zh-base cosine centroid.
- 3-phase cluster narrowing: 忍者 first surfaced at rank #3 with
  score 0.612; subsequent probes of semantically-similar Japanese-martial
  terms (剑客, 武士, 浪人) shifted the centroid; final batch hit
  忍者 = 0.989 + `doubleScore:true` + `correct:true`.
- **Established the Round-1 envelope.**

## Case 2 — specialised noun (`correct` plateau)

- A category-specific word (proper-noun / brand) where the embedding
  centroid drifted off-cluster because no neighbour in the candidate
  pool was close enough.
- **Lesson**: Round 1 alone is not enough for proper nouns. Flagged
  for Round 3 active-learning (specialised-corpus augmentation).

## Case 3 — semantic plateau

- Multiple high-cosine neighbours but none hits `correct`. Centroid
  convergence stalls after 50+ probes.
- **Lesson**: a top-k plateau is a signal to switch strategies (raise
  temperature, switch to a different embedding model, or escalate to
  a browser-driven grid search). Out of scope for Round 1.

## Case 4 — shareId `373443369893`

- Historical case (pre-Round-1-design). Replayed offline to validate
  the NDJSON + fingerprint tools. sha256 stable across re-imports.

## Case 5 — shareId `376634286041`, answer = **萧山** (0.989)

- The cleanest 3-phase trajectory in the study:
  1. **Phase 1** — 100-word seed → cluster around city names → pivot
     上海 (0.722).
  2. **Phase 2** — pivot neighbours → 浙江 (0.741) → 杭州 (0.812).
  3. **Phase 3** — district pivot → 余杭 (0.876) → 萧山 (0.989 +
     `doubleScore:true` + `correct:true`).
- **Established that 3-phase cluster narrowing works for hierarchical
  semantic categories.** This is the canonical demo in `README.md`.

## Why Round 1 was enough

Cases 1 and 5 — the two "clean" wins — share a structure: the answer
sits in a coherent semantic cluster that the BGE centroid can drift
toward with 4-5 observations. Cases 2 and 3 are documented as *known
failure modes* — Round 1 should fail loudly on them, not silently
degrade, which is why the ranker raises rather than returns a guess
with low confidence.

## Replay fingerprinting

Every replay NDJSON is sha256-hashed via `sgs.replay.fingerprint()`. We
store the fingerprint alongside the file in case-study notes so future
contributors can verify they are looking at the exact same oracle
response sequence.

```bash
python -c "from sgs.replay import fingerprint; print(fingerprint('replay/example-376634286041-phase3.ndjson'))"
```

### Known fingerprints (v0.1.0)

| File | sha256 |
|------|--------|
| `replay/example-376634286041-phase1.ndjson` | `956b648a3d1fa846336f1546827262a802cd1067bc638b262dbfd10500ba8385` |
| `replay/example-376634286041-phase3.ndjson` | `7f894d7bc82235b580223e1cb356167fa582ee3eea14f87de314e0a99f12a6c9` |

If the fingerprint on disk differs from the table above, somebody
edited the file. The hash is content-addressed, so even an invisible
whitespace change will trip it.