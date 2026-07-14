"""Cosine-similarity ranking against a frozen embedding matrix.

This is the *brain* of Round 1. Given:

* a candidate pool (list of Chinese words) and their L2-normalised
  embeddings (``(N, D)`` float32 array),
* a small set of (word, score) observations from the black-box oracle,

estimate the unknown answer's embedding as a *score-weighted centroid* of the
known vectors, then rank every candidate by cosine similarity to that
estimate. High cosine ⇒ likely near the answer in embedding space.

Why this works (Round 1 hypothesis):

* BGE-zh-base places semantically similar Chinese tokens close in cosine
  space.
* The oracle's score is monotonic in cosine — case-1..5 all show
  monotonic alignment with embedding distance.
* Even 3-5 noisy observations let a centroid drift toward the correct
  semantic cluster (the active-learning driver).

API summary
-----------

* :func:`load_corpus` — read (words.json, emb.npy) pair from disk, validate
  shapes match and embeddings are unit-norm.
* :func:`fit_centroid` — score-weighted mean of labelled vectors, then
  re-normalise to unit length.
* :func:`rank` — given observations + corpus, return ``[(word, sim)]``
  sorted descending.

Example
-------
>>> import numpy as np
>>> from sgs.rank import load_corpus, fit_centroid, rank
>>> words, emb = load_corpus("/data/cand.json", "/data/cand.npy")
>>> obs = [("忍者", 0.612), ("剑客", 0.398), ("武士", 0.481)]
>>> top10 = rank(obs, words, emb, top_k=10)
>>> top10[0][0]  # likely cluster pivot (case-1)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np


def load_corpus(
    words_json: str | Path,
    emb_npy: str | Path,
) -> tuple[list[str], np.ndarray]:
    """Load candidate pool: a JSON list of words + an (N, D) float32 matrix.

    Validates:

    * the JSON parses to a list of ``str``,
    * the .npy shape's first axis equals ``len(words)``,
    * dtype is float (we cast to float32 for downstream normalisation),
    * rows are unit-norm (within ``1e-3`` tolerance) — re-normalise if not.

    Returns ``(words, emb)`` where ``emb`` is guaranteed to be float32 with
    L2-unit rows.
    """
    words_raw = json.loads(Path(words_json).read_text(encoding="utf-8"))
    if not isinstance(words_raw, list) or not all(
        isinstance(w, str) for w in words_raw
    ):
        raise ValueError(
            f"{words_json}: expected list[str], got "
            f"{type(words_raw).__name__}"
        )
    emb = np.load(emb_npy)
    if emb.ndim != 2:
        raise ValueError(
            f"{emb_npy}: expected 2-D array, got shape {emb.shape}"
        )
    if emb.shape[0] != len(words_raw):
        raise ValueError(
            f"{words_json} (n={len(words_raw)}) and {emb_npy} "
            f"(rows={emb.shape[0]}) disagree"
        )
    if emb.dtype != np.float32:
        emb = emb.astype(np.float32, copy=False)
    # L2-normalise rows if not already normalised
    norms = np.linalg.norm(emb, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        emb = emb / np.clip(norms[:, None], 1e-12, None)
        emb = emb.astype(np.float32)
    return words_raw, emb


def fit_centroid(
    observations: Sequence[tuple[str, float]],
    words: Sequence[str],
    emb: np.ndarray,
) -> np.ndarray:
    """Compute score-weighted centroid of observed word embeddings.

    ``observations`` is ``[(word, score)]``. Scores are clipped to
    ``[0, 1]`` to guard against malformed replay records. Centroid is
    re-normalised to unit length before return.

    Words not in the corpus raise ``KeyError`` — fail fast rather than
    silently drop; the corpus is meant to be exhaustive for the search
    domain.
    """
    if not observations:
        raise ValueError("need at least one observation to fit centroid")
    word_to_idx = {w: i for i, w in enumerate(words)}
    vec = np.zeros(emb.shape[1], dtype=np.float32)
    total_w = 0.0
    for word, score in observations:
        if word not in word_to_idx:
            raise KeyError(
                f"observed word {word!r} not in candidate corpus "
                f"(size={len(words)})"
            )
        s = float(score)
        if s < 0.0 or s > 1.0 or not (s == s):  # NaN-safe
            raise ValueError(f"score for {word!r} out of [0,1]: {score!r}")
        vec += s * emb[word_to_idx[word]]
        total_w += s
    if total_w <= 0.0:
        raise ValueError("all observation scores are zero — cannot fit")
    centroid = vec / total_w
    n = float(np.linalg.norm(centroid))
    if n < 1e-12:
        raise ValueError("centroid collapsed to zero vector")
    return (centroid / n).astype(np.float32)


def rank(
    observations: Sequence[tuple[str, float]],
    words: Sequence[str],
    emb: np.ndarray,
    *,
    top_k: int = 30,
    exclude_observed: bool = True,
) -> list[tuple[str, float]]:
    """Score every corpus word by cosine similarity to the fitted centroid.

    Returns top-``top_k`` ``(word, cosine)`` pairs sorted descending.

    Parameters
    ----------
    observations
        ``[(word, score)]`` pairs from the oracle — used to fit the centroid.
    words, emb
        Candidate corpus and its L2-normalised ``(N, D)`` embeddings.
    top_k
        Number of results to return. Must be positive.
    exclude_observed
        If ``True`` (default), drop already-observed words from the ranking
        — they are redundant to probe again. Set ``False`` for audit
        (``--include-correct`` CLI flag) where the user wants to see how
        the answer ranks relative to other candidates.

    The cosine is clipped to ``[-1, 1]`` before return to absorb floating-
    point overshoot from unit-norm maths.
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    obs_words = {w for w, _ in observations}
    centroid = fit_centroid(observations, words, emb)
    # emb is (N, D), centroid is (D,). Single matmul gives all cosines.
    sims = emb @ centroid
    sims = np.clip(sims, -1.0, 1.0)
    # Build mask: drop observed words unless explicitly told to keep them.
    mask = np.ones(len(words), dtype=bool)
    if exclude_observed:
        for i, w in enumerate(words):
            if w in obs_words:
                mask[i] = False
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    sub_sims = sims[idx]
    order = np.argsort(-sub_sims, kind="stable")[:top_k]
    return [(words[idx[i]], float(sub_sims[i])) for i in order]


__all__ = ["load_corpus", "fit_centroid", "rank"]