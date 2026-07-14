"""Tests for ``sgs.rank`` — centroid fitting + cosine ranking on toy corpus.

We build a tiny 4-word corpus with a planted answer to verify the math
without pulling in real BGE vectors (those weigh ~120 MB).

Reference cases:
* case-1 (shareId 375865943437, answer=忍者) — centroid shift concept.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sgs.rank import fit_centroid, load_corpus, rank


@pytest.fixture()
def toy_corpus(tmp_path: Path) -> tuple[Path, Path, list[str], np.ndarray]:
    """A 4-word 3-D toy corpus where '忍者' sits at the +X axis tip."""
    words = ["忍者", "剑客", "武士", "浪人"]
    # 忍者 at (1, 0, 0); 剑客 at (0, 1, 0); 武士 at (0, 0, 1); 浪人 mixed.
    raw = np.array(
        [
            [1.0, 0.0, 0.0],  # 忍者
            [0.0, 1.0, 0.0],  # 剑客
            [0.0, 0.0, 1.0],  # 武士
            [0.5, 0.5, 0.0],  # 浪人
        ],
        dtype=np.float32,
    )
    # L2-normalise rows so load_corpus treats it as pre-normalised.
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    emb = (raw / norms).astype(np.float32)
    w_path = tmp_path / "words.json"
    e_path = tmp_path / "emb.npy"
    w_path.write_text(json.dumps(words), encoding="utf-8")
    np.save(e_path, emb)
    return w_path, e_path, words, emb


def test_load_corpus_returns_unit_norm(tmp_path: Path) -> None:
    """Raw (non-unit-norm) input gets auto-normalised."""
    words = ["a", "b"]
    raw = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32)
    (tmp_path / "w.json").write_text(json.dumps(words))
    np.save(tmp_path / "e.npy", raw)
    _, emb = load_corpus(tmp_path / "w.json", tmp_path / "e.npy")
    norms = np.linalg.norm(emb, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_load_corpus_shape_mismatch_raises(tmp_path: Path) -> None:
    (tmp_path / "w.json").write_text(json.dumps(["a", "b"]))
    np.save(tmp_path / "e.npy", np.zeros((3, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="disagree"):
        load_corpus(tmp_path / "w.json", tmp_path / "e.npy")


def test_load_corpus_wrong_json_type(tmp_path: Path) -> None:
    (tmp_path / "w.json").write_text(json.dumps({"a": 1}))
    np.save(tmp_path / "e.npy", np.zeros((1, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="list\\[str\\]"):
        load_corpus(tmp_path / "w.json", tmp_path / "e.npy")


def test_fit_centroid_unit_norm(
    toy_corpus: tuple[Path, Path, list[str], np.ndarray],
) -> None:
    _, _, _, _ = toy_corpus
    c = fit_centroid([("忍者", 1.0)], ["忍者"], np.array([[1.0, 0.0, 0.0]]))
    assert np.allclose(np.linalg.norm(c), 1.0, atol=1e-5)


def test_fit_centroid_weighted_pulls_to_high_score(
    toy_corpus: tuple[Path, Path, list[str], np.ndarray],
) -> None:
    """With 忍者=0.9, 剑客=0.1, the centroid should land closer to 忍者."""
    _, _, words, emb = toy_corpus
    c = fit_centroid(
        [("忍者", 0.9), ("剑客", 0.1)], words, emb
    )
    # centroid x-component must dominate y-component.
    assert c[0] > c[1]


def test_fit_centroid_empty_observations_raises() -> None:
    with pytest.raises(ValueError, match="at least one"):
        fit_centroid([], ["a"], np.array([[1.0]]))


def test_fit_centroid_unknown_word_raises(
    toy_corpus: tuple[Path, Path, list[str], np.ndarray],
) -> None:
    _, _, words, emb = toy_corpus
    with pytest.raises(KeyError, match="not in candidate corpus"):
        fit_centroid([("幽灵", 0.5)], words, emb)


def test_fit_centroid_out_of_range_score_raises(
    toy_corpus: tuple[Path, Path, list[str], np.ndarray],
) -> None:
    _, _, words, emb = toy_corpus
    with pytest.raises(ValueError, match="out of"):
        fit_centroid([("忍者", 1.5)], words, emb)


def test_rank_returns_top_k_with_observed_excluded(
    toy_corpus: tuple[Path, Path, list[str], np.ndarray],
) -> None:
    """Probing 忍者+武士 should rank 浪人 first; observed words excluded.

    Toy corpus layout (unit-norm, 3-D):

        忍者 = (1, 0, 0)
        剑客 = (0, 1, 0)
        武士 = (0, 0, 1)
        浪人 = (0.5, 0.5, 0) / |..|  ≈ (0.707, 0.707, 0)

    Centroid from (忍者=0.5, 武士=0.9) lands at (0.357, 0, 0.643)/norm
    ≈ (0.485, 0, 0.874).  Cosines:
        浪人: 0.485 × 0.707 + 0 + 0    ≈ 0.343
        剑客: 0
    So 浪人 wins; observed (忍者, 武士) are excluded.
    """
    _, _, words, emb = toy_corpus
    obs = [("忍者", 0.5), ("武士", 0.9)]
    top = rank(obs, words, emb, top_k=4)
    assert "忍者" not in {w for w, _ in top}
    assert "武士" not in {w for w, _ in top}
    assert top[0][0] == "浪人"
    sims = [s for _, s in top]
    assert sims == sorted(sims, reverse=True)


def test_rank_include_correct_keeps_observed(
    toy_corpus: tuple[Path, Path, list[str], np.ndarray],
) -> None:
    """``exclude_observed=False`` lets the answer surface in the ranking."""
    _, _, words, emb = toy_corpus
    obs = [("忍者", 1.0)]
    top = rank(obs, words, emb, top_k=4, exclude_observed=False)
    assert top[0][0] == "忍者"
    assert top[0][1] == pytest.approx(1.0, abs=1e-5)


def test_rank_all_exhausted_returns_empty(
    toy_corpus: tuple[Path, Path, list[str], np.ndarray],
) -> None:
    _, _, words, emb = toy_corpus
    obs = [(w, 0.5) for w in words]
    assert rank(obs, words, emb, top_k=10) == []


def test_rank_top_k_must_be_positive(
    toy_corpus: tuple[Path, Path, list[str], np.ndarray],
) -> None:
    _, _, words, emb = toy_corpus
    with pytest.raises(ValueError, match="top_k"):
        rank([("忍者", 0.5)], words, emb, top_k=0)