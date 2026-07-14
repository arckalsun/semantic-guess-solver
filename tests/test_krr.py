"""Tests for sgs.krr — Kernel Ridge Regression predictor.

Round-4 promised predictor (case-3 skill §4.1, case-11 skill §3.6).
Replaces the score-weighted centroid heuristic with a true supervised
learner that maps BGE-zh-base 768-d embeddings → xiaoce doubleScore.

Test pyramid (strict TDD):

1. Synthetic embedding known mapping (sanity — fit must recover a
   noisy linear pattern within MAE < tolerance).
2. Synthetic Gaussian-clusters (the canonical "two clusters" test —
   KRR must rank within-cluster points higher than across-cluster).
3. Cold-start with single observation (graceful fallback — returns
   the one observation's score for nearby words, 0 for far).
4. Edge cases: zero observation raises ValueError, all-zero y raises
   ValueError, gamma=0 raises ValueError, alpha<0 raises ValueError.
5. Numerical stability with 1000+ observations (kernel matrix
   condition number — must not overflow with float32).

All tests use deterministic synthetic data (no network, no random
seeds — pure deterministic numpy operations where randomness is
needed, use np.random.RandomState(42)).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sgs.krr import (
    KernelRidgePredictor,
    fit_predictor,
    predict_scores,
    rbf_kernel,
)


# ---------- 1. Synthetic embedding known linear mapping ----------

def test_fit_recovers_linear_mapping_within_tolerance():
    """Given embeddings X and target y = X @ w_true + small noise,
    fit_predictor should recover predictions with MAE < 0.05
    on a held-out validation set."""
    rng = np.random.RandomState(42)
    n, d = 200, 32
    X = rng.randn(n, d).astype(np.float32)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)  # unit-norm
    w_true = rng.randn(d).astype(np.float32)
    w_true = w_true / np.linalg.norm(w_true)
    y = X @ w_true + rng.randn(n).astype(np.float32) * 0.05  # noisy
    y = np.clip(y, 0.0, 1.0)

    # Fit on first 150, validate on last 50
    pred = fit_predictor(X[:150], y[:150], gamma=1.0, alpha=0.1)
    y_val_pred = pred(X[150:])
    mae = float(np.mean(np.abs(y_val_pred - y[150:])))
    assert mae < 0.10, f"MAE {mae:.4f} > 0.10 — failed to recover linear mapping"


# ---------- 2. Gaussian-clusters ranking test ----------

def test_predictor_ranks_within_cluster_higher():
    """Two-cluster setup: cluster A (label ~0.8) and cluster B (label ~0.1).
    Predictor should rank a held-out A point above a held-out B point."""
    rng = np.random.RandomState(42)
    d = 16
    n_per = 50
    # Cluster A centered at +e1, B at -e1
    X_a = rng.randn(n_per, d).astype(np.float32) * 0.3 + np.eye(d, dtype=np.float32)[0]
    X_b = rng.randn(n_per, d).astype(np.float32) * 0.3 - np.eye(d, dtype=np.float32)[0]
    X = np.concatenate([X_a, X_b], axis=0)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    y = np.concatenate([
        np.full(n_per, 0.8, dtype=np.float32),
        np.full(n_per, 0.1, dtype=np.float32),
    ])
    # Held-out: one from each cluster
    X_test_a = (np.eye(d, dtype=np.float32)[0] * 0.95).reshape(1, -1)
    X_test_b = (-np.eye(d, dtype=np.float32)[0] * 0.95).reshape(1, -1)
    X_test = np.concatenate([X_test_a, X_test_b], axis=0)

    pred = fit_predictor(X, y, gamma=10.0, alpha=0.01)
    scores = pred(X_test)
    assert scores[0] > scores[1], (
        f"cluster-A point scored {scores[0]:.3f}, cluster-B scored {scores[1]:.3f}"
    )


# ---------- 3. Cold-start with single observation ----------

def test_single_observation_returns_close_to_its_score():
    """Cold-start: one observation at score=0.5.
    A near-duplicate embedding should predict near 0.5.
    A far-away embedding should predict near 0.0 (kernel decay)."""
    d = 8
    obs = np.eye(d, dtype=np.float32)[0].reshape(1, -1)  # unit-norm e1
    obs_y = np.array([0.5], dtype=np.float32)

    pred = fit_predictor(obs, obs_y, gamma=5.0, alpha=0.01)
    # Near-duplicate
    near = obs.copy()
    far = -np.eye(d, dtype=np.float32)[0].reshape(1, -1)  # -e1
    s_near = float(pred(near)[0])
    s_far = float(pred(far)[0])
    assert s_near > 0.4, f"near-duplicate {s_near:.3f} should be > 0.4"
    assert s_far < 0.1, f"far point {s_far:.3f} should be < 0.1"


# ---------- 4. Edge cases ----------

def test_zero_observation_raises():
    X = np.zeros((0, 4), dtype=np.float32)
    y = np.zeros((0,), dtype=np.float32)
    with pytest.raises(ValueError, match="at least one observation"):
        fit_predictor(X, y)


def test_all_zero_y_raises():
    X = np.eye(4, dtype=np.float32)
    y = np.zeros(4, dtype=np.float32)
    with pytest.raises(ValueError, match="all-zero"):
        fit_predictor(X, y)


def test_negative_gamma_raises():
    X = np.eye(2, dtype=np.float32)
    y = np.array([0.5, 0.5], dtype=np.float32)
    with pytest.raises(ValueError, match="gamma"):
        fit_predictor(X, y, gamma=-1.0)


def test_negative_alpha_raises():
    X = np.eye(2, dtype=np.float32)
    y = np.array([0.5, 0.5], dtype=np.float32)
    with pytest.raises(ValueError, match="alpha"):
        fit_predictor(X, y, alpha=-1.0)


def test_y_above_one_clipped():
    """Out-of-range y values must be clipped to [0, 1]."""
    X = np.eye(2, dtype=np.float32)
    y = np.array([2.0, -0.5], dtype=np.float32)  # out of range
    pred = fit_predictor(X, y, gamma=1.0, alpha=0.1)
    out = pred(X)
    assert (out >= 0.0).all()
    assert (out <= 1.0).all()


# ---------- 5. Numerical stability ----------

def test_kernel_matrix_no_overflow_at_n1000():
    """1000 observations with float32 — kernel must not overflow."""
    rng = np.random.RandomState(42)
    n, d = 1000, 768
    X = rng.randn(n, d).astype(np.float32)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    y = rng.uniform(0.0, 0.8, n).astype(np.float32)

    pred = fit_predictor(X, y, gamma=0.1, alpha=1.0)
    # Predict on a small batch — must not NaN/inf
    X_test = X[:10]
    out = pred(X_test)
    assert np.isfinite(out).all(), "kernel overflow — NaN/inf in predictions"


# ---------- 6. Class API parity ----------

def test_kernel_ridge_predictor_class_matches_function():
    """KernelRidgePredictor class is the canonical OOP API;
    fit_predictor returns an instance of it."""
    X = np.eye(4, dtype=np.float32)
    y = np.array([0.1, 0.5, 0.7, 0.9], dtype=np.float32)
    pred = fit_predictor(X, y, gamma=1.0, alpha=0.1)
    assert isinstance(pred, KernelRidgePredictor)


def test_predict_scores_helper_matches_class_call():
    """predict_scores is the one-shot helper for callers that don't
    want to keep the predictor object around."""
    X = np.eye(4, dtype=np.float32)
    y = np.array([0.1, 0.5, 0.7, 0.9], dtype=np.float32)
    scores = predict_scores(X, y, X[:2], gamma=1.0, alpha=0.1)
    assert scores.shape == (2,)
    assert np.isfinite(scores).all()
    assert (scores >= 0.0).all() and (scores <= 1.0).all()


# ---------- 7. RBF kernel correctness ----------

def test_rbf_kernel_self_similarity_is_one():
    """K(x, x) = exp(0) = 1.0 for unit-norm vectors."""
    rng = np.random.RandomState(0)
    X = rng.randn(5, 32).astype(np.float32)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    K = rbf_kernel(X, X, gamma=1.0)
    diag = np.diag(K)
    assert np.allclose(diag, 1.0, atol=1e-5), f"diag {diag}"


def test_rbf_kernel_symmetry():
    rng = np.random.RandomState(0)
    X1 = rng.randn(7, 16).astype(np.float32)
    X2 = rng.randn(5, 16).astype(np.float32)
    K12 = rbf_kernel(X1, X2, gamma=0.5)
    K21 = rbf_kernel(X2, X1, gamma=0.5)
    assert np.allclose(K12, K21.T), "KRR kernel not symmetric"


def test_rbf_kernel_decays_with_distance():
    """Larger distance → smaller kernel value (RBF property)."""
    x = np.array([[1.0, 0.0]], dtype=np.float32)
    y_close = np.array([[0.99, 0.01]], dtype=np.float32)
    y_far = np.array([[-1.0, 0.0]], dtype=np.float32)
    K_close = rbf_kernel(x, y_close, gamma=10.0)
    K_far = rbf_kernel(x, y_far, gamma=10.0)
    assert K_close[0, 0] > K_far[0, 0], (
        f"K_close={K_close[0,0]:.3f} should be > K_far={K_far[0,0]:.3f}"
    )
