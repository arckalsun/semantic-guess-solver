"""Kernel Ridge Regression predictor for the GuessWord oracle.

Round-4 promised predictor (case-3 skill §4.1, case-11 skill §3.6).

Background
==========

The sgs v0.6.0 ``fit_centroid`` heuristic was the *only* candidate
ranking function: it computed a score-weighted mean of observed
embeddings and ranked candidates by cosine similarity to that mean.
This works when the answer is well-defined within a single semantic
cluster, but **fails when the embedding space used by the ranker
(BGE-zh-base) is misaligned with the embedding space used by the
oracle** (xiaoce.fun's private model).

In case-11 (shareId 377580436223), this manifested as a 0.82 plateau:
``通过`` (pass) peaked at 0.8245 across 632 brute-force probes, but
no corpus word scored above the 0.95 ``correct=true`` threshold. The
centroid couldn't escape the abstract-verb cluster because BGE-zh-base
places ``继续/停止/结束/决定/执行`` far from ``通过/履行/允许`` even
though xiaoce's private embedding treats them as near-neighbors.

Solution
========

Fit a **Kernel Ridge Regression** (KRR) model that learns the
BGE → xiaoce doubleScore mapping from observed (word, score) pairs.
Once fit, the model predicts a *continuous* score for every
candidate in the corpus. This bypasses the embedding-space
mismatch because the kernel matrix is computed in BGE space (where
both observed and candidate words live) but the regression weights
are learned to match xiaoce's score space.

Empirical result (case-11 R14, 9 probes after 632 brute-force):
    继续  predicted=0.5576  actual=0.9890  correct=true  ★SOLVED★

API
===

::

    pred = fit_predictor(X_obs, y_obs, gamma=0.1, alpha=0.1)
    scores = pred(X_candidates)  # (N,) float32 in [0, 1]

    # Or one-shot:
    scores = predict_scores(X_obs, y_obs, X_candidates,
                             gamma=0.1, alpha=0.1)

Hyperparameters
===============

``gamma`` (RBF kernel width): controls how local vs global the fit is.
    Small gamma (e.g. 0.01) → almost linear, slow kernel decay.
    Large gamma (e.g. 10.0) → very local, sharp boundaries.

``alpha`` (regularization): ridge penalty on the dual coefficients.
    Small alpha (e.g. 0.01) → low bias, risk of overfit.
    Large alpha (e.g. 10.0) → high bias, smoother.

Defaults chosen from case-11 5-fold CV: ``gamma=0.1, alpha=0.1``
gave MAE=0.0992 across hyperparameter grid {0.01, 0.1, 1.0, 10.0}^2.

Why these specific defaults work for xiaoce
============================================

xiaoce scores live in [0, 1]. The BGE-zh-base cosine-similarity
distribution is approximately Gaussian-centered with std ≈ 0.1-0.2
between semantically-related words. ``gamma=0.1`` makes the kernel
exp(-0.1 * sq_dist) decay over distance scale ≈ 1.0 (matches BGE's
natural scale), and ``alpha=0.1`` provides mild regularization
without washing out the signal.

Numerical notes
===============

* Kernel matrix K is (n, n) for n observations. For n > 5000, the
  Cholesky solve becomes the bottleneck (~1s on CPU). Acceptable
  for sgs's use case where the corpus is small (~2000 words) and
  observation history is at most a few hundred by the time we hit
  the 0.82 plateau.
* ``alpha * I`` regularizer is added before solve — guards against
  singular K when observations are linearly dependent in RBF space.
* y is clipped to [0, 1] before fit (defensive — protects against
  malformed replay records that contain score > 1.0).
* Predictions are clipped to [0, 1] after solve (RBF can extrapolate).
"""

from __future__ import annotations

import logging

import numpy as np

__all__ = [
    "KernelRidgePredictor",
    "fit_predictor",
    "predict_scores",
    "rbf_kernel",
]


logger = logging.getLogger("sgs.krr")


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


def rbf_kernel(X1: np.ndarray, X2: np.ndarray, *, gamma: float) -> np.ndarray:
    """Radial basis function (Gaussian) kernel.

    K[i, j] = exp(-gamma * ||X1[i] - X2[j]||^2)

    Uses the identity ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x.y when the
    inputs are unit-norm (which is sgs's corpus invariant). Falls
    back gracefully when inputs are not unit-norm.

    Parameters
    ----------
    X1, X2 : (n1, D), (n2, D) float32 arrays
    gamma : float
        Kernel width. Must be > 0.

    Returns
    -------
    K : (n1, n2) float32 array, values in (0, 1].
    """
    if gamma <= 0.0:
        raise ValueError(f"gamma must be > 0, got {gamma!r}")
    # Promote to float64 for the squared-distance intermediate —
    # float32 underflows when X has 768 dims and ||x||^2 ≈ 1
    X1f = X1.astype(np.float64, copy=False)
    X2f = X2.astype(np.float64, copy=False)
    sq1 = (X1f * X1f).sum(axis=1)[:, None]  # (n1, 1)
    sq2 = (X2f * X2f).sum(axis=1)[None, :]  # (1, n2)
    cross = X1f @ X2f.T
    sq_dist = sq1 + sq2 - 2.0 * cross
    # Numerical floor — sq_dist can dip slightly below 0 due to
    # cancellation in the cross-product step for unit-norm inputs.
    sq_dist = np.maximum(sq_dist, 0.0)
    K = np.exp(-gamma * sq_dist)
    return K.astype(np.float32)


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------


class KernelRidgePredictor:
    """Fitted kernel ridge regression model.

    Closed-form solution: α = (K + α·I)^-1 y, predict K_test @ α.

    The predictor is callable: ``predictor(X_test) → (N,) float32``.
    Internally stores the training embeddings (X_train) and dual
    coefficients (dual_coef_) so prediction is just a kernel
    evaluation against the support set.

    Attributes
    ----------
    X_train_ : (n_obs, D) float32
    dual_coef_ : (n_obs,) float32
    gamma : float
    alpha : float
    """

    def __init__(
        self,
        X_train: np.ndarray,
        dual_coef: np.ndarray,
        *,
        gamma: float,
        alpha: float,
    ) -> None:
        self.X_train_ = np.ascontiguousarray(X_train, dtype=np.float32)
        self.dual_coef_ = np.asarray(dual_coef, dtype=np.float32)
        self.gamma = gamma
        self.alpha = alpha

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Predict scores for X.

        Parameters
        ----------
        X : (N, D) float32

        Returns
        -------
        scores : (N,) float32 in [0, 1]
        """
        K = rbf_kernel(X, self.X_train_, gamma=self.gamma)
        scores = K @ self.dual_coef_
        return np.clip(scores, 0.0, 1.0).astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Alias for __call__ — matches scikit-learn naming."""
        return self(X)


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


def fit_predictor(
    X_obs: np.ndarray,
    y_obs: np.ndarray,
    *,
    gamma: float = 0.1,
    alpha: float = 0.1,
) -> KernelRidgePredictor:
    """Fit a kernel ridge regression model.

    Parameters
    ----------
    X_obs : (n_obs, D) float32 — observation embeddings
    y_obs : (n_obs,) float-like — observation scores in [0, 1]
    gamma : float — RBF kernel width (default 0.1)
    alpha : float — ridge regularization (default 0.1)

    Returns
    -------
    KernelRidgePredictor — call it on candidate embeddings to get
    predicted scores.

    Raises
    ------
    ValueError
        If X_obs is empty, y_obs is empty, all y_obs are zero,
        gamma is non-positive, or alpha is negative.
    """
    X = np.asarray(X_obs, dtype=np.float32)
    y = np.asarray(y_obs, dtype=np.float64)

    if X.size == 0 or y.size == 0:
        raise ValueError("fit_predictor: at least one observation required")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X_obs ({X.shape[0]} rows) and y_obs ({y.shape[0]} entries) disagree"
        )
    if gamma <= 0.0:
        raise ValueError(f"gamma must be > 0, got {gamma!r}")
    if alpha < 0.0:
        raise ValueError(f"alpha must be >= 0, got {alpha!r}")

    # Clip y to [0, 1] — defends against malformed replay records.
    y_clipped = np.clip(y, 0.0, 1.0)
    if not np.any(y_clipped > 0.0):
        raise ValueError("fit_predictor: all-zero y — nothing to fit")

    # Build kernel matrix and solve (K + α·I) α_dual = y
    K = rbf_kernel(X, X, gamma=gamma)
    n = K.shape[0]
    # Symmetrize to fight float32 asymmetry in rbf_kernel
    K = 0.5 * (K + K.T)
    # Regularize
    A = K.astype(np.float64) + float(alpha) * np.eye(n, dtype=np.float64)
    # Solve in float64 for numerical stability.
    # We use np.linalg.solve directly — cho_solve lives in scipy.linalg
    # and sgs is pure-stdlib + numpy, so we can't depend on scipy.
    # np.linalg.solve with a symmetric positive-definite matrix is
    # just as fast as Cholesky + cho_solve for the sizes we deal with
    # (n_obs ≤ ~1000 in practice).
    try:
        dual_coef = np.linalg.solve(A, y_clipped.astype(np.float64))
    except np.linalg.LinAlgError:
        # Fall back to least-squares if direct solve fails (rare — only
        # when K is degenerate under extreme gamma/alpha combos).
        logger.warning("np.linalg.solve failed — falling back to lstsq")
        dual_coef, *_ = np.linalg.lstsq(A, y_clipped.astype(np.float64), rcond=None)

    return KernelRidgePredictor(
        X_train=X,
        dual_coef=dual_coef.astype(np.float32),
        gamma=gamma,
        alpha=alpha,
    )


def predict_scores(
    X_obs: np.ndarray,
    y_obs: np.ndarray,
    X_candidates: np.ndarray,
    *,
    gamma: float = 0.1,
    alpha: float = 0.1,
) -> np.ndarray:
    """One-shot helper: fit predictor, predict scores for candidates.

    Equivalent to::

        pred = fit_predictor(X_obs, y_obs, gamma=gamma, alpha=alpha)
        return pred(X_candidates)

    Provided for callers that don't need to keep the predictor object
    alive (e.g. CLI batch runners).
    """
    pred = fit_predictor(X_obs, y_obs, gamma=gamma, alpha=alpha)
    return pred(X_candidates)
