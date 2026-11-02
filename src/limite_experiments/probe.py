"""Linear probe on hidden states: standardised ridge regression and problem-grouped folds."""

from __future__ import annotations

import numpy as np


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float):
    """Standardised ridge regression on {0,1} labels (only the ranking of its predictions is used)."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    A = Z.T @ Z + lam * len(Z) * np.eye(Z.shape[1])
    w = np.linalg.solve(A, Z.T @ (y - y.mean()))
    return {"mu": mu, "sd": sd, "w": w, "b": float(y.mean())}


def ridge_predict(model, X: np.ndarray) -> np.ndarray:
    return ((X - model["mu"]) / model["sd"]) @ model["w"] + model["b"]


def grouped_folds(groups: list[str], k: int = 5, seed: int = 0) -> list[np.ndarray]:
    """k folds with every group (problem) in exactly one fold."""
    uniq = sorted(set(groups))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    fold_of = {g: i % k for i, g in enumerate(uniq)}
    f = np.array([fold_of[g] for g in groups])
    return [np.where(f == i)[0] for i in range(k)]
