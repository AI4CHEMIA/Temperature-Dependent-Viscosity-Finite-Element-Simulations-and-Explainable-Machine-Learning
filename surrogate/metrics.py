"""Error metrics (physical units) and bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ape = np.abs(y_pred - y_true) / np.abs(y_true) * 100.0
    mse = mean_squared_error(y_true, y_pred)
    out = {"n": len(y_true), "R2": r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan,
           "MAE": mean_absolute_error(y_true, y_pred), "MSE": mse, "RMSE": np.sqrt(mse),
           "MAPE_%": ape.mean(), "MedAPE_%": np.median(ape), "MaxAPE_%": ape.max()}
    return out


def bootstrap_ci(y_true, y_pred, n_boot=2000, seed=0, alpha=0.05) -> dict:
    """Percentile bootstrap CI for MAPE, RMSE and R2 (resampling test points)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    n = len(y_true)
    stats = {"MAPE_%": [], "RMSE": [], "R2": []}
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        m = metrics(y_true[i], y_pred[i])
        for k in stats:
            stats[k].append(m[k])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {f"{k}_CI": (np.nanpercentile(v, lo), np.nanpercentile(v, hi)) for k, v in stats.items()}
