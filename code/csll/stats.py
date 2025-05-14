"""Statistical significance tests for forecast comparisons."""
from __future__ import annotations

from typing import Tuple

import numpy as np


def diebold_mariano(errors_a: np.ndarray, errors_b: np.ndarray, h: int = 1,
                    power: int = 2) -> Tuple[float, float]:
    """Diebold-Mariano test on two models' per-observation errors.

    errors_* : arrays of per-sample forecast errors (any shape); flattened.
    Returns (DM statistic, two-sided p-value). Uses a Newey-West (h-1 lag) HAC variance
    with the Harvey-Leybourne-Newbold small-sample correction. H0: equal predictive accuracy.
    """
    from scipy import stats as sps

    ea = np.asarray(errors_a).reshape(-1)
    eb = np.asarray(errors_b).reshape(-1)
    n = min(ea.shape[0], eb.shape[0])
    ea, eb = ea[:n], eb[:n]
    d = np.abs(ea) ** power - np.abs(eb) ** power
    dbar = d.mean()
    # HAC variance with (h-1) lags
    gamma0 = np.mean((d - dbar) ** 2)
    var = gamma0
    for lag in range(1, h):
        cov = np.mean((d[lag:] - dbar) * (d[:-lag] - dbar))
        var += 2.0 * (1.0 - lag / h) * cov
    var = var / n
    if var <= 0:
        return 0.0, 1.0
    dm = dbar / np.sqrt(var)
    # Harvey-Leybourne-Newbold correction
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm *= corr
    p = 2.0 * (1.0 - sps.t.cdf(abs(dm), df=n - 1))
    return float(dm), float(p)


def paired_ttest(scores_a: np.ndarray, scores_b: np.ndarray) -> Tuple[float, float]:
    """Paired t-test across seeds/runs. Returns (t-stat, two-sided p-value)."""
    from scipy import stats as sps

    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape[0] < 2:
        return float("nan"), float("nan")
    t, p = sps.ttest_rel(a, b)
    return float(t), float(p)
