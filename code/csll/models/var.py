"""Vector Autoregression baseline (statsmodels). Non-torch; fit once per dataset and used
to forecast each sliding window. Only run on low-dimensional datasets (N<=21) where VAR is
tractable/estimable (standard practice)."""
from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np


class VARForecaster:
    def __init__(self, maxlags: int = 24):
        self.maxlags = maxlags
        self.results = None
        self.k_ar = 1

    def fit(self, train_series: np.ndarray) -> "VARForecaster":
        from statsmodels.tsa.api import VAR

        # Keep VAR estimable AND stable for long-horizon recursion: cap lags tightly and use
        # BIC (parsimonious). If the fit is not stable (explosive roots -> overflow over the
        # horizon), fall back to smaller orders until stable.
        T, N = train_series.shape
        maxlags = int(max(1, min(self.maxlags, 8, T // (2 * (N + 1)))))
        model = VAR(train_series.astype(np.float64))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                res = model.fit(maxlags=maxlags, ic="bic")
                if res.k_ar < 1:
                    res = model.fit(1)
            except Exception:
                res = model.fit(1)
            # stability fallback
            for p in range(res.k_ar, 0, -1):
                try:
                    cand = res if p == res.k_ar else model.fit(p)
                    stable = cand.is_stable(verbose=False) if hasattr(cand, "is_stable") else True
                    if stable:
                        res = cand
                        break
                    res = cand
                except Exception:
                    break
        self.results = res
        self.k_ar = max(1, res.k_ar)
        return self

    def predict_over(self, series: np.ndarray, seq_len: int, pred_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """Forecast every sliding window over a standardised slice, vectorised across windows.

        Uses the fitted VAR(p) recursion  y_t = c + sum_l A_l y_{t-l}  directly (much faster
        than statsmodels.forecast per window). Returns (preds, trues), each (num, pred_len, N).
        """
        assert self.results is not None, "call fit() first"
        p = self.k_ar
        coefs = np.asarray(self.results.coefs, dtype=np.float64)      # (p, N, N): A_1..A_p
        intercept = np.asarray(self.results.intercept, dtype=np.float64)  # (N,)
        T, N = series.shape
        num = T - seq_len - pred_len + 1
        if num <= 0:
            return (np.zeros((0, pred_len, N), np.float32),) * 2

        # last-p history for each window: (num, p, N), most recent at index -1
        hist = np.stack([series[i + seq_len - p: i + seq_len] for i in range(num)], axis=0).astype(np.float64)
        cur = hist.copy()
        out = np.empty((num, pred_len, N), dtype=np.float32)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for step in range(pred_len):
                yhat = np.repeat(intercept[None, :], num, axis=0)         # (num, N)
                for l in range(p):
                    yhat += cur[:, -1 - l, :] @ coefs[l].T                # y_{t-1-l} A_{l+1}^T
                # numerical safeguard against explosive roots over long horizons
                yhat = np.clip(np.nan_to_num(yhat, nan=0.0, posinf=1e3, neginf=-1e3), -1e3, 1e3)
                out[:, step, :] = yhat.astype(np.float32)
                cur = np.concatenate([cur[:, 1:, :], yhat[:, None, :]], axis=1)
        trues = np.stack([series[i + seq_len: i + seq_len + pred_len] for i in range(num)], axis=0).astype(np.float32)
        return out, trues
