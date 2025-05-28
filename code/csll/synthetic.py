"""Synthetic multivariate series with PLANTED lead-lag structure.

Each channel is a scaled, delayed copy of a shared latent source plus noise, with KNOWN
integer delays tau_i. The ground-truth pairwise delay is D_{ij} = tau_i - tau_j. Used ONLY
as an interpretability diagnostic: does CSLL's learned delay matrix recover the planted
lags? Never used in headline benchmark tables.

We also save it to a CSV compatible with the data pipeline (as dataset name 'synthetic').
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def make_synthetic(T: int = 8000, n_vars: int = 8, seed: int = 0,
                   max_delay: int = 20, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Return (data (T,N) float32, delays tau (N,) int)."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    # latent source: mixture of sinusoids (multi-frequency) + slow AR drift
    source = (np.sin(2 * np.pi * t / 24)
              + 0.6 * np.sin(2 * np.pi * t / 168 + 0.7)
              + 0.4 * np.sin(2 * np.pi * t / 12 + 1.3))
    ar = np.zeros(T)
    for i in range(1, T):
        ar[i] = 0.95 * ar[i - 1] + rng.normal(0, 0.3)
    source = source + 0.5 * ar
    source = (source - source.mean()) / (source.std() + 1e-8)

    # per-channel planted integer delays and gains
    taus = rng.integers(0, max_delay + 1, size=n_vars)
    taus[0] = 0  # anchor
    gains = rng.uniform(0.7, 1.3, size=n_vars) * rng.choice([-1.0, 1.0], size=n_vars)

    data = np.empty((T, n_vars), dtype=np.float32)
    for i in range(n_vars):
        shifted = np.roll(source, int(taus[i]))          # circular delay (matches model)
        data[:, i] = gains[i] * shifted + rng.normal(0, noise, size=T)
    return data.astype(np.float32), taus.astype(np.int64)


def write_synthetic_csv(path: Path, T: int = 8000, n_vars: int = 8, seed: int = 0,
                        max_delay: int = 20) -> np.ndarray:
    """Generate and save as CSV (date + N cols, last named OT). Returns delays tau."""
    data, taus = make_synthetic(T=T, n_vars=n_vars, seed=seed, max_delay=max_delay)
    cols = [str(i) for i in range(n_vars - 1)] + ["OT"]
    df = pd.DataFrame(data, columns=cols)
    dates = pd.date_range("2016-07-01", periods=T, freq="H")
    df.insert(0, "date", dates)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    np.save(path.with_suffix(".delays.npy"), taus)
    return taus


# ---------------------------------------------------------------------------
# Controlled "value-envelope" regimes (used by scripts/controlled_experiments.py)
# ---------------------------------------------------------------------------
def make_leadfollower(T: int = 12000, n_follow: int = 6, seed: int = 0, noise: float = 0.05,
                      tau_lo: int = 24, tau_hi: int = 72, block: int = 0):
    """Leader--follower regime where cross-series lead--lag is *essential* for forecasting.

    Channel 0 is a WHITE-NOISE leader/source s(t) (not self-predictable). Each follower i is the
    leader delayed by tau_i in [tau_lo, tau_hi]:  x_{i}(t) = s(t - tau_i) + noise. With a look-back
    L>=tau_hi and a horizon H<=tau_lo, every future value of a follower equals a leader value that
    lies *inside the look-back window* -- so only a model that reads the correct per-channel DELAY
    from another series can forecast it. A channel-independent model cannot (white noise), and a
    real-valued same-time mixer cannot (a delay is needed). This isolates the phase mechanism.

    If block>0, the delay vector SWITCHES between two regimes every `block` steps (non-stationary
    lead--lag) so that input-adaptive *dynamic* delays can be probed against static ones.

    Returns (data (T,N) float32, tauA (n_follow,) int, tauB (n_follow,) int or None).
    """
    rng = np.random.default_rng(seed)
    pad = tau_hi + 8
    s = rng.normal(0.0, 1.0, size=T + pad)
    tauA = rng.integers(tau_lo, tau_hi + 1, size=n_follow)
    tauB = rng.integers(tau_lo, tau_hi + 1, size=n_follow) if block > 0 else None
    N = n_follow + 1
    data = np.zeros((T, N), dtype=np.float32)
    data[:, 0] = s[pad:pad + T]                       # leader
    for k in range(T):
        tau = tauA if (block <= 0 or (k // block) % 2 == 0) else tauB
        for i in range(n_follow):
            data[k, i + 1] = s[pad + k - int(tau[i])] + rng.normal(0.0, noise)
    return data.astype(np.float32), tauA.astype(np.int64), (None if tauB is None else tauB.astype(np.int64))
