"""Data pipeline for multivariate LTSF benchmarks.

One loader for every model, implementing the *standard* protocol used across the LTSF
literature (Informer/Autoformer/DLinear/PatchTST):
  * multivariate -> multivariate ("M" setting): input all N channels, predict all N.
  * chronological splits: fixed ETT borders; 70/10/20 for the other datasets.
  * z-normalisation with statistics fit on the TRAINING split only (no leakage).
  * sliding windows of (seq_len -> pred_len); the val/test windows may look back into
    the preceding split by `seq_len` (standard border construction).
  * NO drop-last: every full window in the split is evaluated (avoids the TFB pitfall).

Primary error metrics (MSE/MAE/RMSE) are computed on the *standardised* series, matching
the dominant LTSF convention; MASE is computed on the *original* scale (scale-free) using a
train seasonal-naive denominator. See metrics.py.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
DATASET_FILES: Dict[str, str] = {
    "ETTh1": "ETTh1.csv",
    "ETTh2": "ETTh2.csv",
    "ETTm1": "ETTm1.csv",
    "ETTm2": "ETTm2.csv",
    "weather": "weather.csv",
    "exchange": "exchange_rate.csv",
    "ili": "national_illness.csv",
    "electricity": "electricity.csv",
    "traffic": "traffic.csv",
    "synthetic": "synthetic.csv",
    # spatio-temporal traffic (5-min sampling; propagation delays are representable,
    # unlike the hourly 'traffic' benchmark where they are sub-sample)
    "metr_la": "metr_la.csv",
    "pems_bay": "pems_bay.csv",
    "pems04": "pems04.csv",
    "pems08": "pems08.csv",
    # regional weekly ILI (Cola-GNN suite): candidate tau >= h regime — inter-region
    # epidemic spread lags (1-3 weeks) are comparable to the forecast horizons
    "ili_japan": "ili_japan.csv",
    "ili_us_hhs": "ili_us_hhs.csv",
    "ili_us_state": "ili_us_state.csv",
}

# split scheme: 'ett_h' / 'ett_m' use fixed calendar borders; 'custom' uses 70/10/20.
SPLIT_TYPE: Dict[str, str] = {
    "ETTh1": "ett_h", "ETTh2": "ett_h",
    "ETTm1": "ett_m", "ETTm2": "ett_m",
    "weather": "custom", "exchange": "custom", "ili": "custom",
    "electricity": "custom", "traffic": "custom", "synthetic": "custom",
    "metr_la": "custom", "pems_bay": "custom", "pems04": "custom", "pems08": "custom",
    "ili_japan": "custom", "ili_us_hhs": "custom", "ili_us_state": "custom",
}

# seasonal period m (in steps) for the MASE seasonal-naive denominator.
SEASONAL_PERIOD: Dict[str, int] = {
    "ETTh1": 24, "ETTh2": 24, "ETTm1": 96, "ETTm2": 96,
    "weather": 144, "exchange": 1, "ili": 52,
    "electricity": 24, "traffic": 24, "synthetic": 24,
    "metr_la": 288, "pems_bay": 288, "pems04": 288, "pems08": 288,   # daily @ 5-min
    "ili_japan": 52, "ili_us_hhs": 52, "ili_us_state": 52,           # annual @ weekly
}


def data_root() -> Path:
    """Locate mtsf-research/data/raw regardless of CWD (env override supported)."""
    env = os.environ.get("MTSF_DATA_ROOT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "data" / "raw"


def load_matrix(name: str) -> np.ndarray:
    """Load a dataset as a float32 array of shape (T, N) (date column dropped)."""
    if name not in DATASET_FILES:
        raise KeyError(f"unknown dataset '{name}'. Known: {list(DATASET_FILES)}")
    path = data_root() / DATASET_FILES[name]
    df = pd.read_csv(path)
    if df.columns[0].lower() == "date" or df.columns[0].lower().startswith("date"):
        df = df.iloc[:, 1:]
    return df.to_numpy(dtype=np.float32)


def split_indices(name: str, n: int, seq_len: int) -> Dict[str, Tuple[int, int]]:
    """Return (start, end) row indices for train/val/test raw slices.

    end indices are exclusive. val/test starts are shifted back by seq_len so the first
    window of a split can see the required look-back (standard LTSF border construction).
    """
    st = SPLIT_TYPE[name]
    if st == "ett_h":
        n_train, n_val, n_test = 12 * 30 * 24, 4 * 30 * 24, 4 * 30 * 24
    elif st == "ett_m":
        n_train, n_val, n_test = 12 * 30 * 24 * 4, 4 * 30 * 24 * 4, 4 * 30 * 24 * 4
    else:  # custom 70/10/20
        n_train = int(n * 0.7)
        n_test = int(n * 0.2)
        n_val = n - n_train - n_test
    border1 = {"train": 0, "val": n_train - seq_len, "test": n_train + n_val - seq_len}
    border2 = {"train": n_train, "val": n_train + n_val, "test": n_train + n_val + n_test}
    # guard against tiny datasets
    for k in border1:
        border1[k] = max(0, border1[k])
        border2[k] = min(n, border2[k])
    return {k: (border1[k], border2[k]) for k in ["train", "val", "test"]}


class WindowDataset(Dataset):
    """Sliding-window (seq_len -> pred_len) samples over a standardised slice."""

    def __init__(self, series: np.ndarray, seq_len: int, pred_len: int):
        # series: (T_slice, N) already standardised
        self.series = torch.from_numpy(np.ascontiguousarray(series)).float()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_samples = max(0, series.shape[0] - seq_len - pred_len + 1)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        s = idx
        m = s + self.seq_len
        e = m + self.pred_len
        x = self.series[s:m]          # (seq_len, N)
        y = self.series[m:e]          # (pred_len, N)
        return x, y


@dataclass
class DataBundle:
    name: str
    n_vars: int
    seq_len: int
    pred_len: int
    train: WindowDataset
    val: WindowDataset
    test: WindowDataset
    mean: np.ndarray                      # (N,) train mean (original scale)
    std: np.ndarray                       # (N,) train std
    mase_scale: np.ndarray                # (N,) train seasonal-naive MAE (original scale)
    seasonal_period: int = 1

    def loaders(self, batch_size: int, num_workers: int = 0) -> Dict[str, DataLoader]:
        common = dict(num_workers=num_workers, drop_last=False, pin_memory=False)
        return {
            "train": DataLoader(self.train, batch_size=batch_size, shuffle=True, **common),
            "val": DataLoader(self.val, batch_size=batch_size, shuffle=False, **common),
            "test": DataLoader(self.test, batch_size=batch_size, shuffle=False, **common),
        }


def build_dataset(name: str, seq_len: int, pred_len: int, train_frac: float = 1.0) -> DataBundle:
    """Load, split, standardise (train stats), and window a dataset.

    train_frac < 1.0 keeps only the most recent `train_frac` of the TRAINING split (val/test
    are untouched), for data-efficiency studies. Standardisation stats are recomputed on the
    kept subset so there is no leakage from the discarded early history.
    """
    mat = load_matrix(name)                       # (T, N)
    n_total, n_vars = mat.shape
    idx = split_indices(name, n_total, seq_len)

    tr0, tr1 = idx["train"]
    if train_frac < 1.0:
        keep = max(seq_len + pred_len + 1, int(round((tr1 - tr0) * train_frac)))
        tr0 = max(idx["train"][0], tr1 - keep)   # keep the most recent slice
        idx = dict(idx, train=(tr0, tr1))
    train_raw = mat[tr0:tr1]                       # (T_train, N)
    mean = train_raw.mean(axis=0)
    std = train_raw.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    mean = mean.astype(np.float32)

    def standardise(a: np.ndarray) -> np.ndarray:
        return (a - mean) / std

    # MASE seasonal-naive denominator on the *original-scale* training series.
    m = SEASONAL_PERIOD[name]
    if train_raw.shape[0] > m:
        naive_err = np.abs(train_raw[m:] - train_raw[:-m]).mean(axis=0)  # (N,)
    else:
        naive_err = np.abs(np.diff(train_raw, axis=0)).mean(axis=0)
    mase_scale = np.where(naive_err < 1e-8, 1.0, naive_err).astype(np.float32)

    slices = {}
    for split, (a, b) in idx.items():
        slices[split] = standardise(mat[a:b]).astype(np.float32)

    return DataBundle(
        name=name, n_vars=n_vars, seq_len=seq_len, pred_len=pred_len,
        train=WindowDataset(slices["train"], seq_len, pred_len),
        val=WindowDataset(slices["val"], seq_len, pred_len),
        test=WindowDataset(slices["test"], seq_len, pred_len),
        mean=mean, std=std, mase_scale=mase_scale, seasonal_period=m,
    )
