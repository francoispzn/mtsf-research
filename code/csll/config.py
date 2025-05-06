"""Default hyper-parameters. Kept modest and *equal-budget* across models for fair
comparison; anchored on each method's published settings. Dataset-aware only for
batch size, look-back/horizon feasibility, and CSLL's low-rank switch on high-dim data."""
from __future__ import annotations

from typing import Dict, List, Optional

from .data import SEASONAL_PERIOD

HIGH_N = {"electricity", "traffic", "metr_la", "pems_bay", "pems04", "pems08"}
LOWDIM_FOR_VAR = {"ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "exchange", "ili", "synthetic",
                  "ili_japan", "ili_us_hhs", "ili_us_state"}
ST_TRAFFIC = {"metr_la", "pems_bay", "pems04", "pems08"}     # 5-min spatio-temporal sets
EPI = {"ili_japan", "ili_us_hhs", "ili_us_state"}            # weekly regional ILI


def horizons_for(dataset: str) -> List[int]:
    if dataset == "ili":
        return [24, 36, 48, 60]
    if dataset in ST_TRAFFIC:
        # 1h / 2h / 4h / 8h at 5-min sampling; propagation delays live at these scales
        return [12, 24, 48, 96]
    if dataset in EPI:
        # weekly: 2/4/8 weeks — inter-region spread lags (1-3 wk) straddle these horizons,
        # putting H=2 inside and H=8 outside the tau >= h envelope
        return [2, 4, 8]
    return [96, 192, 336, 720]


def seq_len_for(dataset: str) -> int:
    if dataset == "ili":
        return 36
    if dataset in EPI:
        return 36                                            # ~8.3 months of weekly context
    return 96


def model_config(model: str, dataset: str, seq_len: int, pred_len: int, n_vars: int,
                 overrides: Optional[Dict] = None) -> Dict:
    cfg = dict(seq_len=seq_len, pred_len=pred_len, n_vars=n_vars)
    if model == "CSLL":
        cfg.update(n_bands=4, use_phase=True, dynamic=True, free_complex=False,
                   low_rank=(16 if n_vars > 64 else 0), kernel_size=25,
                   use_revin=True, tau_max=seq_len / 2.0)
    elif model == "CSLLX":            # strong iTransformer backbone + lead-lag module
        cfg.update(backbone="itransformer", d_model=128, nhead=8, num_layers=3,
                   dim_ff=256, dropout=0.1, n_bands=4, use_phase=True, dynamic=True,
                   free_complex=False, low_rank=(16 if n_vars > 64 else 0),
                   use_revin=True, tau_max=seq_len / 2.0)
    elif model == "CSLL2":            # v2: fixed bound, linear shift, warm gate, direct forecast
        cfg.update(n_bands=4, use_phase=True, dynamic=True, free_complex=False,
                   low_rank=(16 if n_vars > 64 else 0), kernel_size=25,
                   use_revin=True, tau_max=seq_len / 2.0,
                   strict_bound=True, pad2x=True, gate_init=0.05, direct=True)
    elif model == "CSLL2H":           # hybrid: direct-phase + real-mixing branches, each gated
        cfg.update(n_bands=4, use_phase=True, dynamic=True, free_complex=False,
                   low_rank=(16 if n_vars > 64 else 0), kernel_size=25,
                   use_revin=True, tau_max=seq_len / 2.0,
                   strict_bound=True, pad2x=True, gate_init=0.05, direct=True, hybrid=True)
    elif model == "CSLL2X":
        cfg.update(backbone="itransformer", d_model=128, nhead=8, num_layers=3,
                   dim_ff=256, dropout=0.1, n_bands=4, use_phase=True, dynamic=True,
                   free_complex=False, low_rank=(16 if n_vars > 64 else 0),
                   use_revin=True, tau_max=seq_len / 2.0,
                   strict_bound=True, pad2x=True, gate_init=0.05, direct=True)
    elif model == "DLinear":
        cfg.update(kernel_size=25)
    elif model == "NLinear":
        pass
    elif model == "LSTM":
        cfg.update(hidden_size=128, num_layers=2, dropout=0.1, use_revin=True)
    elif model == "Transformer":
        cfg.update(d_model=64, nhead=4, num_layers=2, dim_ff=128, dropout=0.1, use_revin=True)
    elif model == "PatchTST":
        cfg.update(patch_len=16, stride=8, d_model=64, nhead=4, num_layers=2,
                   dim_ff=128, dropout=0.1, use_revin=True)
    elif model == "iTransformer":
        cfg.update(d_model=128, nhead=8, num_layers=3, dim_ff=256, dropout=0.1, use_revin=True)
    elif model == "SeasonalNaive":
        cfg.update(period=SEASONAL_PERIOD.get(dataset, 24))
    if overrides:
        cfg.update(overrides)
    return cfg


def train_config(model: str, dataset: str, n_vars: int, overrides: Optional[Dict] = None) -> Dict:
    bs = 32
    if dataset == "traffic":
        bs = 8
    elif dataset == "electricity" or dataset in ST_TRAFFIC:
        bs = 16                                   # high-dim sets: memory safety
    elif dataset in ("ETTm1", "ETTm2", "weather"):
        bs = 64                                   # larger batch on the bigger medium sets
    lr = 3e-4 if model in ("Transformer", "PatchTST", "iTransformer", "CSLLX", "CSLL2X") else 1e-3
    _ = model  # CSLL2H uses the CSLL2 lr (1e-3)
    cfg = dict(lr=lr, batch_size=bs, epochs=10, patience=3, grad_clip=5.0,
               weight_decay=0.0, num_workers=0)
    if model in ("CSLL", "CSLLX", "CSLL2", "CSLL2X"):
        # No weight decay: even 1e-4 stalls the branch before it engages on lead-lag data
        # (e.g. Traffic) at the training budget used here. The alpha-gate provides graceful
        # degradation; residual mild over-fit on near-independent series is reported honestly.
        cfg["branch_wd"] = 0.0
    if model in ("CSLL2", "CSLL2X", "CSLL2H"):
        # v2: correlation-based delay initialisation (classic TDE remedy for the oscillatory
        # loss surface in D). Network datasets use the pairwise LS mode (corridor-local lags);
        # shared-source-style data uses the global two-pass mode. The v2 CAMPAIGN raises
        # epochs to 30/patience 5 for ALL models uniformly via train_over; default stays 10/3.
        cfg["xcorr_init"] = "pairwise" if dataset in (ST_TRAFFIC | EPI) else True
        cfg["gate_select"] = False    # honest gate: no undisclosed val-based branch zeroing (v1 M-5)
    if overrides:
        cfg.update(overrides)
    return cfg
