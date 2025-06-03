#!/usr/bin/env python3
"""End-to-end smoke test: FFT correctness, model shapes, and a tiny train/eval per model.
Prints per-model metrics + wall-time so we can size the Phase-4 sweep from real numbers."""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from csll import set_seed                                   # noqa: E402
from csll.config import model_config, train_config          # noqa: E402
from csll.data import build_dataset                          # noqa: E402
from csll.evaluate import score_torch, metrics_from_arrays   # noqa: E402
from csll.models import build_model                          # noqa: E402
from csll.models.csll import CSLL                            # noqa: E402
from csll.train import get_device, train_torch, count_params  # noqa: E402


def test_fft_roundtrip():
    print("\n== 1. FFT round-trip & analysis correctness ==")
    ok = True
    for L in (96, 97, 128):
        m = CSLL(seq_len=L, pred_len=48, n_vars=5, use_revin=False)
        x = torch.randn(3, L, 5)
        Xr, Xi = m._rfft(x)
        # compare against numpy rfft
        np_fft = np.fft.rfft(x.numpy(), axis=1)
        err_r = np.abs(Xr.numpy() - np_fft.real).max()
        err_i = np.abs(Xi.numpy() - np_fft.imag).max()
        recon = m._irfft(Xr, Xi)
        err_rt = np.abs(recon.numpy() - x.numpy()).max()
        good = max(err_r, err_i, err_rt) < 1e-3
        ok = ok and good
        print(f"  L={L:4d}  |rfft.re err|={err_r:.2e}  |rfft.im err|={err_i:.2e}  "
              f"|roundtrip err|={err_rt:.2e}  {'OK' if good else 'FAIL'}")
    return ok


def test_shapes():
    print("\n== 2. Model forward shapes (B=4, L=96, H=192, N=7) ==")
    B, L, H, N = 4, 96, 192, 7
    x = torch.randn(B, L, N)
    ok = True
    for name in ["CSLL", "DLinear", "NLinear", "LSTM", "Transformer", "PatchTST",
                 "iTransformer", "SeasonalNaive"]:
        try:
            cfg = model_config(name, "ETTh1", L, H, N)
            net = build_model(name, cfg) if name in (
                "CSLL", "DLinear", "NLinear", "LSTM", "Transformer", "PatchTST", "iTransformer"
            ) else __import__("csll.models.naive", fromlist=["SeasonalNaive"]).SeasonalNaive(**cfg)
            y = net(x)
            good = tuple(y.shape) == (B, H, N)
            ok = ok and good
            print(f"  {name:14s} -> {tuple(y.shape)} params={count_params(net):>8d} "
                  f"{'OK' if good else 'FAIL'}")
        except Exception as e:
            ok = False
            print(f"  {name:14s} FAIL: {e}")
    # low-rank CSLL for high-N
    try:
        cfg = model_config("CSLL", "traffic", L, H, 300)
        net = build_model("CSLL", cfg)
        y = net(torch.randn(2, L, 300))
        print(f"  CSLL(low_rank,N=300) -> {tuple(y.shape)} params={count_params(net)} "
              f"{'OK' if tuple(y.shape)==(2,H,300) else 'FAIL'}")
    except Exception as e:
        ok = False
        print(f"  CSLL(low_rank) FAIL: {e}\n{traceback.format_exc()}")
    return ok


def test_train_eval():
    print("\n== 3. Tiny train/eval on ETTh1 (L=96,H=96,epochs=2) ==")
    device = get_device()
    print(f"  device = {device}")
    bundle = build_dataset("ETTh1", 96, 96)
    print(f"  ETTh1 windows: train={len(bundle.train)} val={len(bundle.val)} test={len(bundle.test)}")
    rows = []
    for name in ["SeasonalNaive", "NLinear", "DLinear", "CSLL", "LSTM", "PatchTST",
                 "iTransformer", "Transformer"]:
        try:
            set_seed(0)
            cfg = model_config(name, "ETTh1", 96, 96, bundle.n_vars)
            if name == "SeasonalNaive":
                from csll.models.naive import SeasonalNaive
                net = SeasonalNaive(**cfg)
            else:
                net = build_model(name, cfg)
            tcfg = train_config(name, "ETTh1", bundle.n_vars, {"epochs": 2, "patience": 2})
            t0 = time.time()
            net, hist = train_torch(net, bundle, device, tcfg)
            loaders = bundle.loaders(batch_size=tcfg["batch_size"])
            metrics, _ = score_torch(net, loaders["test"], device, bundle)
            dt = time.time() - t0
            rows.append((name, metrics["mse"], metrics["mae"], hist.get("params", 0), dt))
            print(f"  {name:14s} MSE={metrics['mse']:.4f} MAE={metrics['mae']:.4f} "
                  f"params={hist.get('params',0):>8d} time={dt:5.1f}s")
        except Exception as e:
            print(f"  {name:14s} FAIL: {e}\n{traceback.format_exc()}")
    # VAR
    try:
        set_seed(0)
        from csll.models.var import VARForecaster
        t0 = time.time()
        var = VARForecaster(maxlags=12).fit(bundle.train.series.numpy())
        preds, trues = var.predict_over(bundle.test.series.numpy(), 96, 96)
        metrics, _ = metrics_from_arrays(preds, trues, bundle)
        print(f"  {'VAR':14s} MSE={metrics['mse']:.4f} MAE={metrics['mae']:.4f} "
              f"k_ar={var.k_ar} time={time.time()-t0:5.1f}s")
    except Exception as e:
        print(f"  {'VAR':14s} FAIL: {e}\n{traceback.format_exc()}")
    return rows


if __name__ == "__main__":
    print("torch", torch.__version__, "| mps", torch.backends.mps.is_available())
    a = test_fft_roundtrip()
    b = test_shapes()
    test_train_eval()
    print(f"\nFFT_OK={a}  SHAPES_OK={b}")
    print("SMOKE_DONE")
