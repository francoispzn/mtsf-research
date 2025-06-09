#!/usr/bin/env python3
"""Fast compute-budget probe: time K training steps per (model,dataset) and extrapolate to
a full epoch. Also re-checks VAR stability. Does NOT write results/raw."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from csll import set_seed                                    # noqa: E402
from csll.config import model_config, train_config           # noqa: E402
from csll.data import build_dataset                           # noqa: E402
from csll.evaluate import metrics_from_arrays                 # noqa: E402
from csll.models import build_model                           # noqa: E402
from csll.train import get_device, count_params              # noqa: E402

PAIRS = [
    ("ettm1_medium", "ETTm1", ["DLinear", "CSLL", "iTransformer", "PatchTST", "LSTM"]),
    ("weather_medium", "weather", ["CSLL", "iTransformer", "PatchTST", "LSTM"]),
    ("electricity_heavy", "electricity", ["DLinear", "CSLL", "iTransformer", "PatchTST"]),
    ("traffic_heavy", "traffic", ["DLinear", "CSLL", "iTransformer", "PatchTST"]),
]
K = 15  # timed steps


def probe():
    device = get_device()
    print(f"device={device}\n")
    for label, ds, models in PAIRS:
        L, H = 96, 96
        bundle = build_dataset(ds, L, H)
        tr = len(bundle.train)
        print(f"== {label} ({ds}) N={bundle.n_vars} train_windows={tr} ==")
        for m in models:
            try:
                set_seed(0)
                cfg = model_config(m, ds, L, H, bundle.n_vars)
                net = build_model(m, cfg).to(device)
                tcfg = train_config(m, ds, bundle.n_vars)
                bs = tcfg["batch_size"]
                loader = bundle.loaders(batch_size=bs)["train"]
                opt = torch.optim.Adam(net.parameters(), lr=tcfg["lr"])
                crit = torch.nn.MSELoss()
                net.train()
                it = iter(loader)
                # warmup 2
                for _ in range(2):
                    x, y = next(it)
                    opt.zero_grad(); loss = crit(net(x.to(device)), y.to(device)); loss.backward(); opt.step()
                t0 = time.time()
                steps = 0
                for _ in range(K):
                    try:
                        x, y = next(it)
                    except StopIteration:
                        break
                    opt.zero_grad(); loss = crit(net(x.to(device)), y.to(device)); loss.backward(); opt.step()
                    steps += 1
                dt = (time.time() - t0) / max(1, steps)
                nb = (tr + bs - 1) // bs
                print(f"  {m:14s} bs={bs:>3d} {dt*1000:7.1f} ms/step  ~{dt*nb:7.1f} s/epoch  "
                      f"params={count_params(net):>9d}")
            except Exception as e:
                print(f"  {m:14s} ERROR: {e}")
        print()

    # VAR stability re-check on weather (N=21) and ETTh1
    from csll.models.var import VARForecaster
    for ds in ["ETTh1", "weather"]:
        b = build_dataset(ds, 96, 96)
        t0 = time.time()
        var = VARForecaster().fit(b.train.series.numpy())
        preds, trues = var.predict_over(b.test.series.numpy(), 96, 96)
        m, _ = metrics_from_arrays(preds, trues, b)
        print(f"VAR {ds}: k_ar={var.k_ar} MSE={m['mse']:.4f} MAE={m['mae']:.4f} "
              f"finite={bool(__import__('numpy').isfinite(preds).all())} time={time.time()-t0:.1f}s")
    print("PROBE_DONE")


if __name__ == "__main__":
    probe()
