#!/usr/bin/env python3
"""Run a single (model, dataset, horizon, seed) experiment end-to-end and write a JSON result.

Usage:
  python code/scripts/run_experiment.py --model CSLL --dataset ETTh1 --pred_len 96 --seed 0
Results go to results/raw/<dataset>__<model>__L<seq>__H<pred>__s<seed>.json
(plus a .perwin.npy of per-window MSE for Diebold-Mariano tests).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import os
import torch

# Cap PyTorch intra-op threads IN CODE (OMP_NUM_THREADS alone does not reliably limit
# torch, which otherwise spawns ~#cores threads per process and oversubscribes the CPU
# when several workers run in parallel). Default 4; workers set MTSF_TORCH_THREADS.
torch.set_num_threads(int(os.environ.get("MTSF_TORCH_THREADS", "4")))

# make the package importable when run as a script
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from csll import set_seed                                  # noqa: E402
from csll.config import model_config, train_config, seq_len_for, LOWDIM_FOR_VAR  # noqa: E402
from csll.data import build_dataset                        # noqa: E402
from csll.evaluate import metrics_from_arrays, score_torch  # noqa: E402
from csll.models import build_model                        # noqa: E402
from csll.train import get_device, train_torch, count_params  # noqa: E402

RESULTS = ROOT / "results" / "raw"


def run(model: str, dataset: str, pred_len: int, seq_len: int, seed: int,
        cfg_over: dict | None = None, train_over: dict | None = None,
        tag: str | None = None, save_model: bool = False, train_frac: float = 1.0) -> dict:
    set_seed(seed)
    bundle = build_dataset(dataset, seq_len, pred_len, train_frac=train_frac)
    device = get_device()
    t0 = time.time()

    if model == "VAR":
        if dataset not in LOWDIM_FOR_VAR:
            raise SystemExit(f"VAR not run on high-dim dataset '{dataset}'")
        from csll.models.var import VARForecaster
        var = VARForecaster(maxlags=min(24, seq_len)).fit(bundle.train.series.numpy())
        preds, trues = var.predict_over(bundle.test.series.numpy(), seq_len, pred_len)
        metrics, per_window = metrics_from_arrays(preds, trues, bundle)
        history = {"params": 0, "epochs_run": 0, "train_time_s": time.time() - t0,
                   "k_ar": int(var.k_ar)}
    else:
        mcfg = model_config(model, dataset, seq_len, pred_len, bundle.n_vars, cfg_over)
        net = build_model(model, mcfg)
        tcfg = train_config(model, dataset, bundle.n_vars, train_over)
        net, history = train_torch(net, bundle, device, tcfg)
        loaders = bundle.loaders(batch_size=tcfg["batch_size"])
        metrics, per_window = score_torch(net, loaders["test"], device, bundle)
        history["params"] = count_params(net)

    result = {
        "model": model, "dataset": dataset, "seq_len": seq_len, "pred_len": pred_len,
        "seed": seed, "n_vars": bundle.n_vars, "device": str(device),
        "metrics": metrics, "history": history,
        "wall_time_s": time.time() - t0, "timestamp": time.time(), "tag": tag,
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    stem = tag or f"{dataset}__{model}__L{seq_len}__H{pred_len}__s{seed}"
    with open(RESULTS / f"{stem}.json", "w") as f:
        json.dump(result, f, indent=2)
    np.save(RESULTS / f"{stem}.perwin.npy", per_window.astype(np.float32))
    if save_model and model != "VAR":
        torch.save(net.state_dict(), RESULTS / f"{stem}.pt")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--pred_len", type=int, required=True)
    ap.add_argument("--seq_len", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    seq_len = args.seq_len or seq_len_for(args.dataset)
    train_over = {"epochs": args.epochs} if args.epochs else None
    res = run(args.model, args.dataset, args.pred_len, seq_len, args.seed, train_over=train_over)
    print(json.dumps({"model": res["model"], "dataset": res["dataset"],
                      "pred_len": res["pred_len"], "seed": res["seed"],
                      **res["metrics"], "params": res["history"].get("params"),
                      "wall_s": round(res["wall_time_s"], 1)}, indent=2))


if __name__ == "__main__":
    main()
