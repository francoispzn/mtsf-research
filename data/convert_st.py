#!/usr/bin/env python3
"""Convert raw spatio-temporal traffic files (downloaded by fetch_data.sh into data/raw/st/)
into the CSV layout the harness data pipeline expects (date column + one column per sensor).

Inputs (any subset; missing ones are skipped with a note):
  data/raw/st/metr-la.h5        pandas HDFStore, 34272 x 207 speeds (mph), 5-min, LA 2012
  data/raw/st/pems-bay.h5       pandas HDFStore, 52116 x 325 speeds (mph), 5-min, Bay 2017
  data/raw/st/PEMS04.npz        (16992, 307, 3) [flow, occupancy, speed], 5-min, 2018
  data/raw/st/PEMS08.npz        (17856, 170, 3), 5-min, 2016

Outputs:
  data/raw/metr_la.csv, pems_bay.csv, pems04.csv, pems08.csv
  (PEMS0X use feature 0 = flow, the standard target in the ST literature.)

Missing data note: METR-LA/PEMS-BAY encode missing readings as 0.0 (~8%/~0.03% of entries).
We keep them AS-IS (no imputation): every model in the harness sees identical data, and the
convention is documented in the paper. Column order and sensor ids are preserved so learned
delay matrices can be joined against distances_la_2012.csv / distances_bay_2017.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ST = ROOT / "raw" / "st"
OUT = ROOT / "raw"


def convert_h5(name_in: str, name_out: str, start: str):
    src = ST / name_in
    if not src.exists():
        print(f"[skip] {src} not found")
        return
    df = pd.read_hdf(src)
    # index is datetime already in the originals; enforce and rename
    dates = pd.date_range(start, periods=len(df), freq="5min")
    out = df.copy()
    out.insert(0, "date", dates)
    zero_frac = float((df.to_numpy() == 0).mean())
    out.to_csv(OUT / name_out, index=False)
    print(f"[ok] {name_out}: {df.shape[0]} x {df.shape[1]} sensors, "
          f"zeros(missing) = {100*zero_frac:.2f}%  cols preserved from HDF index")


def convert_npz(name_in: str, name_out: str, start: str, feature: int = 0):
    src = ST / name_in
    if not src.exists():
        print(f"[skip] {src} not found")
        return
    arr = np.load(src)["data"][:, :, feature].astype(np.float32)   # (T, N)
    dates = pd.date_range(start, periods=arr.shape[0], freq="5min")
    df = pd.DataFrame(arr, columns=[str(i) for i in range(arr.shape[1])])
    df.insert(0, "date", dates)
    df.to_csv(OUT / name_out, index=False)
    print(f"[ok] {name_out}: {arr.shape[0]} x {arr.shape[1]} sensors (feature {feature} = flow)")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    convert_h5("metr-la.h5", "metr_la.csv", start="2012-03-01")
    convert_h5("pems-bay.h5", "pems_bay.csv", start="2017-01-01")
    convert_npz("PEMS04.npz", "pems04.csv", start="2018-01-01")
    convert_npz("PEMS08.npz", "pems08.csv", start="2016-07-01")
    print("CONVERT_ST_DONE")
