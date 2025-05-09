#!/usr/bin/env python3
"""Convert the Cola-GNN regional ILI matrices (data/raw/epi/) to harness CSVs.

Files (rows = weeks, cols = locations, weekly ILI patient counts; no header):
  japan.txt      348 x 47  Japan prefectures (IDWR), Aug 2012 - Mar 2019
  region785.txt  785 x 10  US HHS regions (ILINet), 2002 - 2017
  state360.txt   360 x 49  US states (CDC, Florida excluded), 2010 - 2017

Outputs data/raw/ili_japan.csv, ili_us_hhs.csv, ili_us_state.csv with a synthetic weekly
date column (documented start dates) and integer-indexed location columns, preserving
column order so adjacency files (data/raw/epi/*-adj.txt) join by index.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
EPI = ROOT / "raw" / "epi"
OUT = ROOT / "raw"

SPECS = [
    ("japan.txt", "ili_japan.csv", "2012-08-05"),
    ("region785.txt", "ili_us_hhs.csv", "2002-01-06"),
    ("state360.txt", "ili_us_state.csv", "2010-01-03"),
]

if __name__ == "__main__":
    for src, dst, start in SPECS:
        arr = np.loadtxt(EPI / src, delimiter=",")
        df = pd.DataFrame(arr, columns=[str(i) for i in range(arr.shape[1])])
        df.insert(0, "date", pd.date_range(start, periods=len(df), freq="W-SUN"))
        df.to_csv(OUT / dst, index=False)
        print(f"[ok] {dst}: {arr.shape[0]} weeks x {arr.shape[1]} locations, "
              f"range {arr.min():.0f}-{arr.max():.0f}")
    print("CONVERT_EPI_DONE")
