#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# fetch_data.sh  --  Reproducibly download the benchmark datasets used in this
# study and verify their integrity against recorded SHA-256 checksums.
#
# Usage:   bash data/fetch_data.sh
# Result:  populates data/raw/ with 5 CSV files (ETTh1/h2, ETTm1/m2, weather)
#
# All datasets are public. Provenance and licensing are documented in
# data/PROVENANCE.md. If a checksum mismatches, the upstream file changed and
# the results in the paper should be regenerated and re-verified.
# ---------------------------------------------------------------------------
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW="$SCRIPT_DIR/raw"
mkdir -p "$RAW"
cd "$RAW"

# Canonical ETT source: original authors' repository (Zhou et al., AAAI 2021).
ETT_BASE="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small"
# Weather: standard Autoformer LTSF benchmark version (21 vars, Jena 2020),
# mirrored on the Hugging Face Hub for direct download.
WEATHER_URL="https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets/resolve/main/weather.csv"

echo "== Downloading ETT (ETTh1, ETTh2, ETTm1, ETTm2) =="
for f in ETTh1 ETTh2 ETTm1 ETTm2; do
  curl -fsSL -o "$f.csv" "$ETT_BASE/$f.csv" && echo "  ok  $f.csv" || { echo "  FAIL $f.csv"; exit 1; }
done

echo "== Downloading Weather + Exchange + ILI + Electricity + Traffic (HF mirror) =="
HF_BASE="https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets/resolve/main"
for fn in weather.csv exchange_rate.csv national_illness.csv electricity.csv traffic.csv; do
  curl -fsSL -o "$fn" "$HF_BASE/$fn" && echo "  ok  $fn" || { echo "  FAIL $fn"; exit 1; }
done

echo "== Verifying SHA-256 checksums =="
cat > /tmp/mtsf_checksums.txt <<'SUMS'
f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066  ETTh1.csv
a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b  ETTh2.csv
6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e  ETTm1.csv
db973ca252c6410a30d0469b13d696cf919648d0f3fd588c60f03fdbdbadd1fd  ETTm2.csv
34ee981d07313e51da2a50bb600072c8ae4a69cb4b0651f4cb93a069d7a2ba63  weather.csv
48b4d9d3d508f5104162e85b9a6042e3557fde11aa9f2944eba8c0d0efc89842  exchange_rate.csv
93601f64d2566dc796ca4305adad8b8560c2db1a1ff04543c3bd813a7263570a  national_illness.csv
7e45845d54c5219bad0ae6bc1b5316cf8ff9cead5d33fa998a5a51c2e4a497ad  electricity.csv
cb06463d56fa17d87f47027cd9389ceae82a69eddee51cdb61480e120dab0b16  traffic.csv
SUMS

if command -v shasum >/dev/null 2>&1; then
  shasum -a 256 -c /tmp/mtsf_checksums.txt
elif command -v sha256sum >/dev/null 2>&1; then
  sha256sum -c /tmp/mtsf_checksums.txt
else
  echo "WARNING: no shasum/sha256sum available; skipping integrity check."
fi

echo "== Done. Files in $RAW =="
ls -la "$RAW"

# ---------------------------------------------------------------------------
# Spatio-temporal traffic datasets (5-min sampling; used by the v2 campaign)
# Sources verified 2026-07-04; sha256-cross-checked against independent mirrors.
# ---------------------------------------------------------------------------
echo "== Downloading spatio-temporal traffic datasets (METR-LA / PEMS-BAY / PEMS04/08) =="
mkdir -p st && cd st
curl -fsSL -o metr-la.h5 "https://raw.githubusercontent.com/deepkashiwa20/DL-Traff-Graph/master/METRLA/metr-la.h5"
curl -fsSL -o pems-bay.h5 "https://huggingface.co/datasets/jimmygao3218/PEMSBAY/resolve/main/PEMS-BAY.h5"
curl -fsSL -o PEMS04.npz "https://huggingface.co/datasets/jimmygao3218/PEMS04/resolve/main/PEMS04.npz"
curl -fsSL -o PEMS04.csv "https://huggingface.co/datasets/jimmygao3218/PEMS04/resolve/main/PEMS04.csv"
curl -fsSL -o PEMS08.npz "https://huggingface.co/datasets/jimmygao3218/PEMS08/resolve/main/PEMS08.npz"
curl -fsSL -o PEMS08.csv "https://huggingface.co/datasets/jimmygao3218/PEMS08/resolve/main/PEMS08.csv"
DCRNN="https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph"
for fn in distances_la_2012.csv graph_sensor_locations.csv distances_bay_2017.csv \
          graph_sensor_ids.txt graph_sensor_locations_bay.csv; do
  curl -fsSL -o "$fn" "$DCRNN/$fn" && echo "  ok  $fn" || { echo "  FAIL $fn"; exit 1; }
done

cat > /tmp/mtsf_st_checksums.txt <<'SUMS'
64784b76d6fb8ec9bff4b6decafb354da2bb37840468fdccee5044e511277c05  metr-la.h5
65d69fb0a2323dba9867179eb7af47c8b814186bc459ff0a4937d21614153c8f  pems-bay.h5
95a3c9b720fffdb85f0330d09bfab41b0b3cad0ca86c0d7d5f3accacb4ac999a  PEMS04.npz
3e36226ec088ab5fb7d7896f5ae733153e6e27477a759427b65e64adb8be1d23  PEMS04.csv
e1d03ce74e9fb79149e6e7c37c680214822c4a285baa2f42278cacabcd25c075  PEMS08.npz
e5ab2a62f275741e6b07d3fba1623883ff9d692d406a44724d26efddc5e24b20  PEMS08.csv
a576a2a3e28dbb959be6da22688e24dd1b246b81264595e129147c256cd53de5  distances_la_2012.csv
eb8ea96e07358b45d0e4ba3b89c2673fa20c54af50150249e627389e749ade6f  graph_sensor_locations.csv
e5feed06bfa1ba4c554a946d0e03d99f2018365eec5a8f28fd8504dea9d082b5  distances_bay_2017.csv
3ba026caa2e6263ab0ea54b0fa1b125dbfa7216544cd05313b555e826292b990  graph_sensor_ids.txt
276ee01059610774d4e59572507f7e32eaac21f1f5882fcd9e3d7d426a4b7a6c  graph_sensor_locations_bay.csv
SUMS
if command -v shasum >/dev/null 2>&1; then shasum -a 256 -c /tmp/mtsf_st_checksums.txt
elif command -v sha256sum >/dev/null 2>&1; then sha256sum -c /tmp/mtsf_st_checksums.txt; fi
cd ..

echo "== Converting ST datasets to harness CSVs (needs pandas+pytables; see convert_st.py) =="
python3 "$(dirname "$0")/convert_st.py" || /opt/anaconda3/bin/python3 "$(dirname "$0")/convert_st.py" || \
  echo "WARNING: conversion needs a python with pandas+pytables (pip install tables); run data/convert_st.py manually."
