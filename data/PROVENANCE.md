# Dataset Provenance

All datasets used in this study are **public benchmarks** widely used in the
long-term time-series forecasting (LTSF) literature. Run `bash data/fetch_data.sh`
to download them into `data/raw/` and verify integrity against the checksums below.

## Files

| File | Rows | Cols | Sampling | Span | Target var |
|------|------|------|----------|------|------------|
| `ETTh1.csv` | 17,420 | 7 (+date) | 1 hour | 2016-07-01 → 2018-06-26 | `OT` (oil temperature) |
| `ETTh2.csv` | 17,420 | 7 (+date) | 1 hour | 2016-07-01 → 2018-06-26 | `OT` |
| `ETTm1.csv` | 69,680 | 7 (+date) | 15 min | 2016-07-01 → 2018-06-26 | `OT` |
| `ETTm2.csv` | 69,680 | 7 (+date) | 15 min | 2016-07-01 → 2018-06-26 | `OT` |
| `weather.csv` | 52,696 | 21 (+date) | 10 min | 2020-01-01 → 2021-01-01 | `OT` (CO2 / last col) |
| `exchange_rate.csv` | 7,588 | 8 (+date) | 1 day | 1990 → 2010 | `OT` |
| `national_illness.csv` | 966 | 7 (+date) | 1 week | 2002 → 2020-06 | `OT` |
| `electricity.csv` | 26,304 | 321 (+date) | 1 hour | 2016-07 → 2019-07 | `OT` |
| `traffic.csv` | 17,544 | 862 (+date) | 1 hour | 2016-07 → 2018-07 | `OT` |

## Sources & Licensing

### ETT (Electricity Transformer Temperature) — ETTh1, ETTh2, ETTm1, ETTm2
- **Origin:** Zhou, H. et al. "Informer: Beyond Efficient Transformer for Long
  Sequence Time-Series Forecasting." AAAI 2021.
- **Canonical repository:** https://github.com/zhouhaoyi/ETDataset (`ETT-small/`)
- **License:** Creative Commons Attribution (CC BY) per the ETDataset repository.
- **Description:** 7 features (6 power-load covariates: HUFL, HULL, MUFL, MULL,
  LUFL, LULL) plus the target `OT` (transformer oil temperature) from two Chinese
  regions. `h` = hourly, `m` = 15-minute resolution.

### Weather (Jena)
- **Origin:** Max-Planck-Institute for Biogeochemistry, Jena weather station
  (recorded every 10 minutes during 2020). Popularised as an LTSF benchmark by
  Wu, H. et al. "Autoformer" (NeurIPS 2021).
- **Direct-download mirror used here:** Hugging Face Hub dataset
  `pkr7098/time-series-forecasting-datasets` (`weather.csv`), which is byte-identical
  to the Autoformer-preprocessed benchmark version (21 meteorological variables +
  target `OT`). Autoformer's original distribution is via Google Drive, which is not
  directly `curl`-able; this mirror provides the same file over HTTPS.
- **Underlying data source:** https://www.bgc-jena.mpg.de/wetter/ (public).

### Exchange, ILI, Electricity, Traffic
- **Origin:** the standard LTSF versions popularised by Lai et al. (LSTNet, SIGIR 2018) and
  Wu et al. (Autoformer, NeurIPS 2021). Exchange = daily exchange rates of 8 countries
  (1990–2010); ILI = weekly influenza-like-illness ratios, US CDC; Electricity = hourly
  consumption of 321 clients; Traffic = hourly road-occupancy rates from 862 San-Francisco-Bay
  sensors (California Dept. of Transportation).
- **Direct-download mirror used here:** Hugging Face Hub dataset
  `pkr7098/time-series-forecasting-datasets` (`exchange_rate.csv`, `national_illness.csv`,
  `electricity.csv`, `traffic.csv`), byte-identical to the Autoformer-preprocessed benchmark files.
- **Note:** Traffic (862 vars) and Electricity (321 vars) are high-dimensional; the CSLL model
  uses its low-rank `A_b` variant on these. VAR is not run on them (intractable at that `N`).

## SHA-256 checksums (recorded 2026-07-01)

```
f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066  ETTh1.csv
a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b  ETTh2.csv
6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e  ETTm1.csv
db973ca252c6410a30d0469b13d696cf919648d0f3fd588c60f03fdbdbadd1fd  ETTm2.csv
34ee981d07313e51da2a50bb600072c8ae4a69cb4b0651f4cb93a069d7a2ba63  weather.csv
```

## Standard LTSF splits (chronological, no shuffling)

Following the established protocol (Informer/Autoformer/DLinear/PatchTST):
- **ETTh1/h2:** train 12 months, val 4 months, test 4 months → 60/20/20 by the
  fixed index cut (12·30·24 / 4·30·24 / 4·30·24 hours).
- **ETTm1/m2:** same wall-clock split at 15-min resolution (12·30·24·4, etc.).
- **Weather:** 70% / 10% / 20% chronological train/val/test.
Normalisation statistics (mean/std) are computed on the **training split only** and
applied to val/test to avoid leakage.

## Spatio-temporal traffic datasets (added 2026-07-04, v2 campaign)

| Dataset | File | Shape | Origin | Mirror used |
|---|---|---|---|---|
| METR-LA | st/metr-la.h5 | 34272 x 207, 5-min speeds (mph), LA 2012-03-01..06-27 | Li et al., DCRNN (ICLR 2018) | github.com/deepkashiwa20/DL-Traff-Graph (sha256 matches HF mirror jimmygao3218/METRLA) |
| PEMS-BAY | st/pems-bay.h5 | 52116 x 325, 5-min speeds, Bay Area 2017-01..06 | Li et al., DCRNN | huggingface.co/datasets/jimmygao3218/PEMSBAY |
| PEMS04 | st/PEMS04.npz (+.csv edges) | 16992 x 307 x 3 (flow/occ/speed), 2018-01..02 | Guo et al., ASTGCN (AAAI 2019) | huggingface.co/datasets/jimmygao3218/PEMS04 (git oid also matches Neerajjjjjjj/PEMS04) |
| PEMS08 | st/PEMS08.npz (+.csv edges) | 17856 x 170 x 3, 2016-07..08 | Guo et al., ASTGCN | huggingface.co/datasets/jimmygao3218/PEMS08 |
| Sensor graph | st/distances_la_2012.csv, st/graph_sensor_locations.csv, st/distances_bay_2017.csv, st/graph_sensor_ids.txt, st/graph_sensor_locations_bay.csv | road-network distances (m) + lat/lon | github.com/liyaguang/DCRNN data/sensor_graph | raw.githubusercontent.com (same repo) |

SHA-256 checksums are enforced in fetch_data.sh. Conversion to harness CSVs (date column +
one column per sensor; PEMS0X use feature 0 = flow) is done by `data/convert_st.py`; the
resulting metr_la.csv / pems_bay.csv / pems04.csv / pems08.csv preserve sensor-id column
order so learned delay matrices can be joined to the distance files.

Missing data: METR-LA encodes missing readings as 0.0 (8.11% of entries; PEMS-BAY ~0%).
We keep them unmodified — every model in the harness sees identical inputs — and note the
convention in the paper. Sampling is 5 minutes: freeway propagation delays (congestion
back-propagation ~15-20 km/h, i.e. ~3.7 min/km) are therefore REPRESENTABLE at 1-3 samples
for 1-4 km sensor spacings, unlike the hourly LTSF 'traffic' benchmark where they are
sub-sample. This is the motivation for the v2 evaluation on these sets.
