# CSLL: Complex Spectral Lead–Lag Networks for Delay-Aware Multivariate Time-Series Forecasting

**Francois Petizon** — independent researcher.

> ### 📄 [**Read the paper (PDF)**](https://francoispzn.github.io/mtsf-research/paper/main_ieee.pdf)
> [LaTeX source](paper/main_ieee.tex)

---

Most multivariate forecasters that model dependencies *between* series do so in a **phase-blind**
way: cross-variate attention, channel-mixing MLPs, and spectral/graph models represent inter-series
coupling as real-valued and same-timestep, and cannot express that one series **leads** another by a
delay. This project introduces a cross-channel operator whose **phase encodes a continuous
inter-series delay** (exact by the Fourier shift theorem), and asks a single question:

> **When does modelling that lead–lag delay actually improve forecasts?**

The answer is a characterisation, not a leaderboard win — and the project is deliberately built to
report where the mechanism helps, where it is redundant, and where a stronger model already subsumes
it, rather than a single headline number.

## What the project actually finds (the honest version)

- **A provably-exact delay estimator.** On controlled planted-delay data the operator recovers the
  true delays exactly (slope ≈ 1, *r* = 1.00) where a naive gradient-only estimator collapses
  (slope ≈ 0). Escaping the classic time-delay-estimation local-minimum trap requires a
  correlation-based initialisation, a zero-padded (linear, not circular) shift, and a head-free
  *direct* forecast — each individually necessary. *(§IV, Fig. 2.)*
- **A clean empirical dichotomy.** On weekly regional influenza — where an epidemic reaches regions
  in sequence over weeks — the phase branch lowers a lightweight model's error by **13–19%** (3
  seeds). On 5-minute traffic and the standard long-horizon benchmarks, where coupling is
  same-time, real mixing suffices and the phase branch is redundant or mildly harmful. *(§V.)*
- **An honest negative result.** Simple a-priori statistics of the inter-series lead (a lead/horizon
  ratio; a "delay-advantage" statistic Δ(H)) **do not cleanly predict** which regime a dataset is
  in — Δ(H) gets the sign right in fewer than half of settings. So the project does *not* claim a
  predictive scalar. *(§V, Table II.)*
- **An adaptive model that resolves it.** Because you cannot cheaply predict the regime, a gated
  **hybrid** runs a phase branch *and* a mixing branch and learns their blend per dataset. Without a
  regime label it stays within seed noise of the best specialised variant everywhere, and removes
  the ~17% regression a phase-only model suffers on same-time data. *(§VI, Fig. 1.)*
- **Stated scope.** CSLL does **not** beat the strongest baselines in absolute accuracy — iTransformer
  and PatchTST win outright on the epidemic data — because a strong attention backbone already
  captures the delayed structure the operator supplies to a *weak* backbone. Its case is efficiency
  (tens of thousands of parameters vs hundreds of thousands), adaptivity, and a characterised
  mechanism. Diebold–Mariano tests show it is *significantly better* than the best baseline on
  METR-LA traffic and *significantly worse* on influenza — reported both ways. *(§VII.)*

> This repository is as much a demonstration of *how* to do the science honestly — chasing a
> hypothesis, finding it does not beat SOTA, resisting the temptation to overclaim (two candidate
> "wins" turned out to be an artifact and a fragile statistic, and are reported as such), and
> extracting a real, useful result anyway — as it is a forecasting method.

## Repository layout
```
mtsf-research/
  paper/     main_ieee.tex (IEEE journal LaTeX), references.bib, main_ieee.pdf (compiled)
  code/csll/ Python package: method (models/csll.py), baselines, data pipeline, RevIN,
             metrics, stats, synthetic generators, training/eval harness
  code/scripts/  experiment campaigns, aggregation, figures, delay validation
  data/      fetch_data.sh, convert_st.py, convert_epi.py, PROVENANCE.md
             (datasets download into data/raw/, which is git-ignored)
  results/   tables/ (CSV + .tex) and figures/ (PDF + PNG) are kept in-repo for browsing;
             raw/ (per-run JSON) is regenerated and git-ignored
```

## Method at a glance
```
Y = RevIN⁻¹( CI-backbone(X)  +  α_p · PhaseBranch(X)  +  α_m · MixingBranch(X) )
```
Per frequency band *b*, the cross-channel operator is `W_b(f)_{ij} = A_{ij} · exp(−i ω_f (p_i − q_j))`
— magnitude = coupling strength, phase = a continuous inter-series delay `D_{ij} = p_i − q_j`, bounded
to the identifiable range and factorised additively to O(N). The **phase branch** forecasts by
synthesising the shifted spectrum *directly* at the future positions (so the forecast-optimal delay
equals the physical delay); a zero-delay band reads the zero padding and contributes nothing, so it
cannot express same-time coupling. The **mixing branch** supplies that via real-valued spectral
coupling and a linear head. Warm gates `α_p, α_m` let the data choose the blend.

- Method + all ablation switches (`hybrid`, `direct`, `pad2x`, `strict_bound`, xcorr init): `code/csll/models/csll.py`
- Architecture figure: Fig. 1 of the paper.

## Reproducing the results

**1. Environment** (Python 3.9+, CPU is sufficient and is the default):
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r code/requirements.txt          # torch 2.8, numpy 2.0.2, pandas 2.3.3, ...
```
Runs are **CPU-only** for reproducibility (Apple MPS aborts on the custom spectral einsums). Cap
threads with `MTSF_TORCH_THREADS=4`.

**2. Get the data** (public; ~700 MB into the git-ignored `data/raw/`):
```bash
bash data/fetch_data.sh          # LTSF + spatio-temporal traffic; SHA-256 verified
python3 data/convert_st.py        # METR-LA/PEMS-BAY/PEMS0x -> harness CSVs
python3 data/convert_epi.py       # Cola-GNN regional influenza -> harness CSVs
```
Provenance, licences, and checksums: [`data/PROVENANCE.md`](data/PROVENANCE.md).

**3. Run the experiments** (sequential, resumable via skip-existing; fixed seeds):
```bash
# The controlled kill-test (proves the estimator recovers planted delays)
python3 code/scripts/controlled_v2.py

# The main campaign: epidemics + LTSF control + spatio-temporal traffic
python3 code/scripts/campaign_v2.py          # then: traffic_finish.py for PEMS-BAY
python3 code/scripts/hybrid_campaign.py      # the adaptive hybrid, multi-seed

# A single run
MTSF_DEVICE=cpu python3 code/scripts/run_experiment.py --model CSLL2H --dataset ili_us_state --pred_len 2
```

**4. Build every table and figure:**
```bash
python3 code/scripts/aggregate_v2.py   # -> results/tables/v2_*.{csv,tex}  (all paper tables)
python3 code/scripts/plots_v2.py       # -> results/figures/fig_{envelope,pareto_v2,mechanism,mseh}.pdf
python3 code/scripts/fig_delays.py     # -> results/figures/fig_{killtest,delay_real}.pdf
```

**5. Compile the paper** (install tectonic once; it auto-fetches LaTeX packages):
```bash
cd paper && tectonic main_ieee.tex     # -> paper/main_ieee.pdf
```

## Reproducibility notes
- Every number in the paper comes from `results/raw/` produced by the scripts above — nothing is
  hand-entered; `aggregate_v2.py` regenerates the exact tables.
- All models share one harness (`code/csll/{train,evaluate}.py`): identical chronological splits,
  train-only normalisation, RevIN, no dropped last batch, a **uniform training budget**, and an
  **honestly-trained gate** (no post-hoc validation selection of the branch).
- Protocols are declared, including reduced ones: PEMS-BAY uses 1 seed at H ∈ {12, 96} as the
  confirmation speed benchmark, disclosed in the paper's limitations.

## Status / caveats
- **Not benchmarked against domain-specific baselines** (LIFT, VCformer; DCRNN, Graph WaveNet;
  Cola-GNN). These are the natural next step before a peer-review submission.
- Single look-back (L = 96); a lookback-sensitivity study is future work.
- Interpretability transfers imperfectly: delays are exact on controlled data but align only weakly
  with physical geography on real data — reported plainly (§VII).
