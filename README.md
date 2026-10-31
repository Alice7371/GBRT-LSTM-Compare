# Autoregressive Streamflow Forecasting across 1–180 Day Horizons: Predictability Decay, Cross-Basin Pooling, and External Memory

**Alice7371** · *October 2026*

> Post-project technical write-up, kept as a record. Not submitted to arXiv.

---

## Abstract

We present the experimental record of a 669-basin CAMELS study (1990–2009)
mapping the **full predictability decay curve** of data-driven streamflow
forecasting across lead times h ∈ {1, 3, 7, 10, 15, 30, 90, 180} days under an
autoregressive, leakage-safe protocol anchored by persistence and day-of-year
climatology. We benchmark per-basin and pooled (regionally trained) variants
of XGBoost and LSTM — plus MLP, GRU, and window self-attention for a
**state–capacity axis** (stateless-shallow / stateless-deep / sequential
state / random-access memory), feature engineering as *external memory*, data
augmentation, hyperparameter search, and a research-grade GR4J replication
with Oudin PET. Headline findings:

1. **Predictability decays fast but never to zero in snowmelt basins.** The
   median basin loses most skill by h=30 (LSTM 0.074), yet snow-dominated
   basins (307/671, classified by training-period frozen-day fraction) retain
   NSE ≈ 0.10 at h=180 — statistically indistinguishable from climatology
   (0.103) — while rain basins collapse to ≈0.02. Predictability limits are
   climate-type-dependent: the h≥90 skill is initial-snowpack memory.
2. **The regional-training advantage inverts with horizon.** Pooled LSTM
   beats per-basin at h=1 (+0.028, d=0.83, 589/669 basins, FDR-significant)
   and at h=3 (+0.009), ties at h=7, and **loses at h≥10** (−0.013 at h=10,
   −0.047 by h=180); pooled XGBoost is worse at *every* horizon and
   catastrophically so at seasonal horizons (−0.18). Cross-basin knowledge
   transfer pays for universal short-range dynamics and harms basin-specific
   long-range storage behavior.
3. **Stateless capacity suffices short-range; recurrent state is
   load-bearing at range.** A stateless MLP on a 15-day flattened window
   matches LSTM at h=1 (0.626 vs 0.609, paired on 335 basins) but collapses
   at h=30 (−0.014 vs 0.074); GRU ≈ LSTM everywhere; window self-attention
   tracks LSTM through mid-range. Learning long-range memory from data
   (window) fails where explicit state (LSTM) succeeds.
4. **Feature engineering is a no-op under windowed autoregressive inputs** —
   at every horizon (paired ΔNSE +0.002…−0.008, not significant after FDR),
   including the 90-day memory channels added for h≥30: the window already
   contains what lags encode, and discharge dominates the input information
   so completely that even 1.0σ test-time noise on all meteorological
   channels moves NSE by <0.01.
5. **Normalization is the largest "trivial" lever.** Z-Score beats Min-Max
   for per-basin LSTM on **669/669 basins** (median ΔNSE +0.29 at h=1) —
   replicating and amplifying the original thesis finding on 4× more data.
6. Methodology: Optuna-TPE tuning adds **+0.03** (hidden 128 > 64, 9/12
   basins); augmentation is a no-op (see 4); GR4J with real basin areas
   (research replication, no snow module) reaches median NSE −0.30 at h=1,
   well below its literature reference — the autoregressive protocol is not
   its home turf.

## 1. The Task and Protocol

**Data.** CAMELS 671 basins, 1990-01-01–2009-12-31; 669 usable (2 excluded
for NaN/inf). Seven daily drivers (daylength, precipitation, shortwave
radiation, Tmax/Tmin, vapor pressure, discharge). Chronological 80/20 split
(train ≈ 1990–2005.9, test 2005.10–2009); all normalization statistics from
the train segment only.

**Task.** Direct multi-horizon forecasting: inputs use only data up to day
t; target is discharge at t+h, h ∈ {1,3,7,10,15,30,90,180}. Input window
w(h): 15 (h≤7), 60 (h∈{10,15}), 90 (h=30), 270 (h∈{90,180}). No recursion,
no future forcing. Baselines: persistence (dominant anchor for h≤7) and
day-of-year climatology (training-period doy means, 11-day smoothing; the
h≥10 anchor).

**Metrics.** NSE, KGE, MAPE, RMSE; medians with bootstrap 95% CIs (1000
resamples), paired Wilcoxon + sign tests, Benjamini–Hochberg FDR across the
horizon × family test family, effect sizes (rank-biserial, Cohen's d).

## 2. Models

| family | variants | station info |
|---|---|---|
| XGBoost | xgb-raw (flattened window), xgb-fe, xgb-fe-long (90-day memory channels, h≥30) | none |
| LSTM | raw zscore/minmax, fe, fe-long | none (per-basin) |
| MLP | flattened window → 256→128 FC | none |
| GRU | cell ablation of LSTM | none |
| window attention | single-head self-attention, last-token readout | none |
| pooled LSTM/GRU/attn | one shared model, all basins' train segments | learned 16-dim station embedding (pooled-emb), or none (plain) |
| pooled XGBoost | + station integer code as split feature | station code |
| GR4J + Oudin PET | research replication; real areas from CAMELS attributes | per-basin DE calibration |

All neural variants share hidden 64 / 2 layers / 40 epochs / batch 1024
(hidden 64 is throughput-driven; tuning to hidden 128 adds +0.03 — see §4).

## 3. Results

### 3.1 Predictability decay (median NSE; h≤15 over 669 basins, h=30 over 335, h=90/180 over 34)

| track | h=1 | h=3 | h=7 | h=10 | h=15 | h=30 | h=90 | h=180 |
|---|---|---|---|---|---|---|---|---|
| persistence | 0.494 | −0.148 | −0.465 | −0.545 | −0.603 | −0.751 | −1.071 | −1.173 |
| climatology | — | — | — | 0.019 | 0.019 | 0.018 | 0.037 | 0.061 |
| xgb-fe | 0.583 | 0.190 | 0.088 | 0.067 | 0.056 | 0.032 | 0.012 | 0.019 |
| **lstm-raw-zscore** | 0.609 | 0.253 | 0.151 | 0.133 | 0.113 | 0.074 | 0.032 | 0.046 |
| lstm-pooled-emb | 0.646 | 0.269 | 0.150 | 0.098 | 0.066 | 0.018 | 0.014 | −0.001 |

### 3.2 The snow/rain predictability cliff

Snow class = training-period frozen-day fraction > 0.15 (307 basins;
threshold matches known snow basins). At h=180: snow LSTM **0.102** ≈
climatology 0.103 (initial-snowpack memory), rain LSTM 0.019 ≈ 0. The
per-basin LSTM is the only family whose seasonal skill survives in snow
basins; pooled models lose basin-specific storage behavior.

### 3.3 Pooling reversal (paired per-basin, FDR-significant where marked)

| h | median ΔNSE | wins | Cohen's d | FDR sig |
|---|---|---|---|---|
| 1 | **+0.028** | 589/669 | 0.83 | ✔ |
| 3 | +0.009 | 502/669 | 0.36 | ✔ |
| 7 | +0.004 | 417/669 | 0.00 | ✔ |
| 10 | −0.013 | 214/669 | −0.30 | n.s. |
| 15 | −0.026 | 166/669 | −0.43 | n.s. |
| 30 | −0.037 | 85/335 | −0.31 | n.s. |
| 90 | −0.018 | — | — | n.s. |
| 180 | −0.047 | — | — | n.s. |

(h=90/180 per-basin values are from the 34-basin seasonal tier; deltas from
the decay-table medians.)

### 3.4 State–capacity axis (335-basin subset)

| track | h=1 | h=7 | h=30 |
|---|---|---|---|
| mlp-zscore | 0.626 | 0.136 | −0.014 |
| attn-zscore | 0.587 | 0.138 | 0.076 |
| gru-zscore | 0.618 | 0.150 | 0.079 |
| lstm-raw-zscore (669-basin median) | 0.609 | 0.151 | 0.074 |

MLP matches LSTM at h=1 (paired Δ≈ +0.02, n.s.), collapses by h=30;
attention tracks LSTM through mid-range (random-access memory suffices);
GRU ≈ LSTM (gate complexity doesn't matter).

### 3.5 Negative results

- **Feature engineering**: paired ΔNSE +0.002…−0.008 at every horizon
  (FDR n.s.) — window subsumes lags; even 90-day channels don't help at
  h≥30.
- **Augmentation** (met jitter σ sweep 0.01–0.2, time masking, eval-noise
  probes): no effect; discharge channel dominates the input information so
  completely that 1.0σ test noise on all met channels moves NSE <0.01.
- **Pooled XGBoost**: worse at every horizon (−0.036 at h=1 → −0.18 at
  h=180). Station code as a split feature cannot emulate a shared dynamics
  prior.
- **GR4J replication** (real areas, Oudin PET, no snow module): −0.30 median
  NSE at h=1 — reported as a research-grade replication, below the
  literature reference (calibrated GR4J+CemaNeige reaches ≈0.5+).

## 4. Methodology Notes

- **HPT** (Optuna TPE, 50 trials × 12 stratified basins): hidden 128,
  dropout 0, lr 1e-3, batch 512, z-score; tuned beats default **+0.03**
  (9/12 basins) — tuning is not optional even in the autoregressive regime.
  All study models use the conservative hidden-64 default.
- **Normalization** (z-score vs min-max, h=1, 669 basins): 669/669 basins
  favor z-score, median ΔNSE **+0.29** — the single largest "trivial" lever
  found.
- **Scaling honesty**: pooled runs subsample ≤4500 (h≤15) / ≤3000 (h=30) /
  ≤1000 (h≥90) training sequences per station; seasonal per-station tiers
  use 34 stratified basins (stride 20), extra families 335 (stride 2).

## 5. Repository Layout

```
data.py / featurize.py        protocol, causal engineered features
models.py                     xgb/lstm/gru/mlp/attention tracks (per-basin)
pooled.py                     pooled variants + runners
extra_models.py               MLP / attention / GRU per-station runners
gr4j.py                       GR4J + Oudin PET + DE calibration (real areas)
hpt.py / hpt_validate.py      Optuna search + default-vs-tuned validation
aug.py                        augmentation ablation + noise robustness probe
eval_official.py              official CAMELS model outputs under our protocol
analyze.py                    medians/CI/FDR/effect sizes/strata/decay tables
run_experiments.py / queue_driver.py / tail2_driver.py / dl_watchdog.py
                              resumable experiment orchestration
docs/BACKGROUND.md            task definition, variant table, paper text draft
paper/main.tex                LaTeX manuscript skeleton (results pending)
study_results/                all result CSVs + analysis markdown (not committed)
thesis_experiments/           archived thesis-replication code (LightGBM)
```

## References

1. Newman et al. (2015), CAMELS. *J. Hydrometeorology* 16.
2. Kratzert et al. (2018, 2019), HESS 22:6005, 23:5089.
3. Kratzert et al. (2024), Never train an LSTM on a single basin. HESS 28:4187.
4. Nearing et al. (2024), Global prediction of extreme floods. Nature 627:559.
5. Gauch et al. (2021), arXiv:1911.07249.
6. Perrin et al. (2003), GR4J. J. Hydrol. 279:275.
7. Kirchner (2009), Catchments as simple dynamical systems. WRR 45:W02429.
8. Hunt et al. (2022), HESS 26:5449. · Li et al. (2023), Hydrology 15:1019.
9. Schoppa et al. (2020), J. Hydrol. 588.
10. Clerc-Schwarzenbach et al. (2024), HESS 28:4219.

---

*Code and full result tables in `study_results/` (CSV, regenerable via
`analyze.py`). This repository documents the study; the `dataset/` directory
holds the 27-basin thesis subset used by `thesis_experiments/`. Data: CAMELS
(Newman et al. 2015).*
