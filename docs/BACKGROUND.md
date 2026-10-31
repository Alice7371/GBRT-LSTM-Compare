# Research Background & Positioning

*Working document, updated 2026-10-29. Sections 1–2 are the draft paper text
(Introduction + Related Work); sections 3+ are engineering definitions.*

---

## 1. Introduction (paper draft)

Reliable streamflow prediction underpins flood early warning, reservoir
operation, and water-supply planning. The task spans three regimes with distinct
information sources: *short-range* forecasting (1–7 days), where catchment
storage makes discharge strongly autocorrelated; *medium-range* forecasting
(10–30 days), where skill must be extracted from slow storage anomalies; and
*seasonal* prediction (90–180 days), where predictability rests almost entirely
on initial states — chiefly snowpack and groundwater — plus climatology. Most
machine-learning hydrology research addresses only the first regime; how model
skill decays across the full spectrum, and whether that decay is
climate-dependent, remains unmapped.

Two modeling paradigms dominate data-driven hydrology. Gradient-boosted tree
ensembles are stateless function approximators whose temporal competence must
be supplied externally, through windowed inputs or engineered lag/state
features. Long Short-Term Memory (LSTM) networks carry state internally and
can, in principle, learn storage–discharge dynamics themselves. Whether the
second paradigm's structural advantage grows, shrinks, or inverts as the
forecast horizon extends — and whether cross-basin training changes this
balance — is an open empirical question with direct practical consequences.

## 2. Background and Related Work (paper draft)

**Large-sample hydrology and CAMELS.** The CAMELS dataset (Newman et al.,
2015) — meteorological forcings, discharge, and attributes for 671 CONUS
basins — established the standard testbed for data-driven rainfall-runoff
research, later extended globally by Caravan (Kratzert et al., 2023) and
critically assessed for large-sample practice by Clerc-Schwarzenbach et al.
(2024). Calibrated outputs of process models (SAC-SMA, VIC, HBV, FUSE, mHM)
accompany the dataset, providing physics-based reference points; the lumped
conceptual model GR4J (Perrin et al., 2003) remains a widely used parsimonious
baseline.

**Deep learning rainfall-runoff.** Kratzert et al. (2018) showed a single LSTM
trained across hundreds of CAMELS basins outperforms calibrated conceptual
models; catchment-aware variants that ingest static attributes (Kratzert et
al., 2019) generalize further, and "Never train an LSTM on a single basin"
(Kratzert et al., 2024) demonstrated that regional training dominates
per-basin calibration even with long records. Gauch et al. (2021) benchmarked
trees against LSTMs under varying data volume and studied augmentation;
Frame et al. (2022) addressed extremes; Schoppa et al. (2020) characterized
random-forest behavior in large-sample hydrology.

**Operational multi-horizon forecasting.** Google's global flood system
extended reliable lead times to 7 days, rivaling GloFAS even in ungauged
basins (Nearing et al., 2024). Hunt et al. (2022) showed LSTM skill persists
to ~10 days; beyond that, studies report steep degradation (e.g., Li et al.,
2023; Islam et al., 2025). Seasonal (multi-month) prediction is traditionally
served by physically-based procedures such as ESP and SWE regression on the
western-US snowpack; observational work on storage–discharge dynamics
(Kirchner, 2009) explains why initial storage carries multi-month memory in
snow-dominated catchments. No existing work traces a *single* modeling
framework across the full 1–180 day range.

**The gap.** Three questions are open. (i) The CAMELS line evaluates almost
exclusively forcing-only, single-step simulation; the autoregressive protocol
— where past discharge is available, as in operational forecasting — has
different anchors (persistence dominates to ~7 days) and is rarely benchmarked
at scale. (ii) Kratzert et al. (2024) established regional>local for
forcing-only LSTMs; whether this transfers to autoregressive forecasting, and
whether stateless tree ensembles benefit from pooling differently than
recurrent networks, is untested. (iii) Feature engineering and data
augmentation act as *external memory* for stateless models; no study
quantifies how their value interacts with pooling and horizon. This study
addresses all three on 669 CAMELS basins over 1990–2009, forecasting at
h ∈ {1, 3, 7, 10, 15, 30, 90, 180} days with per-basin and pooled
XGBoost/LSTM variants under a leakage-safe protocol anchored by persistence
and climatology baselines.

### Reference list (to verify against DOIs at LaTeX time)

- Newman, A. et al. (2015). Development and characterization of a
  continental-scale daily watershed model and water-balance dataset. *J. Hydrometeorology* 16.
- Kratzert, F. et al. (2018). Rainfall–runoff modelling using LSTM networks. *HESS* 22, 6005.
- Kratzert, F. et al. (2019). Towards learning universal, regional, and local
  hydrological behaviors … *HESS* 23, 5089.
- Kratzert, F. et al. (2023). Caravan – a global community dataset for
  large-sample hydrology. *ESSD* 15.
- Kratzert, F. et al. (2024). Never train an LSTM on a single basin. *HESS* 28, 4187.
- Nearing, G. et al. (2024). Global prediction of extreme floods in ungauged
  watersheds. *Nature* 627, 559.
- Gauch, M. et al. (2021). The proper care and feeding of CAMELS. arXiv:1911.07249.
- Frame, J. et al. (2022). Deep learning rainfall–runoff predictions of
  extreme events. *HESS*.
- Schoppa, A. et al. (2020). Evaluating the performance of random forest for
  large-sample hydrology. *J. Hydrol.* 588.
- Hunt, E. et al. (2022). Using an LSTM to generate medium-range flood scenarios. *HESS* 26, 5449.
- Perrin, C. et al. (2003). Improvement of a parsimonious model for streamflow
  simulation (GR4J). *J. Hydrol.* 279, 275.
- Kirchner, J. (2009). Catchments as simple dynamical systems. *WRR* 45, W02429.
- Clerc-Schwarzenbach, F. et al. (2024). Large-sample hydrology – a few camels
  or a whole caravan? *HESS* 28, 4219.
- Li, Z. et al. (2023). Cascade LSTM daily streamflow forecasts. *Hydrology* 15(6), 1019.
- Cheng, M. et al. (2020). Long lead-time daily and monthly streamflow
  forecasting using ML. *J. Hydrol.*.
- Islam, M. et al. (2025). From short to long-term horizons. *Results in Eng.*

---

## 3. Task definition (engineering)

- **Lead time h**: predict daily discharge at day t+h; **input window w(h)**:
  only data ≤ t. Direct multi-horizon (no recursion, no future forcing).
- w(h): 15 for h≤7; 60 for h∈{10,15}; 90 for h=30; 270 for h∈{90,180}.
- Sub-regimes for interpretation: short-range (h≤7, autocorrelation),
  medium-range (10–30, storage anomalies), seasonal (90–180, initial state +
  climatology). Snow/rain classification: fraction of training days with
  Tmean < 0 °C (threshold to calibrate against known snow basins).

## 4. Model variants (engineering)

| variant | input | station info | notes |
|---|---|---|---|
| per-basin LSTM zscore/minmax | w×7, per-station norm | none | hidden 64×2, 40 ep, batch 1024 |
| per-basin XGB raw/fe | flatten window / same-day FE vector | none | thesis-grade fixed params |
| pooled-plain | same | **none** (Kratzert-2018 style) | shared weights |
| pooled-emb | same | nn.Embedding(669,16) per timestep | +10.7k params |
| pooled-emb-fe | +8 FE state/seasonality channels | emb | input w×(7+8+16) |
| pooled-emb-minmax | pooled-emb | emb | normalization ablation |
| xgb-pooled raw/fe | window/FE + station integer code | tree split grouping | — |
| GR4J | precip + Oudin PET | per-basin DE calibration | physics baseline |

## 5. Baselines & augmentation (engineering)

- persistence (report h≤7); **climatology = training-period day-of-year mean,
  11-day centered smoothing window** (report all h)
- augmentation (train only): meteorological jitter σ∈{0.01,0.05,0.1,0.2} in
  normalized space; time masking (10% of met timesteps zeroed, discharge
  channel untouched); variable-start subsampling. NO discharge scaling.
- robustness check: ±aug models evaluated on noise-injected test sets.

## 6. Statistics (engineering)

- paired Wilcoxon + sign test robustness; Bootstrap 95% CI (1000 resamples,
  basin-clustered); Benjamini-Hochberg FDR across the horizon×family family;
  effect sizes: rank-biserial correlation and Cohen's d.

## 7. Status log

- 2026-09-30: main phase complete (669 basins × 6 tracks × h{1,3,7}).
  Headlines: z-score vs min-max 669/669 basins (Δ+0.29); LSTM>XGB widening
  with h (+0.022→+0.028); FE ≈ 0 gain at short range (window subsumes it —
  the external-memory test moves to h≥30 with long-range FE features).
