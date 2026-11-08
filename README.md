# Beijing Air Quality — Bayesian Modelling of PM2.5 and Haze Events

**Campus statistics society team project, 2026.** Four members each modelled the same data with a different method (classical statistics, XGBoost, TDA, Bayesian). This repository holds the shared preprocessing and the **Bayesian stream**, which I owned.

Data: [UCI Beijing Multi-Site Air-Quality](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data) — hourly pollutant and weather readings from 12 stations, March 2013 to February 2017.

## Questions

1. How much of the variation in PM2.5 is shared across the city, and how much is station-specific?
2. When does pollution shift level at each station?
3. Can haze days be predicted on data the model has not seen?

## Pipeline

| Stage | Script | Model |
|---|---|---|
| Preprocess | `01_preprocess.py` | Merge 12 station files, handle missing values, daily aggregation |
| EDA | `02_eda_viz.py` | Distributions, seasonality, station contrasts (`beijing/outputs/figures`) |
| 1 | `03_bayesian_hierarchical.py` | log PM2.5 on weather covariates; station intercepts partially pooled within urban / suburban-industrial / background groups; AR(1) errors and a day-level shared effect |
| 2 | `04_bayesian_changepoint.py` | Soft-sigmoid changepoint model with a month-level shared effect and AR(1) errors |
| 3 | `05_bayesian_haze_event.py` | Hierarchical logistic regression for haze days, last 12 months held out |

`final/` holds the current scripts and all three stages of output. `beijing/` holds the original version, its EDA, and the raw station data.

## The finding the whole stream turns on

**The 12 stations are not 12 independent samples.** Decomposing the variance with a day-level shared effect gives σ_day ≈ 0.71 against σ_e ≈ 0.172, so **94% of daily variation is a city-wide component** and only 6% is station-specific. The monthly model reproduces this independently at 94.8%. On top of that, residuals carry ρ ≈ 0.40 day to day.

Treating the stations as independent therefore inflates the effective sample size for any day-level quantity by up to 12×. That is not a technicality: it overturned the stage 2 conclusion (below).

## What changed after review

- **Priors stated and checked.** Every prior is written down with its reason, and a prior predictive check confirms it covers physically plausible PM2.5 levels.
- **Group effects made fixed.** With only three station groups, a group-level variance would be set by the prior rather than the data, so group means became fixed effects; station intercepts stay partially pooled around them.
- **AR(1) errors added.** Daily PM2.5 is strongly autocorrelated; treating days as independent would make credible intervals too narrow.
- **Repeated-measures structure added.** A shared day-level (month-level in stage 2) random effect separates the component all 12 stations move on together from station-specific variation.
- **Time-ordered test split.** The haze model holds out the final 12 months instead of a random split, so it is never scored on days interleaved with its training data.
- **Sum-to-zero constraint on the shared effect.** Leaving the day effects unconstrained lets their mean trade off against the global intercept along a ridge the sampler cannot traverse, which is what caused the stage 1 convergence failure. `pm.ZeroSumNormal` removes it — on matched simulated data, R-hat went 1.126 → 1.018 and minimum ESS 28 → 477, with point estimates unchanged.

## Results

**Stage 1 — stations differ in level, not in mechanism.** Baseline PM2.5 runs from 46.8 µg/m³ (Huairou, background) to 62.6 µg/m³ (Dongsi, urban), a factor of 1.33, and the ordering matches the raw station means exactly. Weather coefficients, by contrast, are nearly identical across stations (between-station heterogeneity 0.03–0.07): the mechanism by which weather drives PM2.5 is the same city-wide, and what differs is each area's background level. LOO ranks partial pooling first, with no pooling close behind (ELPD difference −6, dse 3.8) and complete pooling clearly worse (−700, dse 51).

**Stage 2 — no single step change is identifiable.** All 12 stations have a negative posterior mean for the level shift, averaging δ = −0.112 (≈ 10.6% lower), but the 94% HDI is −0.475 to +0.125 and P(δ < 0) is only 0.73. The changepoint itself is not identified at all: τ = 21.63 with a 94% HDI of −4.61 to 54.79, which covers the entire 48-month window.

This is a limit of the data, not a broken model. A power check on simulated data matched to the real conditions recovers a δ = −0.40 shift accurately (τ̂ = 20.2 against a true 20, interval excluding zero), but fails on a δ = −0.11 shift (P(δ < 0) = 0.57) — the size actually estimated here. The same simulation shows that dropping the shared monthly effect narrows the interval on δ by a factor of 2.7, which is exactly how the earlier version of this analysis arrived at a confident "city-wide simultaneous decline."

**Stage 3 — haze days are predictable a day ahead.** On the held-out final 12 months, AUC 0.956 and Brier 0.053 (4,392 days, base rate 12.50%), against AUC 0.934 and Brier 0.063 in the training period (13,128 days, base rate 12.61%). Both figures use marginal predictive probabilities that integrate the unknown day effect out of the prior, so neither is an in-sample fit. Six of eight predictors exclude zero — humid, cold, still days and days following a high reading carry the risk — while rainfall and pressure do not once the day effect is controlled for.

## Reading the numbers correctly

- **Odds ratios in stage 3 are conditional, not marginal.** With σ_day ≈ 4.2 on the logit scale, the approximation β_marginal ≈ β / √(1 + 0.346 σ²) implies a shrinkage factor of about 2.7. Dew point's conditional odds ratio of 168.6 corresponds to roughly 6.9 marginally. Use the marginal figures when comparing against a model without random effects.
- **Calibration drifts above 0.3.** In both periods the observed event rate sits above the diagonal in the 0.3–0.9 band, so the model under-warns on high-risk days. The low-probability region, which is most of the data, is well calibrated.
- **Interval conventions differ by stage.** Stage 2 and 3 report 94% HDI. The committed stage 1 summaries predate the `ci_prob` setting and are 89% ETI; do not compare interval widths across stages.

## Limitations

- **Stage 1 has not been re-run since the sum-to-zero fix.** The committed stage 1 outputs come from the run that failed convergence (`sigma_day` R-hat 1.13, minimum ESS 18; `gamma_group` and `beta_global` around 1.05–1.06). Point estimates and the station ordering are reliable — the fix left them unchanged on simulated data — but the intervals and diagnostics should be replaced by a re-run.
- **Weather coefficients in stage 1 share explanatory power with the day effect.** Weather is a day-level variable, so its coefficients are conditional on the shared effect rather than total. `USE_BETWEEN_WITHIN` in the script splits each covariate into a city-daily mean and a station deviation to separate the two; that run is not committed.
- **Stage 2 tests a single step change only.** A gradual four-year decline would not be captured by a sigmoid step. `USE_LINEAR_TREND` provides a trend specification for comparison; it has not been run.
- **Pointwise LOO under an AR(1) conditional likelihood is an approximation**, not true time-series cross-validation, and 29 observations exceed the Pareto k̂ = 0.7 threshold. Treat the model ranking as indicative.
- **No causal claim is supported.** A single city with a single intervention gives no control group, and the 2014 APEC restrictions and staged heating conversions overlap the policy window.
- **With 12 stations**, station-level variance components are estimated from few groups.

## Repository

```text
final/                    current scripts and outputs
├── 01_preprocess.py
├── 03_bayesian_hierarchical.py
├── 04_bayesian_changepoint.py
├── 05_bayesian_haze_event.py
└── outputs/
    ├── stage1/           hierarchical regression (pre-fix run, see Limitations)
    ├── stage2/           changepoint detection
    └── stage3/           haze logistic regression and holdout evaluation
beijing/
├── archive/              raw station CSVs (UCI)
├── 01_preprocess.py
├── 02_eda_viz.py
├── 03_bayesian_hierarchical/
├── 04_bayesian_changepoint/
├── 05_bayesian_haze_event/
├── outputs/              EDA figures and pollutant summary
└── data_outputs.zip      merged and cleaned CSVs
README_KO.md              EDA notes (Korean)
.gitignore
```

Each script carries its prior specification and option switches (`USE_AR1`, `USE_DAY_EFFECT`, `USE_BETWEEN_WITHIN`, `USE_LINEAR_TREND`) at the top, so individual structural assumptions can be turned off and the effect on the intervals observed directly.

## Tools

Python · PyMC (NUTS) · ArviZ · pandas · NumPy · scikit-learn · matplotlib / seaborn
