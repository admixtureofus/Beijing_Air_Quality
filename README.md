# Beijing Air Quality — Bayesian Modelling of PM2.5 and Haze Events

**Campus statistics society team project, 2026.** Four members each modelled the same data with a different method (SARIMA, XGBoost, TDA, Bayesian). This repository holds the shared preprocessing and the **Bayesian stream**, which I owned.

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
| 1 | `03_bayesian_hierarchical.py` | log PM2.5 on weather covariates; station intercepts partially pooled within urban / suburban-industrial / background groups; AR(1) errors |
| 2 | `04_bayesian_changepoint.py` | Soft-sigmoid changepoint model with crossed station and day random effects |
| 3 | `05_bayesian_haze_event.py` | Hierarchical logistic regression for haze days, last 12 months held out |

`2차 시도/` (second attempt) is the revised version after faculty feedback; `beijing/` holds the first version and its outputs.

## What changed after review

- **Priors stated and checked.** Every prior is written down with its reason, and a prior predictive check confirms it covers physically plausible PM2.5 levels.
- **Group effects made fixed.** With only three station groups, a group-level variance would be set by the prior rather than the data, so group means became fixed effects; station intercepts stay partially pooled around them.
- **AR(1) errors added.** Daily PM2.5 is strongly autocorrelated; treating days as independent would make credible intervals too narrow.
- **Time-ordered test split.** The haze model now holds out the final 12 months instead of a random split, so it is never scored on days interleaved with its training data.

## Results

- **Pooling (stage 1, LOO):** partial pooling ranks first; no pooling is close behind (ELPD difference −6, SE 3.8), and complete pooling is clearly worse (−700, SE 51). Stations differ, but not so much that they should be modelled in isolation.
- **Changepoints (stage 2):** station-level changepoint estimates cluster tightly (posterior means ≈ 19.6–19.7 on the model's time scale), pointing to a city-wide shift rather than station-specific ones.
- **Haze (stage 3, first version):** AUC 0.918, Brier 0.066 — **in-sample only**. The revised script evaluates on the held-out year; those outputs are not yet committed.

## Limitations

- Pointwise LOO under an AR(1) conditional likelihood is an approximation, not true time-series cross-validation.
- Convergence is not clean for every parameter in the revised stage 1 run: the day-level scale (`sigma_day`) shows R-hat 1.13 and low effective sample size; `gamma_group` and `beta_global` sit around R-hat 1.05–1.06. Longer chains or reparameterisation are the next step.
- With 12 stations, station-level variance components are estimated from few groups.

## Repository

```text
2차 시도/                 revised scripts (v2) and outputs_2/
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
```

## Tools

Python · PyMC · ArviZ · pandas · NumPy · scikit-learn · matplotlib / seaborn
