# Electricity Price Forecasting — Turkish Day-Ahead Market (Research)

Experiments, benchmarks, and market-analysis companion to a production
system that forecasts the EPİAŞ day-ahead Market Clearing Price (MCP/PTF)
for the Turkish electricity market. Built during a CS395 internship at the
Republic of Türkiye Ministry of Energy and Natural Resources (ETKB),
summer 2026.

**Production repo:** [Enerji-Fiyat-Tahimi](https://github.com/yusufcanbolat1/Enerji-Fiyat-Tahimi)
(ETL pipeline, live model, API, dashboard). This repo shares its database but
has its own git history, so day-to-day model code and one-off research do
not get tangled together.

## What is here

- **Model development.** Walk-forward backtests for every architecture
  tried: gradient-boosted quantile regression, a deep-learning model
  (EPNet), a hybrid router, and a multi-window LightGBM ensemble with a
  causal split-conformal band, which is what is in production today.
- **Benchmarks.** The canonical LEAR model (Lago et al., 2021) run with the
  real `epftoolbox` implementation, and a tuned weighted moving average.
  Neither beats the production model.
- **Literature review.** A citation-graph search of the electricity-price-
  forecasting field and the Turkish market specifically.
- **Market study.** An event-transmission analysis of how outside shocks
  (war, sanctions, drought) reach a price that sits behind an
  administrative cap and an administratively set gas tariff.

## Headline results

| | MAE ($/MWh) | rMAE (naive-2) |
|---|---|---|
| Production ensemble (multi-window + conformal band) | 7.04–7.15 | 0.628 |
| Single LightGBM model | 7.29–7.35 | 0.644–0.647 |
| LEAR ensemble, canonical (Lago et al., 2021) | 8.08 | 0.709 |
| Weighted moving average, tuned | 9.74 | — |

Two-year walk-forward, 2024–2026. Full detail, including the collapse-regime
breakdown and every rejected strategy, is in `EXPERIMENT_REPORT.md` and
`experiments/notebooks/`.

## Layout

```
experiments/
  notebooks/       analysis notebooks, one folder per research phase
  scripts/         backtest runners, benchmark and calibration scripts
  generators/      notebook-generation helpers
literature/        literature search (lit_search.py, citation graph)
docs/               reference PDFs (Lago, O'Connor review, competition write-ups)
src/
  models/          experimental models (EPNet, CQR calibrator)
  eval/            Lago et al. (2021) evaluation protocol
  features/        feature engineering (mirrors the production repo)
db/                 read-mostly connection to the shared PostgreSQL database
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.lock.txt
cp .env.example .env   # fill in EPİAŞ + database credentials
```

`epftoolbox`, the real reference implementation used for the LEAR
benchmark, is not in this environment (`numpy<2` and TensorFlow conflict
with the rest of the stack); it is installed in an isolated venv only for
the benchmark scripts under `experiments/notebooks/07_lago_protocol/`.

No data is bundled with this repo. Everything reads from the shared
PostgreSQL database that the production pipeline populates daily.
