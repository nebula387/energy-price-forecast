# Day-ahead electricity price forecasting under the 2026 Iran-war gas shock (Germany)

Forecasting hourly German (DE-LU) day-ahead power prices, and measuring what the war that began on
**28 February 2026** did to them and to forecasting models trained in "normal" times.

> **What happened.** On 28 Feb 2026 US/Israeli strikes on Iran started a war that closed the Strait of Hormuz
> (≈20 % of world LNG). Qatar halted LNG output on 2 Mar, Ras Laffan was hit on 18 Mar, a ceasefire on 8 Apr
> did not restore shipping, and hostilities re-escalated in July. Germany buys almost no Qatari LNG directly,
> but it buys gas at European (TTF) prices, and gas plants usually set the German power price.
> Event dates: [`src/events.py`](src/events.py), with sources.

![Price and gas 2026](reports/figures/01b_zoom_2026.png)

## Key results

Crisis window **28 Feb – 13 Oct 2026**, compared with **the same calendar window of 2025** (to remove seasonality):

| | same window 2025 | crisis 2026 | change |
|---|---|---|---|
| Mean day-ahead price, €/MWh | 79.7 | 111.3 | **+40 %** |
| Hours above 150 €/MWh | 4.5 % | 26.9 % | ×6 |
| Mean daily spread (max − min), €/MWh | 146 | 182 | +25 % |
| TTF gas (as known at auction), €/MWh | 35.3 | 55.2 | +56 % |
| German gas storage fill | 49.6 % | 38.0 % | −12 pp |

* **Gas explains ~60 % of the price increase** (+19.5 of +31.6 €/MWh, from a regression on 2023 – Feb 2026).
  More solar pulled prices down by ~8 €/MWh. The remaining ~20 €/MWh is a crisis premium plus drivers not in the
  data (EU carbon price, imports).
* **The supply curve shifted up by 35–50 €/MWh** at the same residual load.
* **Evenings, not middays:** evening hours rose ~50 €/MWh, solar-noon hours ~9 €/MWh.

### Models: a model frozen before the war breaks

Walk-forward backtest, MAE in €/MWh ([`reports/metrics.csv`](reports/metrics.csv)):

| Model | normal year (Mar 2025 – Feb 2026) | crisis (28 Feb 2026 –) |
|---|---|---|
| Naive (same hour last week) | 29.2 | 42.9 |
| LightGBM, trained once before the window | 13.7 | **31.0** (×2.3) |
| LightGBM + gas features, trained once | 15.6 | 30.9 |
| Linear + gas features, trained once | 16.1 | 21.4 |
| **LightGBM + gas features, retrained weekly** | **11.8** | **17.4** |

![MAE normal vs crisis](reports/figures/07_mae_normal_vs_crisis.png)

Having gas features is not enough on its own. **Tree models cannot extrapolate**: once TTF moved beyond the range
seen in training, the static LightGBM kept forecasting pre-war prices. Its bias grew to +50 €/MWh by September
([figure](reports/figures/09_bias.png)). Weekly retraining fixes most of it.

### What is a forecast worth? (1 MW / 1 MWh battery)

One charge/discharge cycle per day, hours chosen from the forecast, revenue settled at actual prices.

| | same window 2025 | crisis 2026 |
|---|---|---|
| Perfect-foresight revenue, € per day | 126 | 156 (**+23 %**) |
| Capture rate: LightGBM rolling | 96.8 % | 97.6 % |
| Capture rate: naive | 87.9 % | 91.8 % |

The battery needs the **shape** of the day (where the trough and the peak are), not the price level. Even the broken
static model captures ~97 % of the perfect revenue. The crisis made flexibility more valuable.

## Data

Only real, observed data is used. **No third-party price forecasts** (exchange, vendor or website "predictions") are
used anywhere. The model inputs are the same information a trader has at auction time.

| Source | Series | Used as |
|---|---|---|
| [ENTSO-E Transparency](https://transparency.entsoe.eu) | day-ahead prices DE-LU | target, price lags |
| ENTSO-E | TSO day-ahead load, wind & solar forecasts | features |
| ENTSO-E | actual load | analysis only |
| [AGSI+](https://agsi.gie.eu) | German gas storage | feature (lag 3 days) |
| Yahoo Finance (unofficial) | TTF front-month; KRBN as a rough CO₂ proxy | features (lag 2 days) |
| [Open-Meteo](https://open-meteo.com) archive | actual weather | analysis only |

## Approach: no look-ahead

The day-ahead auction for day D closes at **12:00 CET on D-1**. Every feature must be known by then:

* price lags are taken by (delivery day, local hour), so the latest is day D-1, which is fully known. DST days are handled;
* TTF/CO₂ use the close of **D-2** (D-1's close comes after the auction);
* AGSI storage uses gas day **D-3** (published the evening after the gas day);
* actual load and actual weather are never model inputs.

These rules are enforced by tests: [`tests/test_leakage.py`](tests/test_leakage.py).
In the backtest every day is forecast only from rows with delivery day < D.

## Reproduce

```bash
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -r requirements.txt
cp .env.example .env                               # add ENTSOE_API_KEY and AGSI_API_KEY

python -m src.fetch_entsoe          # ~20 min first time, cached per month afterwards
python -m src.fetch_weather
python -m src.fetch_gas_storage
python -m src.check_market_tickers
python -m src.build_dataset         # → data/processed/dataset.parquet
python -m pytest -q                 # leakage tests
python -m src.backtest              # → reports/metrics.csv (~4 min)
python -m src.battery               # → reports/battery.csv
```

Notebooks: [`01_eda.ipynb`](notebooks/01_eda.ipynb) (what the war changed) and
[`02_models.ipynb`](notebooks/02_models.ipynb) (models, bias, battery).

## Limitations

* **No EU carbon (EUA) price.** KRBN is a global carbon ETF, only a rough proxy. Part of the "unexplained" crisis premium is probably carbon.
* **TTF from Yahoo Finance** is unofficial (96 % business-day coverage checked in `check_market_tickers`).
* **TSO wind/solar forecasts**: ENTSO-E allows publication up to 18:00 on D-1, i.e. possibly after the auction.
  Using them is standard practice in the price-forecasting literature, but strictly it is a slightly optimistic assumption.
* Since 1 Oct 2025 the auction clears in 15-minute products. Everything here is hourly averages.
* The crisis window is 7.5 months of one market. The battery model is simple: one cycle per day, price-taker, no degradation.

## Structure

```
src/
  config.py               zone, dates, publication lags
  events.py               2026 war timeline (annotation only)
  fetch_*.py              data download (ENTSO-E, Open-Meteo, AGSI+, Yahoo)
  build_dataset.py        hourly table with leakage-safe daily joins
  features.py             model features
  backtest.py             walk-forward backtest, static vs rolling
  battery.py              battery arbitrage value of a forecast
  plots.py                chart style
tests/test_leakage.py     no-look-ahead tests
notebooks/                EDA and models
reports/                  metrics, battery results, figures
docs/WORKFLOW_ru.md       working notes (Russian)
```
