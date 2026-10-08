# Pairs-Trading-Strategy-Backtester
A personal project for statistical arbitrage using cointegration and mean-reversion signals on equity pairs.

## Overview
This repo contains tools to fit linear regression models to compute the head ratios of cointegrated equity pairs, and backtest a mean-reverting strategy based on $z$-scors.

## Strategy & Methodology
1. **Selection:** Search candidate equity pairs for stationarity in their price spread using the Engle-Granger two-step test.
2. **Hedge Ratio:** Fit Ordinary Least Squares (OLS) regression on log prices to determine the dynamic spread.
3. **Signal Generation:** Compute rolling $z$-scores of the spread to generate long/short entry and exit signals.

## Repo structure
```text
├── notebooks/
│   ├── cointegrated_equity_pairs.ipynb  # Pair selection & statistical testing (in progress)
│   └── 03_pairs_regression_idead.ipynb  # Initial regression & signal analysis
│   └── evaluations.ipynb                # Evalutions of each iteration of the different models (in progress)
├── src/                                 # Modular backtesting code (in progress)
├── results/                             # Saved picture files of graphs made using matplotlib 
├── requirements.txt                     # Dependencies
└── README.md
```

## Current Performance & Inefficiencies
* **Status:** Initial prototype phase.
* **Known Limitations:**
  * Iterative loops in the current backtest logic lead to slow runtime on long historical timeframes.
  * Spreads do not currently account for transaction costs, bid-ask spread slippage, or borrow fees for short legs.
* **Planned Improvements:**
  * Vectorize spread calculation and trade signal generation using `numpy` and `pandas`.
  * Incorporate Kalman filtering for dynamic hedge ratio estimation.
  * Integrate Interactive Brokers API for live signal execution.

## Setup & Requirements
```bash
pip install pandas numpy statsmodels yfinance matplotlib 
