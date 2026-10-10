# Pairs-Trading-Strategy-Backtester
A personal project for statistical arbitrage using cointegration and mean-reversion signals on equity pairs.
 
## Overview
This repo contains tools to screen equity pairs for cointegration, fit a linear regression to compute the hedge ratio of each pair, and backtest a mean-reverting strategy based on $z$-scores of the spread. Hedge ratios and half-lives are estimated on a training period only, and the strategy is evaluated out of sample to prevent lookahead bias.
 
## Strategy & Methodology
1. **Selection:** Test candidate equity pairs cointegration using the Engle-Granger two-step test. When many pairs are tested at once, p-values are adjusted using the Holm-Bonferroni method.
2. **Hedge Ratio:** Fit Ordinary Least Squares (OLS) regression on training-period prices, $Y_t = \alpha + \beta X_t + \varepsilon_t$. The hedge ratio $\beta$ is fixed after training and applied to the test period, giving the spread
   $$S_t = Y_t - (\beta X_t + \alpha)$$
3. **Half-life:** Estimate how quickly the spread reverts to its mean by regressing $\Delta S_t = a + b\,S_{t-1} + \varepsilon_t$ on the training spread. The half-life is
   $$h = -\frac{\ln 2}{\ln(1 + b)}$$
4. **Signal Generation:** Compute rolling $z$-scores of the spread,
   $$z_t = \frac{S_t - \mu_t}{\sigma_t}$$
   where $\mu_t$ and $\sigma_t$ are the rolling mean and standard deviation. The rolling window is set from the half-life rather than being fixed. Rolling statistics are computed across the train and test periods and then sliced, so the window is already wready on the first test day and no future data is used. Default strategy:
   * Enter long the spread when $z < -2$, short when $z > 2$.
   * Exit when $z$ crosses 0.
   * Positions are taken at the close of the signal day and PnL is realised at the close of the following day.
5. **Returns & Risk:** Daily PnL is divided by total capital ($Y + |\beta| X$) to give strategy returns. Annualised return and volatility use 252 trading days, and the average risk-free rate comes from the 13-week T-bill yield (`^IRX`). The Sharpe ratio is calculated under two conventions:
   Convention A subtracts the risk-free rate, while convention B does not, since the strategy returns are trading profit only and the interest is earned on top. Both are reported so the choice is explicit.
   $
   \text{Sharpe}_A = \frac{\bar r_{\text{ann}} - r_f}{\sigma_{\text{ann}}} \qquad\qquad \text{Sharpe}_B = \frac{\bar r_{\text{ann}}}{\sigma_{\text{ann}}}
   $
## Usage
```python
from backtester import backtest_pairs_trading_strategy
tickers = ["KO", "PEP"]
train_start = "2015-01-01"
train_end = "2021-12-31"
test_start = "2022-01-01"
test_end = "2026-01-01"
z_threshold = 1.5

results = backtest_pairs_trading_strategy(tickers, train_start, train_end, test_start, test_end, z_threshold)

print("\n========== BACKTEST RESULTS ==========")
print(f"Half-life:              {results['Half-life']:.1f} days")
print(f"Z-score window:         {results['Window']} days")    
print(f"Total Return:           {results['Total return']:.2f}%")
print(f"Annualised Return:      {results['Annualised return']:.2f}%")
print(f"Annualised Volatility:  {results['Annualised volatility']:.2f}%")
print(f"Risk-Free Rate:         {results['Average risk-free rate']:.2f}%")
print(f"Sharpe Ratio (minus rf):{results['Sharpe Ratio']:.2f}")
print(f"Sharpe (no rf):         {results['Sharpe Ratio (no rf)']:.2f}")
print(f"Number of Trades:       {results['n_trades']}")
print("======================================")
# window=None (default)    -> z-score window set from the half-life
# window=30                -> override with a fixed window
# half_life=31.6           -> supply your own half-life estimate
```
 
## Repo structure
```text
├── notebooks/
│   ├── cointegrated_equity_pairs.ipynb  # Initial pair selection & statistical testing
│   └── 03_pairs_regression_ideas.ipynb  # Initial regression & signal analysis
│   └── evaluations.ipynb                # Evaluations of each iteration of the different models (in progress)
│   └── cointegration_test.ipynb         # Testing the cointegration and half-life of different equity pairs (in progress)
├── src/                                 # Backtest engine and data loader (in progress)
├── results/                             # Saved picture files of graphs made using matplotlib 
├── requirements.txt                     # Dependencies
└── README.md
```
 
## Current Findings
* **Setup:** 12 candidate pairs across energy/materials, consumer, financials and technology. Models are trained on 2015-2021 and tested on 2022-2026.
* **Cointegration:** The four lowest raw p-values are GOOGL/GOOG (0.015), V/MA (0.018), KO/PEP (0.026) and BHP/RIO (0.036). None survives the Holm correction (lowest adjusted p-value 0.168), so the evidence for any single pair is weak.
* **Half-life:** Estimated half-lives range from about 25 to about 300 days. The four pairs above have the shortest (25-48 days); the slowest pairs revert too slowly for a short rolling window to make sense.
* **Backtest results:** Per-pair out-of-sample results are mixed and based on only 10-19 trades each, which is too few to distinguish from luck. Returns do not line up with cointegration strength, which is consistent with the results being largely noise or regime-driven.
* **Sharpe convention matters:** In 2022-2026 there were very high rates, and subtracting the risk-free rate (Convention A) pushes low-volatility pairs such as GOOGL/GOOG to strongly negative Sharpe ratios even when they make a profit. Convention B avoids this.
## Current Performance & Inefficiencies
* **Status:** Working single-pair backtester with statistical pair screening; multi-pair portfolio still to do.
* **Known Limitations:**
  * Spreads do not currently account for transaction costs, bid-ask spread slippage, or borrow fees for short legs.
  * The hedge ratio is static (fitted once on the training period) and is applied to prices rather than log prices, so it can go stale as the stocks drift.
  * No stop-loss, so a pair that stops moving together can produce large losses.
  * Few trades per pair, so Sharpe ratios carry wide uncertainty and no confidence intervals are reported yet.
* **Planned Improvements:**
  * Incorporate Kalman filtering for dynamic hedge ratio estimation.
  * Walk-forward re-estimation of hedge ratios, and regression on log prices.
  * Combine the pairs with the best training-period evidence into an equal-weight portfolio.
  * Build an dynamic screener to scan a universe of stocks, (potentially the S&P 500), runs a cointegration test on historical pairs and ranks them
  * Add transaction costs and a market-neutrality check (regressing strategy returns on market returns).
  * Integrate Interactive Brokers API for live signal execution.
## Setup & Requirements
```bash
pip install pandas numpy statsmodels yfinance matplotlib