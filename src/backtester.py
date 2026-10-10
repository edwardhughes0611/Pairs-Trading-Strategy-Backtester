import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import statsmodels.api as sm
from typing import Optional
from data_loader import fetch_multiple_stocks_close

def estimate_half_life(spread: pd>series) -> float:
    """
    Half-life (in trading days) of mean reversion of a spread.
 
    Regress the daily change on yesterday's level:   dS_t = a + b * S_{t-1} + noise
    If b < 0 the spread is pulled back towards its mean: each day the gap
    shrinks by a factor (1 + b). The half-life h solves (1 + b)^h = 1/2,
    so h = -ln(2) / ln(1 + b).
    """
    lag = spread.shift(1).iloc[1:]
    delta = spread.diff().iloc[1:]
    b = sm.OLS(delta, sm.add_constant(lag)).fit().params.iloc[1]
 
    if b >= 0:      # not mean reverting
        return np.inf
    if b <= -1:     # gap closes within about a day
        return 1.0
    return -np.log(2) / np.log(1 + b)    


def window_from_half_life(half_life: float, multiple: float = 2.0, min_window: int = 10, max_window: int = 252) -> int:
    """Rolling z-score window = multiple * half-life, kept within sensible bounds."""

    if not np.isfinite(half_life):
        return max_window
    
    return int(np.clip(round(multiple * half_life), min_window, max_window))


def backtest_pairs_trading_strategy(tickers: list,
                                    train_start: str,
                                    train_end: str,
                                    test_start: str, test_end: str,
                                    half_life: Optional[float] = None,
                                    window: Optional[int] = None,
                                    window_multiple: float = 2.0
                                    ):
    """
    Backtests a pairs trading strategy using the provided tickers and date ranges.
 
    Parameters:
    - tickers: List of two stock tickers to be used in the pairs trading strategy.
    - train_start / train_end: training period ('YYYY-MM-DD').
    - test_start / test_end: testing period ('YYYY-MM-DD').
    - half_life: optionally pass in the half-life from your own table so the numbers
      match; if None it is estimated here from the training spread.
    - window: rolling z-score window in days. If None (default) it is set from the
      pair's half-life: window = window_multiple * half_life.
    - window_multiple: how many half-lives the window spans.
    """

    df = fetch_multiple_stocks_close([tickers[0], tickers[1]], train_start, test_end)

    train_df = df.loc[train_start:train_end]

    y_train = train_df[tickers[0]]
    X_train = sm.add_constant(train_df[tickers[1]])
    model = sm.OLS(y_train, X_train).fit()

    #hedge ratio (beta) and constant (alpha)
    alpha = model.params['const']
    beta = model.params[tickers[1]]
    print(f"Training completed. Alpha: {alpha}, Beta: {beta}")

    if half_life is None:
        half_life = estimate_half_life(model.resid)
    if window is None:
        window = window_from_half_life(half_life, window_multiple)
    print(f"Half-life: {half_life:.1f} days, z-score window: {window} days")
    if half_life > 60:
        print("WARNING: half-life > 60 days")    

    full_df = df.copy()
    full_df['Spread'] = full_df[tickers[0]] - (beta * full_df[tickers[1]] + alpha)

    #Use rolling mean and std to calculate z-score
    full_df['Rolling_Mean'] = full_df['Spread'].rolling(window=window).mean()
    full_df['Rolling_Std'] = full_df['Spread'].rolling(window=window).std()
    full_df['Z_Score'] = (full_df['Spread'] - full_df['Rolling_Mean']) / full_df['Rolling_Std']

    test_df = full_df.loc[test_start:test_end].copy()

    positions = []
    current_position = 0


    for z in test_df['Z_Score']:
        if pd.isna(z):
            positions.append(current_position)
            continue

        if current_position == 0:
            if z < -z_threshold:
                current_position = 1        #Buy spread
            elif z > z_threshold:
                current_position = -1       #Sell spread

        elif current_position == 1:
            if z >= 0:
                current_position = 0        #Exit long position
        
        elif current_position == -1:
            if z <= 0:
                current_position = 0

        positions.append(current_position)

    test_df['Position'] = positions

    #Shift position by 1 so we trade on the close of the singal day and realise PnL the next day
    prev_pos = test_df['Position'].shift(1)

    diff0 = test_df[tickers[0]].diff()
    diff1 = test_df[tickers[1]].diff()

    test_df['Daily PnL'] = prev_pos * (diff0 - (beta * diff1))
    test_df['Capital used'] = test_df[tickers[0]].shift(1) + (abs(beta) * test_df[tickers[1]].shift(1)) #Using absolute value of beta to ensure we account for the correct capital used in both long and short positions.
    test_df['Strategy_return'] = test_df['Daily PnL'] / test_df['Capital used']
    test_df['Cumulative_return'] = (1 + test_df['Strategy_return'].fillna(0)).cumprod()

    final_multiplier = test_df['Cumulative_return'].iloc[-1]
    final_pct_return = (final_multiplier - 1) * 100

    # --- Sharpe Ratio Calculation ---
    test_df['Cumulative_Pct_Return'] = (test_df['Cumulative_return'] - 1) * 100

    mean_daily_return = test_df['Strategy_return'].mean()
    std_daily_return = test_df['Strategy_return'].std()

    annualized_return = mean_daily_return * 252
    annualized_volatility = std_daily_return * np.sqrt(252)

    rf_data = yf.download('^IRX', start=test_start, end=test_end)['Close']

    risk_free_rate = float((rf_data / 100).mean().iloc[0])

    '''
    Two methods of calculating Sharpe ratio.
      1. Subtract the T-bill rate
      2. No subtraction - the returns are trading profit only, so they are already 'excess' returns
    '''
    if annualized_volatility != 0:
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
        sharpe_no_rf = annualized_return / annualized_volatility
    else:
        sharpe_ratio = 0
        sharpe_no_rf = 0

    long_trades = ((test_df['Position'] == 1) & (test_df['Position'].shift(1) != 1)).sum()
    short_trades = ((test_df['Position'] == -1) & (test_df['Position'].shift(1) != -1)).sum()
    exits = ((test_df['Position'] == 0) & (test_df['Position'].shift(1) != 0)).sum()
    n_trades = long_trades + short_trades


    return{
        "Total return": final_pct_return,
        "Annualised return": annualized_return * 100,
        "Annualised volatility": annualized_volatility * 100,
        "Average risk-free rate": risk_free_rate * 100,
        "Sharpe Ratio": sharpe_ratio,
        "Sharpe Ratio (no rf)": sharpe_no_rf,  
        "Half-life": half_life,
        "Window": window,             
        "n_trades": n_trades
    }


if __name__ == "__main__":
    tickers = ["KO", "PEP"]
    train_start = "2015-01-01"
    train_end = "2021-12-31"
    test_start = "2022-01-01"
    test_end = "2026-01-01"
    z_threshold = 1.5
    window = 60

    results = backtest_pairs_trading_strategy(tickers, train_start, train_end, test_start, test_end, window, z_threshold)

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