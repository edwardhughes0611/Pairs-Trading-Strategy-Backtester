import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import statsmodels.api as sm
from data_loader import fetch_multiple_stocks_close

def backtest_pairs_trading_strategy(tickers: list, train_start: str, train_end: str, test_start: str, test_end: str, window: int = 30):
    """
    Backtests a pairs trading strategy using the provided tickers and date ranges.

    Parameters:
    - tickers: List of two stock tickers to be used in the pairs trading strategy.
    - train_start: Start date for the training period (format: 'YYYY-MM-DD').
    - train_end: End date for the training period (format: 'YYYY-MM-DD').
    - test_start: Start date for the testing period (format: 'YYYY-MM-DD').
    - test_end: End date for the testing period (format: 'YYYY-MM-DD').
    """


    df = fetch_multiple_stocks_close([tickers[0], tickers[1]], train_start, test_end)

    train_df = df.loc[train_start:train_end]
    test_df = df.loc[test_start:test_end].copy()

    y_train = train_df[tickers[0]]
    X_train = sm.add_constant(train_df[tickers[1]])
    model = sm.OLS(y_train, X_train).fit()

    #static edge ratio (beta) and constant (alpha)
    alpha = model.params['const']
    beta = model.params['PEP']
    print(f"Training completed. Alpha: {alpha}, Beta: {beta}")

    #Caluclate spread on out of sample data
    test_df['Spread'] = test_df[tickers[0]] - (beta * test_df[tickers[1]] + alpha)

    #Use rolling mean and std to calculate z-score
    test_df['Rolling_Mean'] = test_df['Spread'].rolling(window=window).mean()
    test_df['Rolling_Std'] = test_df['Spread'].rolling(window=window).std()
    test_df['Z_Score'] = (test_df['Spread'] - test_df['Rolling_Mean']) / test_df['Rolling_Std']

    positions = []
    current_position = 0


    for z in test_df['Z_Score']:
        if pd.isna(z):
            positions.append(current_position)
            continue

        if current_position == 0:
            if z < -2:
                current_position = 1        #Buy spread (long KO, short PEP)
            elif z > 2:
                current_position = -1       #Sell spread (short KO, long PEP)

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

    ko_diff = test_df[tickers[0]].diff()
    pep_diff = test_df[tickers[1]].diff()

    test_df['Daily PnL'] = prev_pos * (ko_diff - (beta * pep_diff))
    test_df['Capital used'] = test_df[tickers[0]].shift(1) + (abs(beta) * test_df[tickers[1]].shift(1)) #Using absolute value of beta to ensure we account for the correct capital used in both long and short positions.
    test_df['Strategy_return'] = test_df['Daily PnL'] / test_df['Capital used']
    test_df['Cumulative_return'] = (1 + test_df['Strategy_return'].fillna(0)).cumprod()

    final_multiplier = test_df['Cumulative_return'].iloc[-1]
    final_pct_return = (final_multiplier - 1) * 100

    print(f"Total out of sample returns from {test_start} to {test_end}: {final_pct_return:.2f}%")

    test_df['Cumulative_Pct_Return'] = (test_df['Cumulative_return'] - 1) * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    q = input("Show plots? (y/n): ").strip().lower()
    if q == 'y':

        # --- Subplot 1: Cumulative Returns ---
        ax1.plot(test_df.index, test_df['Cumulative_Pct_Return'], label='Strategy Return (%)', color='blue', linewidth=2)
        ax1.axhline(0, color='red', linestyle='--', linewidth=1.5, label='Baseline (0%)') # The zero baseline
        ax1.set_title(f'Pairs Trading Strategy: Cumulative Percentage Return ({test_start} to {test_end})')
        ax1.set_ylabel('Return (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.5)

        # --- Subplot 2: Spread and Trading Signals ---
        ax2.plot(test_df.index, test_df['Spread'], label=f'Spread ({tickers[0]} - beta*{tickers[1]})', color='black', alpha=0.7)
        ax2.plot(test_df.index, test_df['Rolling_Mean'], label=f'{window}-Day Mean', color='orange', linestyle='--')

        # Identify the exact days we entered a trade to plot the markers
        buys = test_df[(test_df['Position'] == 1) & (test_df['Position'].shift(1) == 0)]
        sells = test_df[(test_df['Position'] == -1) & (test_df['Position'].shift(1) == 0)]

        # Plot buy and sell signals
        ax2.scatter(buys.index, buys['Spread'], marker='^', color='green', s=120, label='Buy Signal (Z < -2)', zorder=5)
        ax2.scatter(sells.index, sells['Spread'], marker='v', color='red', s=120, label='Sell Signal (Z > 2)', zorder=5)

        ax2.set_title(f'{tickers[0]}/{tickers[1]} Spread vs. Mean Reversion')
        ax2.set_ylabel('Spread Value')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.5)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    tickers = ["KO", "PEP"]
    train_start = "2018-01-01"
    train_end = "2021-12-31"
    test_start = "2022-01-01"
    test_end = "2024-01-01"

    backtest_pairs_trading_strategy(tickers, train_start, train_end, test_start, test_end)