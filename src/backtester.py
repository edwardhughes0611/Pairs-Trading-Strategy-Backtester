import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from data_loader import fetch_multiple_stocks_close


df = fetch_multiple_stocks_close(["KO", "PEP"], "2020-01-01", "2024-01-01")

y=df["KO"]

X = pd.DataFrame({
    'intercept': np.ones(df.shape[0]),
    'PEP': df["PEP"]
})

model = sm.OLS(y, X).fit()

spread = model.resid
rolling_mean = spread.rolling(window=30).mean()
rolling_std = spread.rolling(window=30).std()
z_score = (spread - rolling_mean) / rolling_std


positions = []
current_position = 0


for z in z_score:
    if pd.isna(z):
        positions.append(current_position)
        continue

    if current_position == 0:
        if z < -2:
            current_position = 1
        elif z > 2:
            current_position = -1

    elif current_position == 1:
        if z >= 0:
            current_position = 0
    
    elif current_position == -1:
        if z <= 0:
            current_position = 0

    positions.append(current_position)

backtest = pd.DataFrame(index=df.index)
backtest['Position'] = positions

ko_change = df["KO"].pct_change().fillna(0)
pep_change = df["PEP"].pct_change().fillna(0)

spread_return = ko_change - pep_change

backtest['Strategy_Return'] = backtest['Position'].shift(1) * spread_return

backtest['Cumulative_Growth'] = (1 + backtest['Strategy_Return'].fillna(0)).cumprod()

final_multipler = backtest['Cumulative_Growth'].iloc[-1]
final_pct_return = (final_multipler - 1) * 100

print(f"Total compounded returns from the strategy: {final_pct_return:.2f}%")