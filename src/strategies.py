import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from data_loader import fetch_stock_data

def prepare_pair_data(y_ticker: str, x_ticker: str, start_date: str, end_date: str) -> pd.DataFrame:

    data1 = fetch_stock_data(y_ticker, start_date, end_date)
    data2 = fetch_stock_data(x_ticker, start_date, end_date)

    df = pd.DataFrame({
        y_ticker: data1['Close'].squeeze(),
        x_ticker: data2['Close'].squeeze()
    }).dropna()
    return df

def compute_pairs_zscore(df: pd.DataFrame, y_ticker: str, x_ticker: str, window: int = 30) -> pd.Series:
    y = df[y_ticker]
    X = pd.DataFrame({
        'intercept': np.ones(df.shape[0]),
        x_ticker: df[x_ticker]
    })

    model = sm.OLS(y, X)
    results = model.fit()
    spread = results.resid

    rolling_mean = spread.rolling(window=window).mean()
    rolling_std = spread.rolling(window=window).std()

    zscore = (spread - rolling_mean) / rolling_std
    return zscore
    
    
if __name__ == "__main__":

    df = prepare_pair_data("KO", "PEP", "2020-01-01", "2024-01-01")
    zscore = compute_pairs_zscore(df, 'KO', 'PEP', window=30)
    print(f"Z-score of the spread between KO and PEP:\n{zscore}")