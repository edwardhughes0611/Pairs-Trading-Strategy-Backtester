import pandas as pd
import yfinance as yf

#fetch daily historical price data from Yahoo Finance
def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
    df = yf.download(ticker, start=start_date, end=end_date)
    return df

if __name__ == "__main__":
    df = fetch_stock_data("AAPL", "2023-01-01", "2024-01-01")
    print(df.head())