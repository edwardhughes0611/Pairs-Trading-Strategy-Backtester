import pandas as pd
import yfinance as yf

def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
    df = yf.download(ticker, start=start_date, end=end_date)
    return df


def fetch_multiple_stocks_close(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    print(f"Downloading data for {tickers} from {start_date} to {end_date}...")
    df = yf.download(tickers, start=start_date, end=end_date)['Close']

    df = df.dropna()
    return df


if __name__ == "__main__":
    df = fetch_multiple_stocks_close(["KO", "PEP"], "2020-01-01", "2024-01-01")
    print(df)