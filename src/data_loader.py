import pandas as pd
import yfinance as yf

def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
    df = yf.download(ticker, start=start_date, end=end_date)
    return df


def fetch_multiple_stocks_close(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    data_frames = []
    for ticker in tickers:
        df = fetch_stock_data(ticker, start_date, end_date)
        df = df['Close']
        data_frames.append(df)
        combined_df = pd.concat(data_frames, axis=1)
    return combined_df

if __name__ == "__main__":
    df = fetch_multiple_stocks_close(["KO", "PEP"], "2020-01-01", "2024-01-01")
    print(df)