import yfinance as yf
import pandas as pd


def download_market_data(
    ticker="^NSEI",
    start="2020-01-01",
    end="2026-08-01"
):
    data = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True
    )

    if data.empty:
        raise ValueError("No market data downloaded.")

    data = data.reset_index()

    return data


if __name__ == "__main__":
    data = download_market_data()

    print(data.head())
    print(data.shape)

    data.to_csv(
        "data/raw/nifty50_raw.csv",
        index=False
    )

    print("Raw data saved successfully.")