from pathlib import Path

import pandas as pd
import yfinance as yf


DEFAULT_TICKER = "^NSEI"
DEFAULT_START = "2022-01-01"
DEFAULT_END = "2026-08-01"


def download_market_data(
    ticker: str = DEFAULT_TICKER,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> pd.DataFrame:
    """
    Download historical OHLCV data from Yahoo Finance.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol.

    start : str
        Inclusive start date, YYYY-MM-DD.

    end : str
        Exclusive end date, YYYY-MM-DD.

    Returns
    -------
    pd.DataFrame
        Standardized OHLCV dataframe.
    """

    data = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        raise ValueError(
            f"No market data downloaded for {ticker}."
        )

    # yfinance can return MultiIndex columns.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()

    required_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    data = data[required_columns].copy()

    data["Date"] = pd.to_datetime(data["Date"])

    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data = data.sort_values("Date")
    data = data.drop_duplicates(
        subset="Date",
        keep="first",
    )

    data = data.reset_index(drop=True)

    return data


def save_raw_data(
    data: pd.DataFrame,
    output_path: str = "data/raw/nifty50_raw.csv",
) -> None:
    """
    Save raw market data to CSV.
    """

    output = Path(output_path)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(
        output,
        index=False,
    )


if __name__ == "__main__":

    df = download_market_data()

    save_raw_data(df)

    print("Raw market data downloaded successfully.")
    print(f"Rows: {len(df)}")
    print(f"Date range: {df['Date'].min()} -> {df['Date'].max()}")
    print(f"Saved to: data/raw/nifty50_raw.csv")