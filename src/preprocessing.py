from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
]


def load_raw_data(
    input_path: str,
) -> pd.DataFrame:
    """
    Load raw OHLCV CSV.
    """

    df = pd.read_csv(input_path)

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    return df


def clean_market_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Clean and validate OHLCV market data.
    """

    df = df.copy()

    # -------------------------
    # Date
    # -------------------------

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
    )

    # -------------------------
    # Numeric columns
    # -------------------------

    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # -------------------------
    # Remove invalid dates
    # -------------------------

    df = df.dropna(
        subset=["Date"]
    )

    # -------------------------
    # Sort chronologically
    # -------------------------

    df = df.sort_values("Date")

    # -------------------------
    # Remove duplicate dates
    # -------------------------

    df = df.drop_duplicates(
        subset="Date",
        keep="first",
    )

    # -------------------------
    # Remove missing OHLCV
    # -------------------------

    df = df.dropna(
        subset=numeric_columns
    )

    # -------------------------
    # Remove invalid prices
    # -------------------------

    price_columns = [
        "Open",
        "High",
        "Low",
        "Close",
    ]

    for column in price_columns:
        df = df[
            df[column] > 0
        ]

    # -------------------------
    # Volume cannot be negative
    # -------------------------

    df = df[
        df["Volume"] >= 0
    ]

    # -------------------------
    # OHLC consistency
    # -------------------------

    df = df[
        df["High"] >= df["Low"]
    ]

    df = df[
        (df["Open"] >= df["Low"])
        & (df["Open"] <= df["High"])
    ]

    df = df[
        (df["Close"] >= df["Low"])
        & (df["Close"] <= df["High"])
    ]

    # -------------------------
    # Remove infinities
    # -------------------------

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    df = df.dropna()

    # -------------------------
    # Final ordering
    # -------------------------

    df = df[
        REQUIRED_COLUMNS
    ]

    df = df.reset_index(
        drop=True
    )

    return df


def save_clean_data(
    df: pd.DataFrame,
    output_path: str = "data/processed/nifty50_clean.csv",
) -> None:

    output = Path(output_path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output,
        index=False,
    )


if __name__ == "__main__":

    raw = load_raw_data(
        "data/raw/nifty50_raw.csv"
    )

    clean = clean_market_data(raw)

    save_clean_data(clean)

    print("Preprocessing completed.")
    print(f"Rows: {len(clean)}")
    print(
        f"Date range: "
        f"{clean['Date'].min()} -> "
        f"{clean['Date'].max()}"
    )
    print("Saved to:")
    print("data/processed/nifty50_clean.csv")