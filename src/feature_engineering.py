import pandas as pd
import numpy as np


def calculate_rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    return rsi


def create_features(input_path, output_path):

    df = pd.read_csv(input_path)

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values("Date")

    # -----------------------------
    # Returns
    # -----------------------------

    df["return_1d"] = df["Close"].pct_change()

    df["return_5d"] = df["Close"].pct_change(5)

    df["return_20d"] = df["Close"].pct_change(20)

    # -----------------------------
    # Moving averages
    # -----------------------------

    df["ma_20"] = df["Close"].rolling(20).mean()

    df["ma_50"] = df["Close"].rolling(50).mean()

    df["ma_200"] = df["Close"].rolling(200).mean()

    # -----------------------------
    # Trend features
    # -----------------------------

    df["price_ma20_ratio"] = (
        df["Close"] / df["ma_20"]
    )

    df["price_ma50_ratio"] = (
        df["Close"] / df["ma_50"]
    )

    # -----------------------------
    # Volatility
    # -----------------------------

    df["volatility_20d"] = (
        df["return_1d"]
        .rolling(20)
        .std()
    )

    df["volatility_5d"] = (
        df["return_1d"]
        .rolling(5)
        .std()
    )

    # -----------------------------
    # RSI
    # -----------------------------

    df["rsi_14"] = calculate_rsi(
        df["Close"]
    )

    # -----------------------------
    # Price range
    # -----------------------------

    df["high_low_range"] = (
        (df["High"] - df["Low"])
        / df["Close"]
    )

    df["open_close_range"] = (
        (df["Close"] - df["Open"])
        / df["Open"]
    )

    # -----------------------------
    # Volume
    # -----------------------------

    df["volume_change"] = (
        df["Volume"].pct_change()
    )

    df["volume_ma20"] = (
        df["Volume"]
        .rolling(20)
        .mean()
    )

    df["volume_ratio"] = (
        df["Volume"] / df["volume_ma20"]
    )

    # -----------------------------
    # Remove rows created by rolling
    # calculations
    # -----------------------------

    df = df.dropna()

    df = df.reset_index(drop=True)

    df.to_csv(
        output_path,
        index=False
    )

    return df


if __name__ == "__main__":

    df = create_features(
        "data/processed/nifty50_clean.csv",
        "data/features/nifty50_features.csv"
    )

    print("\nFeature matrix created.")
    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst rows:")
    print(df.head())