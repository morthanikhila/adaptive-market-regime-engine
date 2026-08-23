from pathlib import Path

import numpy as np
import pandas as pd


def calculate_rsi(
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Calculate RSI using Wilder's smoothing.
    """

    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    rs = avg_gain / avg_loss

    rsi = 100 - (
        100 / (1 + rs)
    )

    return rsi


def calculate_atr(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.Series:
    """
    Calculate Average True Range using
    Wilder's smoothing.
    """

    previous_close = df["Close"].shift(1)

    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - previous_close).abs(),
            (df["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    return atr


def create_features(
    input_path: str,
    output_path: str,
) -> pd.DataFrame:
    """
    Create the complete feature matrix.
    """

    df = pd.read_csv(
        input_path
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df = df.sort_values(
        "Date"
    )

    df = df.reset_index(
        drop=True
    )

    # ==================================================
    # 1. RETURN
    # ==================================================

    df["return"] = (
        df["Close"]
        .pct_change()
    )

    # ==================================================
    # 2. LOG RETURN
    # ==================================================

    df["log_return"] = np.log(
        df["Close"]
        / df["Close"].shift(1)
    )

    # ==================================================
    # 3. 20-DAY VOLATILITY
    # ==================================================

    df["volatility_20d"] = (
        df["log_return"]
        .rolling(window=20)
        .std()
    )

    # ==================================================
    # 4. RSI-14
    # ==================================================

    df["rsi_14"] = calculate_rsi(
        df["Close"],
        period=14,
    )

    # ==================================================
    # 5. MACD
    # ==================================================

    ema_12 = (
        df["Close"]
        .ewm(
            span=12,
            adjust=False,
        )
        .mean()
    )

    ema_26 = (
        df["Close"]
        .ewm(
            span=26,
            adjust=False,
        )
        .mean()
    )

    df["macd"] = (
        ema_12 - ema_26
    )

    # ==================================================
    # 6. MACD SIGNAL
    # ==================================================

    df["macd_signal"] = (
        df["macd"]
        .ewm(
            span=9,
            adjust=False,
        )
        .mean()
    )

    # ==================================================
    # 7. MACD HISTOGRAM
    # ==================================================

    df["macd_histogram"] = (
        df["macd"]
        - df["macd_signal"]
    )

    # ==================================================
    # 8. ATR-14
    # ==================================================

    df["atr_14"] = calculate_atr(
        df,
        period=14,
    )

    # ==================================================
    # 9. VOLUME CHANGE
    # ==================================================

    previous_volume = (
        df["Volume"].shift(1)
    )

    df["volume_change"] = np.where(
        previous_volume > 0,
        (
            df["Volume"]
            / previous_volume
        ) - 1,
        0.0,
    )

    # ==================================================
    # 10. VOLUME MA-20
    # ==================================================

    df["volume_ma_20"] = (
        df["Volume"]
        .rolling(window=20)
        .mean()
    )

    # ==================================================
    # Remove rows where indicators are not available
    # ==================================================

    feature_columns = [
        "return",
        "log_return",
        "volatility_20d",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_histogram",
        "atr_14",
        "volume_change",
        "volume_ma_20",
    ]

    df = df.dropna(
        subset=feature_columns
    )

    # ==================================================
    # Replace any remaining infinities
    # ==================================================

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    df = df.dropna(
        subset=feature_columns
    )

    # ==================================================
    # Final validation
    # ==================================================

    numeric_columns = [
        column
        for column in df.columns
        if column != "Date"
    ]

    if not np.isfinite(
        df[numeric_columns]
        .to_numpy()
    ).all():

        raise ValueError(
            "Feature matrix contains "
            "NaN or infinite values."
        )

    # ==================================================
    # Save
    # ==================================================

    output = Path(
        output_path
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output,
        index=False,
    )

    return df


if __name__ == "__main__":

    features = create_features(
        "data/processed/nifty50_clean.csv",
        "data/features/nifty50_features.csv",
    )

    print(
        "\nFeature engineering completed."
    )

    print(
        f"Rows: {len(features)}"
    )

    print(
        f"Columns: {len(features.columns)}"
    )

    print(
        f"Date range: "
        f"{features['Date'].min()} -> "
        f"{features['Date'].max()}"
    )

    print(
        "\nFeature columns:"
    )

    print(
        features.columns.tolist()
    )

    print(
        "\nFinal dataset shape:"
    )

    print(
        features.shape
    )