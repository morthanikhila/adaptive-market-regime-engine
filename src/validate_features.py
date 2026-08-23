import numpy as np
import pandas as pd


FEATURE_PATH = (
    "data/features/"
    "nifty50_features.csv"
)


def validate_features():

    df = pd.read_csv(
        FEATURE_PATH,
        parse_dates=["Date"],
    )

    print("=" * 60)
    print("FEATURE MATRIX VALIDATION")
    print("=" * 60)

    # ----------------------------
    # Shape
    # ----------------------------

    print(
        f"\nRows: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    # ----------------------------
    # Dates
    # ----------------------------

    assert df["Date"].notna().all()

    assert df["Date"].is_unique

    assert df[
        "Date"
    ].is_monotonic_increasing

    print(
        "Date validation: PASS"
    )

    # ----------------------------
    # Missing values
    # ----------------------------

    missing = (
        df.isna()
        .sum()
        .sum()
    )

    assert missing == 0

    print(
        "Missing values: PASS"
    )

    # ----------------------------
    # Infinite values
    # ----------------------------

    numeric = df.select_dtypes(
        include=np.number
    )

    assert np.isfinite(
        numeric.to_numpy()
    ).all()

    print(
        "Infinite values: PASS"
    )

    # ----------------------------
    # RSI
    # ----------------------------

    assert df["rsi_14"].between(
        0,
        100,
    ).all()

    print(
        "RSI range: PASS"
    )

    # ----------------------------
    # ATR
    # ----------------------------

    assert (
        df["atr_14"] >= 0
    ).all()

    print(
        "ATR non-negative: PASS"
    )

    # ----------------------------
    # Volatility
    # ----------------------------

    assert (
        df["volatility_20d"] >= 0
    ).all()

    print(
        "Volatility non-negative: PASS"
    )

    # ----------------------------
    # MACD histogram
    # ----------------------------

    macd_difference = (
        df["macd"]
        - df["macd_signal"]
        - df["macd_histogram"]
    ).abs()

    assert (
        macd_difference.max()
        < 1e-10
    )

    print(
        "MACD consistency: PASS"
    )

    print(
        "\nALL VALIDATIONS PASSED."
    )


if __name__ == "__main__":
    validate_features()