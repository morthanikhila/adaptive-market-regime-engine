import numpy as np
import pandas as pd
import pytest


FEATURE_FILE = (
    "data/features/nifty50_features.csv"
)


@pytest.fixture
def df():
    data = pd.read_csv(
        FEATURE_FILE
    )

    data["Date"] = pd.to_datetime(
        data["Date"]
    )

    return data


# ==========================================================
# BASIC DATASET TESTS
# ==========================================================


def test_file_not_empty(df):

    assert not df.empty


def test_dates_are_valid(df):

    assert df["Date"].notna().all()


def test_dates_are_unique(df):

    assert df["Date"].is_unique


def test_dates_are_sorted(df):

    assert df["Date"].is_monotonic_increasing


def test_no_missing_values(df):

    assert not df.isna().any().any()


# ==========================================================
# EXPECTED FEATURES
# ==========================================================


def test_expected_features_exist(df):

    expected = [
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

    for column in expected:
        assert column in df.columns


# ==========================================================
# FINITE VALUES
# ==========================================================


def test_no_infinite_values(df):

    numeric_df = (
        df.select_dtypes(
            include=np.number
        )
    )

    assert np.isfinite(
        numeric_df.to_numpy()
    ).all()


# ==========================================================
# RETURN
# ==========================================================


def test_return_formula(df):

    expected = (
        df["Close"]
        / df["Close"].shift(1)
    ) - 1

    actual = df["return"]

    comparison = pd.concat(
        [
            expected,
            actual,
        ],
        axis=1,
    ).dropna()

    differences = (
        comparison.iloc[:, 0]
        - comparison.iloc[:, 1]
    ).abs()

    assert differences.max() < 1e-10


# ==========================================================
# LOG RETURN
# ==========================================================


def test_log_return_formula(df):

    expected = np.log(
        df["Close"]
        / df["Close"].shift(1)
    )

    actual = df["log_return"]

    comparison = pd.concat(
        [
            expected,
            actual,
        ],
        axis=1,
    ).dropna()

    differences = (
        comparison.iloc[:, 0]
        - comparison.iloc[:, 1]
    ).abs()

    assert differences.max() < 1e-10


# ==========================================================
# VOLATILITY
# ==========================================================


def test_volatility_20d_formula(df):

    log_returns = np.log(
        df["Close"]
        / df["Close"].shift(1)
    )

    expected = (
        log_returns
        .rolling(20)
        .std()
    )

    actual = df["volatility_20d"]

    comparison = pd.concat(
        [
            expected,
            actual,
        ],
        axis=1,
    ).dropna()

    differences = (
        comparison.iloc[:, 0]
        - comparison.iloc[:, 1]
    ).abs()

    assert differences.max() < 1e-10


def test_volatility_non_negative(df):

    assert (
        df["volatility_20d"] >= 0
    ).all()


# ==========================================================
# RSI
# ==========================================================


def test_rsi_range(df):

    assert (
        df["rsi_14"]
        .between(0, 100)
        .all()
    )


def test_rsi_has_variation(df):

    assert (
        df["rsi_14"].nunique()
        > 1
    )


# ==========================================================
# MACD
# ==========================================================


def test_macd_histogram_formula(df):

    expected = (
        df["macd"]
        - df["macd_signal"]
    )

    difference = (
        expected
        - df["macd_histogram"]
    ).abs()

    assert difference.max() < 1e-10


# ==========================================================
# ATR
# ==========================================================


def test_atr_non_negative(df):

    assert (
        df["atr_14"] >= 0
    ).all()


def test_atr_has_variation(df):

    assert (
        df["atr_14"].nunique()
        > 1
    )


# ==========================================================
# VOLUME
# ==========================================================


def test_volume_change_is_finite(df):

    assert np.isfinite(
        df["volume_change"]
    ).all()


def test_volume_ma20_non_negative(df):

    assert (
        df["volume_ma_20"] >= 0
    ).all()


def test_volume_ma20_has_variation(df):

    assert (
        df["volume_ma_20"].nunique()
        > 1
    )