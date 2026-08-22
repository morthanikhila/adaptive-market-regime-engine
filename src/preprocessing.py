import pandas as pd


def clean_market_data(input_path, output_path):

    df = pd.read_csv(input_path)

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values("Date")

    df = df.drop_duplicates(subset=["Date"])

    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=numeric_columns
    )

    df = df.reset_index(drop=True)

    df.to_csv(
        output_path,
        index=False
    )

    return df


if __name__ == "__main__":

    df = clean_market_data(
        "data/raw/nifty50_raw.csv",
        "data/processed/nifty50_clean.csv"
    )

    print("Cleaned data shape:", df.shape)
    print(df.head())
    print(df.isnull().sum())