"""
visualize_features.py

Generates a diagnostic report for the engineered feature matrix used by the
Adaptive Market Regime Engine. Produces:

  1. A multi-panel time series overview (price, return, volatility,
     trend, momentum) saved as a single PNG.
  2. A feature correlation heatmap.
  3. A console/log summary of matrix shape, missing values, and
     per-feature statistics.

Usage
-----
    python src/visualize_features.py \
        --input data/features/nifty50_features.csv \
        --output-dir reports/figures

Can also be imported and called programmatically:

    from visualize_features import FeatureMatrixReport

    report = FeatureMatrixReport(df)
    report.generate_all(output_dir="reports/figures")
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Style configuration
# --------------------------------------------------------------------------- #

PALETTE = {
    "price": "#1a1a2e",
    "return": "#3465a4",
    "volatility": "#c0392b",
    "trend": "#1e8449",
    "momentum": "#6c3483",
    "grid": "#d9d9d9",
    "reference": "#999999",
}

FIGURE_DPI = 160


def _apply_style() -> None:
    """Apply a consistent, presentation-ready matplotlib style."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#4d4d4d",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "legend.frameon": False,
            "figure.titlesize": 13,
            "figure.titleweight": "semibold",
        }
    )


# --------------------------------------------------------------------------- #
# Panel specification
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Panel:
    """Defines a single subplot in the feature overview report."""
    column: str
    title: str
    ylabel: str
    color: str
    reference_lines: tuple[float, ...] = field(default_factory=tuple)


DEFAULT_PANELS: tuple[Panel, ...] = (
    Panel("Close", "Price", "Close", PALETTE["price"]),
    Panel("return_1d", "Daily return", "Return", PALETTE["return"], (0.0,)),
    Panel("volatility_20d", "Realized volatility (20d)", "Std. dev.", PALETTE["volatility"]),
    Panel("price_ma20_ratio", "Trend (price / 20d MA)", "Ratio", PALETTE["trend"], (1.0,)),
    Panel("rsi_14", "Momentum (RSI-14)", "RSI", PALETTE["momentum"], (30.0, 70.0)),
)


# --------------------------------------------------------------------------- #
# Report generator
# --------------------------------------------------------------------------- #

class FeatureMatrixReport:
    """
    Builds diagnostic visualizations for an engineered feature matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix with a "Date" column and the columns referenced
        in `panels`.
    panels : tuple[Panel, ...], optional
        Subplot definitions for the overview figure. Defaults to
        price / return / volatility / trend / momentum.
    """

    def __init__(self, df: pd.DataFrame, panels: tuple[Panel, ...] = DEFAULT_PANELS):
        if "Date" not in df.columns:
            raise ValueError("Feature matrix must contain a 'Date' column.")

        missing = [p.column for p in panels if p.column not in df.columns]
        if missing:
            raise ValueError(f"Feature matrix is missing expected columns: {missing}")

        self.df = df.copy()
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.df = self.df.sort_values("Date").reset_index(drop=True)
        self.panels = panels

        _apply_style()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def generate_all(self, output_dir: str | Path) -> dict[str, Path]:
        """Generate every report artifact and return their output paths."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.log_summary()

        paths = {
            "overview": self.plot_overview(output_dir / "feature_overview.png"),
            "correlation": self.plot_correlation_heatmap(output_dir / "feature_correlation.png"),
        }

        logger.info("Report artifacts written to %s", output_dir.resolve())
        return paths

    def log_summary(self) -> None:
        """Log shape, date range, missing values, and summary statistics."""
        df = self.df
        logger.info("Feature matrix shape: %d rows x %d columns", *df.shape)
        logger.info(
            "Date range: %s -> %s (%d trading days)",
            df["Date"].min().date(),
            df["Date"].max().date(),
            len(df),
        )

        null_counts = df.isnull().sum()
        nulls = null_counts[null_counts > 0]
        if nulls.empty:
            logger.info("No missing values in feature matrix.")
        else:
            logger.warning("Missing values detected:\n%s", nulls.to_string())

        numeric_df = df.select_dtypes(include=[np.number])
        logger.info(
            "Feature summary statistics:\n%s",
            numeric_df.describe().T[["mean", "std", "min", "max"]].round(4).to_string(),
        )

    def plot_overview(self, output_path: str | Path) -> Path:
        """Render the stacked time-series overview panel and save to disk."""
        output_path = Path(output_path)
        n = len(self.panels)

        fig, axes = plt.subplots(
            n, 1, figsize=(13, 2.6 * n), sharex=True,
            gridspec_kw={"hspace": 0.35},
        )
        if n == 1:
            axes = [axes]

        for ax, panel in zip(axes, self.panels):
            ax.plot(
                self.df["Date"], self.df[panel.column],
                color=panel.color, linewidth=0.9,
            )
            for ref in panel.reference_lines:
                ax.axhline(ref, color=PALETTE["reference"], linewidth=0.7, linestyle="--")

            ax.set_title(panel.title)
            ax.set_ylabel(panel.ylabel)
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))

        axes[-1].xaxis.set_major_locator(mdates.YearLocator())
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig.align_ylabels(axes)

        fig.suptitle("Feature matrix overview", x=0.01, ha="left")
        fig.text(
            0.01, 0.985,
            f"{self.df['Date'].min().date()} \u2013 {self.df['Date'].max().date()}"
            f"   |   {len(self.df)} observations   |   {len(self.panels)} features shown",
            fontsize=9, color="#555555",
        )

        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved overview figure to %s", output_path)
        return output_path

    def plot_correlation_heatmap(self, output_path: str | Path) -> Path:
        """Render a correlation heatmap across all numeric feature columns."""
        output_path = Path(output_path)

        numeric_df = self.df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()

        fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(corr)), max(6.5, 0.5 * len(corr))))

        sns.heatmap(
            corr, ax=ax, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            square=True, linewidths=0.4, linecolor="white",
            cbar_kws={"shrink": 0.75, "label": "Pearson correlation"},
            annot_kws={"size": 7.5},
        )
        ax.set_title("Feature correlation matrix", pad=14)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)

        fig.tight_layout()
        fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved correlation heatmap to %s", output_path)
        return output_path


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate feature matrix diagnostic report.")
    parser.add_argument(
        "--input", type=str, default="data/features/nifty50_features.csv",
        help="Path to the engineered feature matrix CSV.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="reports/figures",
        help="Directory to write report figures to.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _parse_args()
    df = pd.read_csv(args.input)

    report = FeatureMatrixReport(df)
    report.generate_all(args.output_dir)


if __name__ == "__main__":
    main()