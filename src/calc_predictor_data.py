"""
Build the tidy monthly panel for the Stambaugh (1999) replication.

This file does DATA CLEANING ONLY (analysis lives elsewhere). It merges the
CRSP market index with the risk-free rate and constructs the paper's two
series:

* the excess market return  r_t = vwretd_t - rf_t   (dependent variable)
* the log dividend-price ratio x_t                  (the predictor)

The dividend-price ratio is reconstructed from index returns alone:
vwretd - vwretx = D_t / P_{t-1}. Compounding vwretx builds a price level
P_t (P_0 = 1), so D_t = (vwretd - vwretx) * P_{t-1}. Summing 12 trailing
monthly dividends gives an annual dividend D12_t, and x_t = log(D12_t / P_t).
The rolling 12-month sum makes the first 11 merged months unusable; they are
dropped, so the panel begins in mid-1927 (matching the paper's sample start).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from settings import config

DATA_DIR = Path(config("DATA_DIR"))


def build_tidy_panel(msi_df, ff_df):
    """Merge raw CRSP index and FF factor frames into the tidy monthly panel."""
    msi = msi_df.copy()
    ff = ff_df.copy()

    # CRSP stamps month-END dates; French stamps month-START. Normalize both
    # to month-end so the merge lines up.
    msi["date"] = pd.to_datetime(msi["date"]) + pd.offsets.MonthEnd(0)
    ff["date"] = pd.to_datetime(ff["date"]) + pd.offsets.MonthEnd(0)

    msi = msi[["date", "vwretd", "vwretx", "totval"]].sort_values("date")
    ff = ff[["date", "rf"]].sort_values("date")

    df = msi.merge(ff, on="date", how="inner").reset_index(drop=True)

    # Price level from the price-only return, anchored at P = 1 before row 0.
    # price_lag holds P_{t-1}, needed for the dividend back-out.
    df["price_index"] = (1.0 + df["vwretx"]).cumprod()
    df["price_lag"] = df["price_index"] / (1.0 + df["vwretx"])  # = P_{t-1}

    # Monthly dividend in index units: D_t = (vwretd - vwretx) * P_{t-1}.
    df["div_monthly"] = (df["vwretd"] - df["vwretx"]) * df["price_lag"]

    # Trailing 12-month dividend; requires a full year of history.
    df["div_ttm"] = df["div_monthly"].rolling(window=12, min_periods=12).sum()

    # The predictor: log dividend-price ratio.
    df["dp_ratio"] = df["div_ttm"] / df["price_index"]
    df["log_dp"] = np.log(df["dp_ratio"])

    # Continuously compounded excess return, as in Stambaugh's Table 1 note:
    # log(1+R_market) - log(1+R_f)
    df["ret_excess"] = np.log1p(df["vwretd"]) - np.log1p(df["rf"])

    cols = ["date", "vwretd", "vwretx", "rf", "ret_excess", "totval",
            "price_index", "div_monthly", "div_ttm", "dp_ratio", "log_dp"]
    return df[cols].dropna().reset_index(drop=True)


def load_tidy_panel(data_dir=DATA_DIR):
    """Load the tidy monthly panel from disk."""
    return pd.read_parquet(Path(data_dir) / "predictor_panel.parquet")


if __name__ == "__main__":
    from pull_CRSP_index import load_CRSP_index
    from pull_fama_french import load_fama_french_factors

    panel = build_tidy_panel(load_CRSP_index(), load_fama_french_factors())
    panel.to_parquet(Path(DATA_DIR) / "predictor_panel.parquet")
    print(f"Saved tidy panel: {len(panel)} rows, "
          f"{panel['date'].min().date()} to {panel['date'].max().date()}")
    print(panel[["date", "ret_excess", "dp_ratio", "log_dp"]].tail())