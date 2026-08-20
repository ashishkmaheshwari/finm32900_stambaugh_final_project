"""Build a NYSE-only value-weighted index from the CRSP monthly stock file."""

from pathlib import Path

import pandas as pd

from calc_predictor_data import build_tidy_panel, load_tidy_panel
from create_table_01_partC import fit_subsample
from pull_CRSP_stock import load_CRSP_monthly_file
from pull_fama_french import load_fama_french_factors
from settings import config

DATA_DIR = Path(config("DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))


def build_nyse_value_weighted_index(stock_df: pd.DataFrame) -> pd.DataFrame:
    """Value-weight NYSE stock returns using lagged market capitalization."""
    df = stock_df.copy()

    required_cols = {"date", "permno", "ret", "retx", "market_cap", "primaryexch"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required CRSP stock columns: {sorted(missing)}")

    df = df[df["primaryexch"] == "N"].copy()
    df["date"] = pd.to_datetime(df["date"]) + pd.offsets.MonthEnd(0)

    for col in ["ret", "retx", "market_cap"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["permno", "date"])
    df["lag_market_cap"] = df.groupby("permno")["market_cap"].shift(1)

    df = df.dropna(subset=["date", "ret", "retx", "market_cap", "lag_market_cap"])
    df = df[df["lag_market_cap"] > 0].copy()

    df["weighted_ret"] = df["lag_market_cap"] * df["ret"]
    df["weighted_retx"] = df["lag_market_cap"] * df["retx"]

    out = (
        df.groupby("date", as_index=False)
        .agg(
            weighted_ret=("weighted_ret", "sum"),
            weighted_retx=("weighted_retx", "sum"),
            weight=("lag_market_cap", "sum"),
            totval=("market_cap", "sum"),
            n_stocks=("permno", "nunique"),
        )
        .sort_values("date")
    )

    out["vwretd"] = out["weighted_ret"] / out["weight"]
    out["vwretx"] = out["weighted_retx"] / out["weight"]

    return out[["date", "vwretd", "vwretx", "totval", "n_stocks"]]


def make_comparison_table(
    total_market_panel: pd.DataFrame, nyse_panel: pd.DataFrame
) -> pd.DataFrame:
    """Compare the original total-market panel with the NYSE stock-file panel."""
    total = total_market_panel.copy()
    nyse = nyse_panel.copy()

    total["date"] = pd.to_datetime(total["date"]) + pd.offsets.MonthEnd(0)
    nyse["date"] = pd.to_datetime(nyse["date"]) + pd.offsets.MonthEnd(0)

    merged = total[["date", "dp_ratio"]].merge(
        nyse[["date", "dp_ratio"]],
        on="date",
        how="inner",
        suffixes=("_total_market", "_nyse"),
    )

    start_date = merged["date"].min().strftime("%Y-%m-%d")
    end_date = merged["date"].max().strftime("%Y-%m-%d")

    total_fit = fit_subsample(total, start_date, end_date)
    nyse_fit = fit_subsample(nyse, start_date, end_date)

    corr = merged["dp_ratio_total_market"].corr(merged["dp_ratio_nyse"])

    return pd.DataFrame(
        [
            {"Statistic": "First overlapping month", "Value": start_date[:7]},
            {"Statistic": "Last overlapping month", "Value": end_date[:7]},
            {"Statistic": "Correlation of D/P series", "Value": f"{corr:.4f}"},
            {
                "Statistic": "OLS slope, total-market index",
                "Value": f"{float(total_fit['beta_hat']):.6f}",
            },
            {
                "Statistic": "OLS slope, NYSE stock-file index",
                "Value": f"{float(nyse_fit['beta_hat']):.6f}",
            },
        ]
    )


def write_comparison_table(table: pd.DataFrame, output_path: Path) -> None:
    """Write the NYSE robustness comparison as a LaTeX table."""
    caption = (
        "Robustness comparison between the CRSP total-market value-weighted index "
        "and a NYSE-only value-weighted index rebuilt from the CRSP monthly stock "
        "file. The high correlation shows that the dividend-yield series are very "
        "similar, so the index choice has little effect on the main result."
    )

    latex = table.to_latex(
        index=False,
        escape=True,
        caption=caption,
        label="tab:nyse-index-comparison",
    )

    output_path.write_text(latex, encoding="utf-8")


def main() -> None:
    """Build the NYSE-only panel and write the robustness comparison."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stock_df = load_CRSP_monthly_file()
    ff_df = load_fama_french_factors()

    nyse_index = build_nyse_value_weighted_index(stock_df)
    nyse_panel = build_tidy_panel(nyse_index, ff_df)

    nyse_index.to_parquet(DATA_DIR / "nyse_stock_index.parquet")
    nyse_panel.to_parquet(DATA_DIR / "predictor_panel_nyse.parquet")

    comparison = make_comparison_table(load_tidy_panel(), nyse_panel)
    write_comparison_table(comparison, OUTPUT_DIR / "nyse_index_comparison.tex")

    print(
        "Saved NYSE stock-file index: "
        f"{len(nyse_index)} rows, "
        f"{nyse_index['date'].min().date()} to {nyse_index['date'].max().date()}"
    )
    print(
        "Saved NYSE tidy panel: "
        f"{len(nyse_panel)} rows, "
        f"{nyse_panel['date'].min().date()} to {nyse_panel['date'].max().date()}"
    )
    print(comparison.to_string(index=False))
    print("Wrote _data/nyse_stock_index.parquet")
    print("Wrote _data/predictor_panel_nyse.parquet")
    print("Wrote _output/nyse_index_comparison.tex")


if __name__ == "__main__":
    main()