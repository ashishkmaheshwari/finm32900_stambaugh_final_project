"""Create summary statistics table and underlying-data chart for the Stambaugh project."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from calc_predictor_data import load_tidy_panel

OUTPUT_DIR = Path("_output")

STAMBAUGH_WINDOWS = {
    "Full sample": (None, None),
    "1927-1996": ("1927-01-01", "1996-12-31"),
    "1927-1951": ("1927-01-01", "1951-12-31"),
    "1952-1996": ("1952-01-01", "1996-12-31"),
    "1977-1996": ("1977-01-01", "1996-12-31"),
}

RETURN_COLUMNS = (
    "excess_return",
    "monthly_excess_return",
    "ret_excess",
    "excess_ret",
    "rx",
    "r_excess",
    "mkt_excess",
    "vwretd_excess",
    "r",
)

DP_COLUMNS = (
    "log_dp",
    "dp",
    "log_d_p",
    "d_p",
    "ln_dp",
    "ln_d_p",
    "log_dividend_price",
    "log_dividend_price_ratio",
    "x",
)

DATE_COLUMNS = ("date", "month", "caldt", "time", "period")

RECESSION_COLUMNS = ("usrec", "USREC", "recession", "nber_recession")


def _find_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    """Find the first available column from a list of candidate names."""
    normalized = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    raise KeyError(
        f"Could not find any of {candidates}. Available columns are: {list(df.columns)}"
    )


def _find_optional_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """Find an optional column, returning None when no candidate is available."""
    normalized = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    return None


def _prepare_panel() -> pd.DataFrame:
    """Load and standardize the tidy predictor panel used by the project."""
    panel = load_tidy_panel().copy()

    if isinstance(panel.index, pd.DatetimeIndex):
        panel = panel.reset_index()
        panel = panel.rename(columns={panel.columns[0]: "date"})

    date_col = _find_column(panel, DATE_COLUMNS)
    return_col = _find_column(panel, RETURN_COLUMNS)
    dp_col = _find_column(panel, DP_COLUMNS)
    recession_col = _find_optional_column(panel, RECESSION_COLUMNS)

    columns = [date_col, return_col, dp_col]
    if recession_col is not None:
        columns.append(recession_col)

    panel = panel[columns].rename(
        columns={
            date_col: "date",
            return_col: "excess_return",
            dp_col: "log_dp",
        }
    )

    if recession_col is not None:
        panel = panel.rename(columns={recession_col: "recession"})

    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values("date").dropna(subset=["date", "excess_return", "log_dp"])

    return panel


def _sample_window(
    panel: pd.DataFrame, start_date: str | None, end_date: str | None
) -> pd.DataFrame:
    """Filter the panel to a specific sample window."""
    sample = panel.copy()

    if start_date is not None:
        sample = sample[sample["date"] >= pd.Timestamp(start_date)]

    if end_date is not None:
        sample = sample[sample["date"] <= pd.Timestamp(end_date)]

    return sample


def _make_summary_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics for the full sample and Stambaugh subsamples."""
    rows = []

    for sample_name, (start_date, end_date) in STAMBAUGH_WINDOWS.items():
        sample = _sample_window(panel, start_date, end_date)

        rows.append(
            {
                "Sample": sample_name,
                "Start": sample["date"].min().strftime("%Y-%m"),
                "End": sample["date"].max().strftime("%Y-%m"),
                "Obs.": int(sample.shape[0]),
                "Mean excess return (%)": sample["excess_return"].mean() * 100,
                "Std. excess return (%)": sample["excess_return"].std() * 100,
                "Mean log D/P": sample["log_dp"].mean(),
                "Std. log D/P": sample["log_dp"].std(),
                "AR(1) log D/P": sample["log_dp"].autocorr(lag=1),
            }
        )

    table = pd.DataFrame(rows)

    numeric_cols = [
        "Mean excess return (%)",
        "Std. excess return (%)",
        "Mean log D/P",
        "Std. log D/P",
        "AR(1) log D/P",
    ]
    table[numeric_cols] = table[numeric_cols].round(3)

    return table


def _write_latex_table(table: pd.DataFrame, output_path: Path) -> None:
    """Write the summary statistics table as a LaTeX table."""
    caption = (
        "Summary statistics for the monthly excess return and log dividend-price "
        "ratio. The high AR(1) of log D/P highlights the persistence that is central "
        "to the Stambaugh bias."
    )

    latex = table.to_latex(
        index=False,
        escape=True,
        caption=caption,
        label="tab:summary-stats",
        column_format="lrrrrrrrr",
    )

    output_path.write_text(latex, encoding="utf-8")


def _shade_recessions(ax: plt.Axes, panel: pd.DataFrame) -> None:
    """Add NBER recession shading when a recession indicator is available."""
    if "recession" not in panel.columns:
        return

    recessions = panel[["date", "recession"]].copy()
    recessions["recession"] = recessions["recession"].fillna(0).astype(float) > 0
    recessions["block"] = recessions["recession"].ne(recessions["recession"].shift()).cumsum()

    for _, block in recessions[recessions["recession"]].groupby("block"):
        ax.axvspan(block["date"].min(), block["date"].max(), alpha=0.2)


def _write_chart(panel: pd.DataFrame, output_path: Path) -> None:
    """Create a chart of log D/P and its relation to next-month excess returns."""
    plot_data = panel.copy()
    plot_data["next_month_excess_return"] = plot_data["excess_return"].shift(-1)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))

    axes[0].plot(plot_data["date"], plot_data["log_dp"])
    _shade_recessions(axes[0], plot_data)
    axes[0].set_ylabel("Log D/P")
    axes[0].set_xlabel("")
    axes[0].set_title("Log dividend-price ratio over time")

    scatter_data = plot_data.dropna(subset=["log_dp", "next_month_excess_return"])
    axes[1].scatter(
        scatter_data["log_dp"],
        scatter_data["next_month_excess_return"] * 100,
        alpha=0.5,
        s=12,
    )
    axes[1].set_xlabel("Lagged log D/P")
    axes[1].set_ylabel("Next-month excess return (%)")
    axes[1].set_title("Next-month excess return against lagged log D/P")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_figure_latex(output_path: Path) -> None:
    """Write a LaTeX figure wrapper with an explanatory caption."""
    figure_tex = r"""
\begin{figure}[!htbp]
\centering
\includegraphics[width=\textwidth]{../_output/summary_stats_dp.png}
\caption{The log dividend-price ratio is highly persistent over time, and its relationship with next-month excess returns is visually noisy. This motivates the Stambaugh correction: the predictor's persistence matters for inference about return predictability.}
\label{fig:summary-stats-dp}
\end{figure}
""".strip()

    output_path.write_text(figure_tex + "\n", encoding="utf-8")


def main() -> None:
    """Create all summary statistics outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    panel = _prepare_panel()
    table = _make_summary_table(panel)

    _write_latex_table(table, OUTPUT_DIR / "summary_stats.tex")
    _write_chart(panel, OUTPUT_DIR / "summary_stats_dp.png")
    _write_figure_latex(OUTPUT_DIR / "summary_stats_dp_figure.tex")

    print(table.to_string(index=False))
    print("Wrote _output/summary_stats.tex")
    print("Wrote _output/summary_stats_dp.png")
    print("Wrote _output/summary_stats_dp_figure.tex")


if __name__ == "__main__":
    main()