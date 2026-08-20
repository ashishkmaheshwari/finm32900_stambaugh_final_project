"""Create summary statistics table and underlying-data chart for Stambaugh data."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from calc_predictor_data import load_tidy_panel
from settings import config

OUTPUT_DIR = Path(config("OUTPUT_DIR"))

STAMBAUGH_WINDOWS = {
    "Full sample": (None, None),
    "1927-1996": ("1927-01-01", "1996-12-31"),
    "1927-1951": ("1927-01-01", "1951-12-31"),
    "1952-1996": ("1952-01-01", "1996-12-31"),
    "1977-1996": ("1977-01-01", "1996-12-31"),
}


def _prepare_panel() -> pd.DataFrame:
    """Load the project tidy panel with fixed column names from build_tidy_panel."""
    panel = load_tidy_panel().copy()

    required_cols = {"date", "ret_excess", "log_dp", "dp_ratio"}
    missing = required_cols.difference(panel.columns)
    if missing:
        raise ValueError(f"Missing required panel columns: {sorted(missing)}")

    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values("date")
    panel = panel.dropna(subset=["date", "ret_excess", "log_dp", "dp_ratio"])

    return panel


def _sample_window(
    panel: pd.DataFrame, start_date: str | None, end_date: str | None
) -> pd.DataFrame:
    """Filter the panel to a sample window."""
    sample = panel.copy()

    if start_date is not None:
        sample = sample[sample["date"] >= pd.Timestamp(start_date)]

    if end_date is not None:
        sample = sample[sample["date"] <= pd.Timestamp(end_date)]

    return sample


def _make_summary_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics for returns, log D/P, and D/P level."""
    rows = []

    for sample_name, (start_date, end_date) in STAMBAUGH_WINDOWS.items():
        sample = _sample_window(panel, start_date, end_date)

        rows.append(
            {
                "Sample": sample_name,
                "Start": sample["date"].min().strftime("%Y-%m"),
                "End": sample["date"].max().strftime("%Y-%m"),
                "Obs.": int(sample.shape[0]),
                "Mean excess return (%)": sample["ret_excess"].mean() * 100,
                "Std. excess return (%)": sample["ret_excess"].std() * 100,
                "Mean D/P": sample["dp_ratio"].mean(),
                "Std. D/P": sample["dp_ratio"].std(),
                "Mean log D/P": sample["log_dp"].mean(),
                "Std. log D/P": sample["log_dp"].std(),
                "AR(1) D/P": sample["dp_ratio"].autocorr(lag=1),
                "AR(1) log D/P": sample["log_dp"].autocorr(lag=1),
            }
        )

    table = pd.DataFrame(rows)

    numeric_cols = [
        "Mean excess return (%)",
        "Std. excess return (%)",
        "Mean D/P",
        "Std. D/P",
        "Mean log D/P",
        "Std. log D/P",
        "AR(1) D/P",
        "AR(1) log D/P",
    ]
    table[numeric_cols] = table[numeric_cols].round(3)

    return table


def _write_latex_table(table: pd.DataFrame, output_path: Path) -> None:
    """Write the summary statistics table as LaTeX."""
    caption = (
        "Summary statistics for the monthly excess return, the dividend-price "
        "ratio in levels, and the log dividend-price ratio. The AR(1) column "
        "shows that log D/P is highly persistent, which is central to the "
        "Stambaugh finite-sample bias."
    )

    n_cols = table.shape[1]
    col_format = "lcc" + "r" * (n_cols - 3)
    latex = table.to_latex(
        index=False,
        escape=True,
        caption=caption,
        label="tab:summary-stats",
        column_format=col_format,
        float_format="%.3f",
    )

    # Wrap the tabular in a resizebox so the eleven columns fit the text width.
    latex = latex.replace(
        r"\begin{tabular}", r"\resizebox{\textwidth}{!}{%" + "\n" + r"\begin{tabular}"
    ).replace(
        r"\end{tabular}", r"\end{tabular}}"
    )
    output_path.write_text(latex, encoding="utf-8")


def _write_chart(panel: pd.DataFrame, output_path: Path) -> None:
    """Create a chart of log D/P and next-month returns against lagged log D/P."""
    plot_data = panel.copy()
    plot_data["log_dp_lag"] = plot_data["log_dp"].shift(1)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))

    axes[0].plot(plot_data["date"], plot_data["log_dp"])
    axes[0].set_ylabel("Log D/P")
    axes[0].set_xlabel("")
    axes[0].set_title("Log dividend-price ratio over time")

    scatter_data = plot_data.dropna(subset=["log_dp_lag", "ret_excess"])
    axes[1].scatter(
        scatter_data["log_dp_lag"],
        scatter_data["ret_excess"] * 100,
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
    """Write a LaTeX figure wrapper with a caption."""
    figure_tex = r"""
\begin{figure}[!htbp]
\centering
\includegraphics[width=\textwidth]{../_output/summary_stats_dp.png}
\caption{The log dividend-price ratio is highly persistent, while the scatter of next-month excess returns against lagged log D/P is noisy. This is the core empirical setting in which Stambaugh's finite-sample bias matters: the predictor moves slowly, but the return relationship is difficult to see cleanly in monthly data.}
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