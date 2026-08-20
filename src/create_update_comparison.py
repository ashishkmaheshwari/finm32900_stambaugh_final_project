"""Create a paper-era versus updated-era comparison exhibit.

The table contrasts the original Stambaugh paper-era sample ending in 1996
with the updated full sample through the latest available data. It reports the
OLS slope, the analytical Stambaugh bias adjustment, predictor persistence,
the naive p-value, and the finite-sample Monte Carlo "true" p-value.
"""

from pathlib import Path

import pandas as pd

from calc_predictor_data import load_tidy_panel
from create_table_01_partC import fit_subsample
from monte_carlo import analytical_slope_bias, simulate_slopes, true_pvalue
from settings import config

OUTPUT_DIR = Path(config("OUTPUT_DIR"))

N_SIMS = 20_000
SEED = 42


def _sample_window(panel: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """Return the panel restricted to a start and end date."""
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])

    return panel[
        (panel["date"] >= pd.Timestamp(start_date))
        & (panel["date"] <= pd.Timestamp(end_date))
    ].copy()


def _monte_carlo_true_pvalue(
    panel: pd.DataFrame,
    start_date: str,
    end_date: str,
    estimates: pd.Series,
    n_sims: int = N_SIMS,
    seed: int = SEED,
) -> float:
    """Compute the finite-sample Monte Carlo p-value using shared simulation code."""
    window = _sample_window(panel, start_date, end_date)

    x0 = float(window["dp_ratio"].iloc[0])
    rho = float(estimates["rho_hat"])
    t_obs = int(estimates["T"])

    sigma = [
        [
            float(estimates["sigma2_u_x1e4"]) * 1e-4,
            float(estimates["sigma_uv_x1e4"]) * 1e-4,
        ],
        [
            float(estimates["sigma_uv_x1e4"]) * 1e-4,
            float(estimates["sigma2_v_x1e4"]) * 1e-4,
        ],
    ]

    beta_hats, _ = simulate_slopes(
        alpha=0.0,
        beta=0.0,
        theta=(1.0 - rho) * x0,
        rho=rho,
        Sigma=sigma,
        T=t_obs,
        x0=x0,
        n_sims=n_sims,
        seed=seed,
    )

    return true_pvalue(float(estimates["beta_hat"]), beta_hats)


def _comparison_row(
    panel: pd.DataFrame,
    label: str,
    start_date: str,
    end_date: str,
    n_sims: int = N_SIMS,
    seed: int = SEED,
) -> dict[str, float | int | str]:
    """Estimate one sample window and return formatted comparison statistics."""
    estimates = fit_subsample(panel, start_date, end_date)

    slope_bias = analytical_slope_bias(
        s_uv=float(estimates["sigma_uv_x1e4"]) * 1e-4,
        s_vv=float(estimates["sigma2_v_x1e4"]) * 1e-4,
        rho=float(estimates["rho_hat"]),
        T=int(estimates["T"]),
    )

    mc_pvalue = _monte_carlo_true_pvalue(
        panel=panel,
        start_date=start_date,
        end_date=end_date,
        estimates=estimates,
        n_sims=n_sims,
        seed=seed,
    )

    return {
        "Statistic": label,
        "Start": pd.Timestamp(start_date).strftime("%Y-%m"),
        "End": pd.Timestamp(end_date).strftime("%Y-%m"),
        "Obs.": int(estimates["T"]),
        r"OLS slope $\hat{\beta}$": float(estimates["beta_hat"]),
        "Bias adjustment": slope_bias,
        "Bias-adjusted slope": float(estimates["beta_hat"]) - slope_bias,
        r"Predictor persistence $\rho$": float(estimates["rho_hat"]),
        "Naive p-value": float(estimates["p_naive"]),
        "Monte Carlo true p-value": mc_pvalue,
    }


def build_update_comparison(
    panel: pd.DataFrame | None = None,
    n_sims: int = N_SIMS,
    seed: int = SEED,
) -> pd.DataFrame:
    """Build the paper-era versus updated-era comparison table."""
    if panel is None:
        panel = load_tidy_panel()

    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])

    start_date = panel["date"].min().strftime("%Y-%m-%d")
    updated_end = panel["date"].max().strftime("%Y-%m-%d")

    rows = [
        _comparison_row(
            panel=panel,
            label="Paper era",
            start_date=start_date,
            end_date="1996-12-31",
            n_sims=n_sims,
            seed=seed,
        ),
        _comparison_row(
            panel=panel,
            label="Updated full sample",
            start_date=start_date,
            end_date=updated_end,
            n_sims=n_sims,
            seed=seed,
        ),
    ]

    table = pd.DataFrame(rows).set_index("Statistic").T.reset_index()
    table = table.rename(columns={"index": "Statistic"})

    numeric_rows = {
        r"OLS slope $\hat{\beta}$": "{:.4f}",
        "Bias adjustment": "{:.4f}",
        "Bias-adjusted slope": "{:.4f}",
        r"Predictor persistence $\rho$": "{:.3f}",
        "Naive p-value": "{:.3f}",
        "Monte Carlo true p-value": "{:.3f}",
    }

    for row_name, formatter in numeric_rows.items():
        mask = table["Statistic"] == row_name
        for col in ["Paper era", "Updated full sample"]:
            table.loc[mask, col] = table.loc[mask, col].astype(float).map(
                formatter.format
            )

    return table


def write_update_comparison(table: pd.DataFrame, output_path: Path) -> None:
    """Write the update comparison as a LaTeX table."""
    caption = (
        "Paper-era versus updated-era comparison of the predictive regression. "
        "The updated sample extends the original 1927--1996 paper-era window "
        "through the latest available data, covering the dot-com period, the "
        "2008 financial crisis, and the modern buyback-heavy period. The main "
        "takeaway is that the predictor remains highly persistent, so the "
        "finite-sample inference issue does not disappear; comparing the naive "
        "and Monte Carlo p-values shows how much of the apparent predictability "
        "survives the Stambaugh correction."
    )

    latex = table.to_latex(
        index=False,
        escape=False,
        caption=caption,
        label="tab:update-comparison",
    )

    output_path.write_text(latex, encoding="utf-8")


def main() -> None:
    """Create the update comparison exhibit."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    table = build_update_comparison()
    output_path = OUTPUT_DIR / "update_comparison.tex"
    write_update_comparison(table, output_path)

    print(table.to_string(index=False))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()