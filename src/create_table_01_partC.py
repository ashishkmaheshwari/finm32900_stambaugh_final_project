"""
Table 1, Part C: sample characteristics and parameter values per subsample.

For each of Stambaugh's four sample periods this estimates, on real data:
the OLS predictive slope beta_hat (return in month t+1 on the dividend-price
ratio at end of month t), the sample size T, the predictor's AR(1) persistence
rho_hat, and the residual covariance matrix of the two regressions
(sigma2_u, sigma2_v, sigma_uv, reported x 10^4 as in the paper).

These estimated (rho, T, Sigma) values are the inputs the Monte Carlo engine
(Table 1, Part A) simulates from. Paper targets for comparison:
beta: 0.21/0.21/0.44/0.19, rho: 0.972/0.948/0.980/0.987, T: 840/300/540/240.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from settings import config
from calc_predictor_data import load_tidy_panel

OUTPUT_DIR = Path(config("OUTPUT_DIR"))

SUBSAMPLES = {
    "1927-1996": ("1927-01-01", "1996-12-31"),
    "1927-1951": ("1927-01-01", "1951-12-31"),
    "1952-1996": ("1952-01-01", "1996-12-31"),
    "1977-1996": ("1977-01-01", "1996-12-31"),
}


def ols(y, X):
    """OLS with intercept; returns (coefs, residuals)."""
    X1 = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return coef, y - X1 @ coef


def fit_subsample(panel, start, end, predictor="dp_ratio"):
    """Estimate beta, T, rho, and the residual covariance on one window.

    The predictive regression pairs r_{t+1} with x_t, so x is lagged by one
    month relative to the return. The AR(1) is x_{t+1} on x_t over the same
    window.
    """
    p = panel[(panel["date"] >= start) & (panel["date"] <= end)].reset_index(drop=True)
    x = p[predictor].to_numpy()
    r = p["ret_excess"].to_numpy()

    x_lag, r_next = x[:-1], r[1:]          # r_{t+1} on x_t
    coef_b, u = ols(r_next, x_lag)         # predictive regression -> u
    coef_r, v = ols(x[1:], x_lag)          # AR(1) -> v

    T = len(r_next)
    s_uu = u @ u / (T - 2)
    s_vv = v @ v / (T - 2)
    s_uv = u @ v / (T - 2)

    return {
        "beta_hat": coef_b[1],
        "T": T,
        "rho_hat": coef_r[1],
        "sigma2_u_x1e4": s_uu * 1e4,
        "sigma2_v_x1e4": s_vv * 1e4,
        "sigma_uv_x1e4": s_uv * 1e4,
        "corr_uv": s_uv / np.sqrt(s_uu * s_vv),
    }


def build_partC(panel=None):
    """Assemble Part C as a DataFrame, one column per subsample (paper layout)."""
    if panel is None:
        panel = load_tidy_panel()
    rows = {name: fit_subsample(panel, s, e) for name, (s, e) in SUBSAMPLES.items()}
    return pd.DataFrame(rows)


if __name__ == "__main__":
    table = build_partC()
    print(table.round(4).to_string())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    latex = table.round(4).to_latex(
        caption=(
            "Sample characteristics and parameter values (Table 1, Part C). "
            "In every subsample the dividend-price ratio is highly persistent "
            "($\\hat\\rho$ near one) and its innovations are strongly negatively "
            "correlated with return innovations ($\\sigma_{uv}<0$): exactly the "
            "configuration in which the OLS predictive slope is biased upward."
        ),
        label="tab:table1_partC",
    )
    (OUTPUT_DIR / "table_01_partC.tex").write_text(latex)
    print(f"\nWrote {OUTPUT_DIR / 'table_01_partC.tex'}")