"""
Figure 1: estimates of beta versus rho, by subsample and method.

Each panel plots the OLS estimate and the bias-adjusted estimate of the
predictive slope against the corresponding persistence estimate. The adjusted
point lies below and to the right of OLS in every subsample: correcting the
finite-sample bias lowers the slope and raises the persistence, the joint
movement Stambaugh's eq. (12) predicts. Bayesian posterior means are added to
this figure once Table 2 is built.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from settings import config
from calc_predictor_data import load_tidy_panel
from create_table_01_partC import SUBSAMPLES, fit_subsample
from stambaugh_bias import bias_adjusted_slope

OUTPUT_DIR = Path(config("OUTPUT_DIR"))


def build_figure_01(panel=None, outfile="figure_01.png"):
    """Create the 2x2 beta-vs-rho figure and save it to OUTPUT_DIR."""
    if panel is None:
        panel = load_tidy_panel()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for ax, (name, (start, end)) in zip(axes.flat, SUBSAMPLES.items()):
        c = fit_subsample(panel, start, end)
        s_uv = c["sigma_uv_x1e4"] * 1e-4
        s_vv = c["sigma2_v_x1e4"] * 1e-4

        beta_adj, bias, rho_bc = bias_adjusted_slope(
            c["beta_hat"], s_uv, s_vv, c["rho_hat"], int(c["T"])
        )

        ax.scatter([c["rho_hat"]], [c["beta_hat"]], color="tab:blue", zorder=3)
        ax.annotate("OLS", (c["rho_hat"], c["beta_hat"]),
                    textcoords="offset points", xytext=(6, 4))
        ax.scatter([rho_bc], [beta_adj], color="tab:red", zorder=3)
        ax.annotate("Bias-adjusted", (rho_bc, beta_adj),
                    textcoords="offset points", xytext=(6, -12))

        ax.set_title(name)
        ax.set_xlabel(r"$\rho$")
        ax.set_ylabel(r"$\hat\beta$")

    fig.suptitle(r"Estimates of $\beta$ and $\rho$: OLS vs bias-adjusted")
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / outfile
    fig.savefig(out, dpi=300)
    print(f"Wrote {out}")

    # Console readout for a quick check against the paper's panels.
    return fig


if __name__ == "__main__":
    build_figure_01()