"""
Figure 1: estimates of beta versus rho, by subsample and method.

Each panel plots the predictive slope against the persistence estimate for one
subsample, under every method the project computes:

    OLS      -- the raw least-squares estimate. The paper does not letter this
                point; we include it as the reference the corrections move
                away from.
    A-D      -- posterior means of the four Bayesian specifications of Table 2.
                Where two specifications coincide to within plotting precision
                they are drawn once and labelled jointly (e.g. "A & B"), as the
                paper does in its 1927-51 panel.
    F        -- the bias-adjusted OLS estimate, using the Kendall-corrected rho.

The paper's point E is a further Bayesian variant discussed in its text, which
this project does not implement.

The signature geometry: every correction lies below and to the right of OLS.
Correcting the finite-sample bias lowers the slope and raises the persistence
together, because both are consequences of the same downward bias in rho_hat.

Reliability note: specifications C and D are obtained by importance-reweighting
the specification B draws. Where the effective sample size of those weights is
low -- notably specification D in the post-war samples -- the posterior mean is
sensitive to weight concentration even though tail probabilities remain stable.
Such points are drawn hollow and flagged in the caption rather than presented
as equally precise.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from settings import config
from calc_predictor_data import load_tidy_panel
from create_table_01_partC import SUBSAMPLES, fit_subsample
from create_table_02 import build_table_02
from stambaugh_bias import bias_adjusted_slope

OUTPUT_DIR = Path(config("OUTPUT_DIR"))

BAYES_SPECS = ("A", "B", "C", "D")

# Two Bayesian points are treated as coincident (and labelled jointly) when
# they differ by less than this fraction of the panel's plotted range.
COINCIDENCE_TOL = 0.02

# Effective sample size below this fraction of nominal draws marks a
# reweighted estimate as low-precision.
LOW_ESS_THRESHOLD = 200


def _group_coincident(points, tol_rho, tol_beta):
    """Group (label, rho, beta) points that coincide to within tolerance.

    Returns a list of (joined_label, rho, beta), where the coordinates are the
    mean of the grouped points. Preserves specification order in the label.
    """
    groups = []
    for label, rho, beta in points:
        for g in groups:
            if abs(rho - g["rho"]) < tol_rho and abs(beta - g["beta"]) < tol_beta:
                g["labels"].append(label)
                g["rhos"].append(rho)
                g["betas"].append(beta)
                g["rho"] = float(np.mean(g["rhos"]))
                g["beta"] = float(np.mean(g["betas"]))
                break
        else:
            groups.append({"labels": [label], "rhos": [rho], "betas": [beta],
                           "rho": rho, "beta": beta})
    return [(" & ".join(g["labels"]), g["rho"], g["beta"]) for g in groups]


def build_figure_01(panel=None, outfile="figure_01.png", subsamples=SUBSAMPLES):
    """Create the 2x2 beta-vs-rho figure and save it to OUTPUT_DIR."""
    if panel is None:
        panel = load_tidy_panel()

    moments, rho_means, ess = build_table_02(
        panel=panel, subsamples=subsamples, return_rho=True, return_ess=True
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    for ax, (name, (start, end)) in zip(axes.flat, subsamples.items()):
        c = fit_subsample(panel, start, end)
        s_uv = c["sigma_uv_x1e4"] * 1e-4
        s_vv = c["sigma2_v_x1e4"] * 1e-4

        beta_adj, _bias, rho_bc = bias_adjusted_slope(
            c["beta_hat"], s_uv, s_vv, c["rho_hat"], int(c["T"])
        )

        # Collect every point so the panel range can be computed before drawing.
        bayes_pts = [
            (spec, float(rho_means.loc[spec, name]),
             float(moments[spec].loc["mean", name]))
            for spec in BAYES_SPECS
        ]
        all_rho = [c["rho_hat"], rho_bc] + [p[1] for p in bayes_pts]
        all_beta = [c["beta_hat"], beta_adj] + [p[2] for p in bayes_pts]
        tol_rho = COINCIDENCE_TOL * (max(all_rho) - min(all_rho) or 1.0)
        tol_beta = COINCIDENCE_TOL * (max(all_beta) - min(all_beta) or 1.0)

        # Raw OLS.
        ax.scatter([c["rho_hat"]], [c["beta_hat"]], color="tab:blue", s=45,
                   zorder=4)
        ax.annotate("OLS", (c["rho_hat"], c["beta_hat"]),
                    textcoords="offset points", xytext=(7, -14), fontsize=9,
                    color="tab:blue")

        # Bayesian specifications, with coincident points merged.
        for label, rho_s, beta_s in _group_coincident(bayes_pts, tol_rho, tol_beta):
            specs = [s.strip() for s in label.split("&")]
            # Draw hollow if any contributing spec has a low effective sample.
            unreliable = any(
                ess.loc[s, name] < LOW_ESS_THRESHOLD for s in specs if s in ess.index
            )
            ax.scatter(
                [rho_s], [beta_s], s=26, zorder=3,
                facecolors="none" if unreliable else "black",
                edgecolors="black",
            )
            ax.annotate(label + ("*" if unreliable else ""), (rho_s, beta_s),
                        textcoords="offset points", xytext=(6, 4), fontsize=9)

        # Bias-adjusted OLS (the paper's point F).
        ax.scatter([rho_bc], [beta_adj], color="tab:red", s=45, zorder=4)
        ax.annotate("F", (rho_bc, beta_adj), textcoords="offset points",
                    xytext=(7, -13), fontsize=9, color="tab:red")

        ax.set_title(name)
        ax.set_xlabel(r"$\rho$")
        ax.set_ylabel(r"$\beta$")
        ax.margins(0.20)

    fig.suptitle(r"Estimates of $\beta$ and $\rho$ by method and subperiod")
    fig.text(0.5, 0.005,
             "Specifications A and B are sampled directly from the conjugate "
             "posterior; C and D by Metropolis-Hastings, all chains reaching "
             "effective sample sizes above 200.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=[0, 0.02, 1, 1])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / outfile
    fig.savefig(out, dpi=300)
    print(f"Wrote {out}")
    return fig


if __name__ == "__main__":
    build_figure_01()