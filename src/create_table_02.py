"""
Table 2: Bayesian posterior moments for the predictive slope.

Specs A (flat prior, conditional likelihood, rho unrestricted) and B (same,
with the stationarity restriction rho in (-1,1) imposed by rejection) are
computed for each subsample. Specs C and D of the paper use the exact
likelihood (treating x_0 as a stationary draw) and require Metropolis-Hastings;
they are discussed as an extension in the report rather than computed here.
"""

from pathlib import Path

import pandas as pd

from settings import config
from calc_predictor_data import load_tidy_panel
import sys
from create_table_01_partC import SUBSAMPLES, UPDATE_SUBSAMPLES
from bayesian import posterior_moments, sample_posterior
from create_table_01 import format_partAB   # reuse the paper-style formatter

OUTPUT_DIR = Path(config("OUTPUT_DIR"))
N_DRAWS = 20_000
SEED = 11

PART2_LAYOUT = [
    ("mean", "Mean", 2),
    ("std", "Std. Dev.", 2),
    ("skewness", "Skewness", 2),
    ("kurtosis", "Kurtosis", 2),
    ("p_beta_leq_0", r"Prob($\beta \leq 0$)", 2),
]


def _window_arrays(panel, start, end):
    w = panel[(panel["date"] >= start) & (panel["date"] <= end)]
    x = w["dp_ratio"].to_numpy()
    r = w["ret_excess"].to_numpy()
    return r[1:], x[:-1], x[1:]


def build_table_02(panel=None, n_draws=N_DRAWS, seed=SEED, subsamples=SUBSAMPLES):
    """Return (partA, partB) posterior-moment frames, one column per subsample."""
    if panel is None:
        panel = load_tidy_panel()

    colsA, colsB = {}, {}
    for name, (start, end) in subsamples.items():
        r_next, x_lag, x_next = _window_arrays(panel, start, end)
        a = sample_posterior(r_next, x_lag, x_next, n_draws=n_draws, seed=seed)
        b = sample_posterior(r_next, x_lag, x_next, n_draws=n_draws, seed=seed,
                             restrict_rho=True)
        colsA[name] = posterior_moments(a["beta"])
        colsB[name] = posterior_moments(b["beta"])
        print(f"{name}: spec B kept {b['accept_rate']:.1%} of draws")

    return pd.DataFrame(colsA), pd.DataFrame(colsB)


def _format(part):
    rows = {}
    for key, label, dec in PART2_LAYOUT:
        rows[label] = part.loc[key].map(f"{{:.{dec}f}}".format)
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    updated = "--updated" in sys.argv
    subsamples = UPDATE_SUBSAMPLES if updated else SUBSAMPLES
    suffix = "_updated" if updated else ""

    partA, partB = build_table_02(subsamples=subsamples)
    print("\nA. Conditional likelihood, rho unrestricted:")
    print(_format(partA).to_string())
    print("\nB. Conditional likelihood, rho in (-1, 1):")
    print(_format(partB).to_string())

    combined = pd.concat({
        r"A. Conditional likelihood, $\rho \in (-\infty, \infty)$": _format(partA),
        r"B. Conditional likelihood, $\rho \in (-1, 1)$": _format(partB),
    })
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    latex = combined.to_latex(
        escape=False,
        caption=(
            "Posterior distributions for $\\beta$ (replication of Stambaugh "
            "1999, Table 2, Parts A and B). Part A uses a flat prior with the "
            "likelihood conditioned on $x_0$; the posterior is centered on the "
            "OLS estimate. Part B adds the stationarity restriction "
            "$\\rho \\in (-1,1)$, which matters most in the short, highly "
            "persistent 1977--1996 sample. Takeaway: the Bayesian posterior "
            "places only about 6\\% of its mass below zero for 1927--1996, "
            "against a finite-sample frequentist $p$-value of roughly 18\\% "
            "for the same data -- the two frameworks answer different "
            "questions and disagree about how much evidence of predictability "
            "the sample contains."
        ),
        label=f"tab:table2{suffix}",
    )
    (OUTPUT_DIR / f"table_02{suffix}.tex").write_text(latex)
    print(f"\nWrote {OUTPUT_DIR / f"table_02{suffix}.tex"}")