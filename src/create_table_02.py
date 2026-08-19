"""
Table 2: Bayesian posterior moments for the predictive slope.

Four specifications, following the paper:

    A: conditional likelihood (x_0 fixed), flat prior, rho unrestricted.
    B: conditional likelihood, rho restricted to (-1, 1) by rejection.
    C: exact likelihood (x_0 drawn from the stationary distribution).
    D: exact likelihood under the paper's alternative prior
       p(b, Sigma) ~ (1 - rho^2)^{-1} sigma_v^2 |Sigma|^{-5/2}.

Specifications A and B are conjugate and sampled directly (bayesian.py).
Specifications C and D are not conjugate -- the stationary density of x_0 is
nonlinear in rho -- and are sampled by Metropolis-Hastings (mcmc.py). An earlier
attempt to reach C and D by importance-reweighting the conditional draws is
retained in bayesian.py and documented there: it succeeds in the long samples
but fails in the post-war samples, where specification D's (1-rho^2)^{-1} prior
factor concentrates the weights on draws pinned against rho = 1.

Posterior means of rho are also available (return_rho=True) so that Figure 1
can plot each specification in (rho, beta) space alongside the OLS and
bias-adjusted estimates.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from settings import config
from calc_predictor_data import load_tidy_panel
from create_table_01_partC import SUBSAMPLES, UPDATE_SUBSAMPLES
from bayesian import posterior_moments, sample_posterior
from mcmc import posterior_moments_from_chain, run_mh

OUTPUT_DIR = Path(config("OUTPUT_DIR"))

N_DRAWS = 20_000          # conjugate draws for specs A and B
N_DRAWS_MH = 200_000      # Metropolis-Hastings iterations for specs C and D
SEED = 11

PART2_LAYOUT = [
    ("mean", "Mean", 2),
    ("std", "Std. Dev.", 2),
    ("skewness", "Skewness", 2),
    ("kurtosis", "Kurtosis", 2),
    ("p_beta_leq_0", r"Prob($\beta \leq 0$)", 2),
]

SPEC_TITLES = {
    "A": r"A. Conditional likelihood; $\rho \in (-\infty, \infty)$",
    "B": r"B. Conditional likelihood; $\rho \in (-1, 1)$",
    "C": r"C. Exact likelihood; $\rho \in (-1, 1)$",
    "D": r"D. Exact likelihood, alternative prior; $\rho \in (-1, 1)$",
}


def _window_arrays(panel, start, end):
    """Return (r_next, x_lag, x_next) for one date window."""
    w = panel[(panel["date"] >= start) & (panel["date"] <= end)]
    x = w["dp_ratio"].to_numpy()
    r = w["ret_excess"].to_numpy()
    return r[1:], x[:-1], x[1:]


def build_table_02(panel=None, n_draws=N_DRAWS, n_draws_mh=N_DRAWS_MH,
                   seed=SEED, subsamples=SUBSAMPLES,
                   return_rho=False, return_ess=False):
    """Compute posterior moments for specs A-D on each subsample.

    Returns a dict of DataFrames keyed by specification. If return_rho is True,
    a second frame of posterior mean rho values is returned; if return_ess is
    also True, a third frame gives the effective sample size behind each
    estimate (the number of conjugate draws for A and B, the chain's effective
    sample size for C and D), indexed by specification and columned by
    subsample.
    """
    if panel is None:
        panel = load_tidy_panel()

    cols = {"A": {}, "B": {}, "C": {}, "D": {}}
    rho_means = {}
    ess = {}

    for name, (start, end) in subsamples.items():
        r_next, x_lag, x_next = _window_arrays(panel, start, end)

        # Specs A and B: conjugate, sampled directly.
        a = sample_posterior(r_next, x_lag, x_next, n_draws=n_draws, seed=seed)
        b = sample_posterior(r_next, x_lag, x_next, n_draws=n_draws, seed=seed,
                             restrict_rho=True)
        cols["A"][name] = posterior_moments(a["beta"])
        cols["B"][name] = posterior_moments(b["beta"])

        spec_rho = {"A": float(a["rho"].mean()), "B": float(b["rho"].mean())}
        spec_ess = {"A": float(len(a["beta"])), "B": float(len(b["beta"]))}
        print(f"{name}: spec B kept {b['accept_rate']:.1%} of draws")

        # Specs C and D: exact likelihood, sampled by Metropolis-Hastings.
        for spec in ("C", "D"):
            out = run_mh(r_next, x_lag, x_next, spec=spec, exact=True,
                         n_draws=n_draws_mh, thin=1, seed=seed)
            cols[spec][name] = posterior_moments_from_chain(out["beta"])
            spec_rho[spec] = float(out["rho"].mean())
            spec_ess[spec] = float(out["n_eff_beta"])
            print(f"{name} spec {spec}: MH acceptance {out['accept_rate']:.1%}, "
                  f"ESS = {out['n_eff_beta']:,.0f}")

        rho_means[name] = spec_rho
        ess[name] = spec_ess

    moments = {k: pd.DataFrame(v) for k, v in cols.items()}
    if return_rho and return_ess:
        return moments, pd.DataFrame(rho_means), pd.DataFrame(ess)
    if return_rho:
        return moments, pd.DataFrame(rho_means)
    return moments


def _format(part):
    """Lay out one specification's moments with the paper's row labels."""
    rows = {}
    for key, label, dec in PART2_LAYOUT:
        rows[label] = part.loc[key].map(f"{{:.{dec}f}}".format)
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    updated = "--updated" in sys.argv
    subsamples = UPDATE_SUBSAMPLES if updated else SUBSAMPLES
    suffix = "_updated" if updated else ""

    parts = build_table_02(subsamples=subsamples)

    for spec in ("A", "B", "C", "D"):
        print(f"\n{SPEC_TITLES[spec]}")
        print(_format(parts[spec]).to_string())

    combined = pd.concat({SPEC_TITLES[s]: _format(parts[s])
                          for s in ("A", "B", "C", "D")})

    caption_extra = (
        " Samples are extended through the most recently available data."
        if updated else ""
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    latex = combined.to_latex(
        escape=False,
        caption=(
            "Posterior distributions for $\\beta$ (replication of Stambaugh "
            "1999, Table 2). Parts A and B condition the likelihood on $x_0$; "
            "with a flat prior the posterior is conjugate and centers on the "
            "OLS estimate, and Part B adds the stationarity restriction "
            "$\\rho \\in (-1,1)$, which binds only in the short, highly "
            "persistent 1977--1996 sample. Parts C and D use the exact "
            "likelihood, treating $x_0$ as a draw from the predictor's "
            "stationary distribution; that term is nonlinear in $\\rho$, so "
            "these posteriors are not conjugate and are sampled by "
            "Metropolis-Hastings. Part D applies the paper's alternative "
            "prior. Takeaway: the exact likelihood strengthens the evidence "
            "for predictability while Part D's prior pulls back toward "
            "$\\rho$ near one, and across all four specifications the "
            "posterior places far less mass below zero than the "
            "finite-sample frequentist $p$-values of "
            "Table~\\ref{tab:table1} would suggest." + caption_extra
        ),
        label=f"tab:table2{suffix}",
    )
    (OUTPUT_DIR / f"table_02{suffix}.tex").write_text(latex)
    print(f"\nWrote {OUTPUT_DIR / f'table_02{suffix}.tex'}")