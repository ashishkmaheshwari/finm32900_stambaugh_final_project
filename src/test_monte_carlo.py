"""
Unit tests for the Monte Carlo engine.

Each test checks one falsifiable prediction of the Stambaugh bias mechanism,
so a failure localizes a specific broken assumption:
  - OLS building block recovers a known slope (no statistics involved);
  - the bias EXISTS under negative innovation correlation with beta = 0;
  - the bias VANISHES when the correlation is zero (the control);
  - the bias SHRINKS like 1/T (it is a finite-sample phenomenon);
  - simulation and the analytical approximation agree to first order;
  - the finite-sample p-value exceeds the naive one for a positive slope.
"""

import numpy as np
import pytest

from monte_carlo import (
    analytical_slope_bias,
    ols_slope,
    simulate_slopes,
    summarize_distribution,
    true_pvalue,
)

RHO, T = 0.97, 240
SIGMA_NEG = [[1.0, -0.9], [-0.9, 1.0]]   # the dividend-yield configuration
SIGMA_ZERO = [[1.0, 0.0], [0.0, 1.0]]    # the control


def _sim(Sigma, T=T, rho=RHO, n_sims=8000, seed=1):
    return simulate_slopes(
        alpha=0.0, beta=0.0, theta=0.0, rho=rho, Sigma=Sigma,
        T=T, x0=0.0, n_sims=n_sims, seed=seed,
    )


def test_ols_recovers_known_slope():
    x = np.linspace(0.0, 1.0, 50)
    y = 3.0 + 2.5 * x
    assert ols_slope(y, x) == pytest.approx(2.5, abs=1e-8)


def test_bias_positive_under_negative_correlation():
    beta_hats, _ = _sim(SIGMA_NEG)
    stats = summarize_distribution(beta_hats)
    # Positive and many Monte Carlo standard errors from zero.
    mc_se = stats["std"] / np.sqrt(len(beta_hats))
    assert stats["bias"] > 10 * mc_se


def test_bias_vanishes_without_correlation():
    beta_hats, _ = _sim(SIGMA_ZERO)
    stats = summarize_distribution(beta_hats)
    mc_se = stats["std"] / np.sqrt(len(beta_hats))
    assert abs(stats["bias"]) < 4 * mc_se


def test_bias_shrinks_like_one_over_T():
    small, _ = _sim(SIGMA_NEG, T=120, seed=2)
    large, _ = _sim(SIGMA_NEG, T=480, seed=3)
    b_small = summarize_distribution(small)["bias"]
    b_large = summarize_distribution(large)["bias"]
    # Quadruple the sample -> roughly a quarter of the bias.
    assert b_large < b_small
    assert b_large == pytest.approx(b_small / 4.0, rel=0.5)


def test_simulation_matches_analytical_bias():
    beta_hats, _ = _sim(SIGMA_NEG, n_sims=20000, seed=4)
    simulated = summarize_distribution(beta_hats)["bias"]
    analytic = analytical_slope_bias(s_uv=-0.9, s_vv=1.0, rho=RHO, T=T)
    assert np.sign(simulated) == np.sign(analytic)
    assert simulated == pytest.approx(analytic, rel=0.25)


def test_true_pvalue_exceeds_naive():
    beta_hats, _ = _sim(SIGMA_NEG, n_sims=20000, seed=5)
    # A slope at the naive one-sided 5% critical value of a mean-zero normal.
    naive_5pct_slope = 1.645 * beta_hats.std(ddof=1)
    assert true_pvalue(naive_5pct_slope, beta_hats) > 0.05