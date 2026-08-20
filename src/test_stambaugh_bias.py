"""Tests for the Stambaugh-bias mechanics.

These tests cover closed-form bias logic separately from Monte Carlo simulation
logic so failures point to the correct part of the implementation.
"""

import numpy as np

from monte_carlo import analytical_slope_bias
from stambaugh_bias import bias_adjusted_slope, bias_correct_rho


# Closed-form rho check: Kendall correction should move persistent rho estimates upward.
def test_bias_correct_rho_moves_persistence_upward_for_persistent_predictor():
    rho_hat = 0.95
    T = 120

    rho_corrected = bias_correct_rho(rho_hat, T)

    assert rho_corrected > rho_hat
    assert rho_corrected < 1.0


# Closed-form bias check: negative return/predictor shock covariance should create upward slope bias.
def test_negative_shock_covariance_creates_positive_slope_bias():
    s_uv = -0.20
    s_vv = 1.00
    rho = 0.95
    T = 120

    bias = analytical_slope_bias(s_uv=s_uv, s_vv=s_vv, rho=rho, T=T)

    assert bias > 0


# Adjustment check: bias_adjusted_slope should subtract the upward finite-sample bias.
def test_bias_adjusted_slope_subtracts_upward_bias():
    beta_hat = 0.08
    s_uv = -0.20
    s_vv = 1.00
    rho_hat = 0.95
    T = 120

    beta_adjusted, bias, rho_used = bias_adjusted_slope(
        beta_hat=beta_hat,
        s_uv=s_uv,
        s_vv=s_vv,
        rho_hat=rho_hat,
        T=T,
    )

    assert bias > 0
    assert rho_used > rho_hat
    assert np.isclose(beta_adjusted, beta_hat - bias)
    assert beta_adjusted < beta_hat