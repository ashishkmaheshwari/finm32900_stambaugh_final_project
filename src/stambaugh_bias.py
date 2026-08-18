"""
Bias-corrected estimators for the predictive regression (Figure 1).

The AR(1) estimate of the predictor's persistence is biased DOWN by
approximately (1 + 3 rho)/T (Kendall 1954); through the negative innovation
covariance, the predictive slope inherits an UPWARD bias of
-(sigma_uv/sigma_vv)(1 + 3 rho)/T (Stambaugh 1999, eq. 12). This module
inverts the Kendall relation to bias-correct rho, then subtracts the implied
slope bias. On a beta-vs-rho plot, the corrected estimate therefore moves
DOWN (smaller slope) and RIGHT (higher persistence) relative to raw OLS --
the signature geometry of the paper's Figure 1.
"""

from monte_carlo import analytical_slope_bias


def bias_correct_rho(rho_hat, T):
    """Invert E[rho_hat] = rho - (1 + 3 rho)/T to recover rho from rho_hat."""
    return (rho_hat + 1.0 / T) / (1.0 - 3.0 / T)


def bias_adjusted_slope(beta_hat, s_uv, s_vv, rho_hat, T, use_corrected_rho=True):
    """Return (beta_adjusted, bias, rho_used).

    The bias is evaluated at the bias-corrected rho by default (Stambaugh's
    preferred variant): since bias grows with rho and rho_hat is too low,
    evaluating at raw rho_hat would understate the correction.
    """
    rho_used = bias_correct_rho(rho_hat, T) if use_corrected_rho else rho_hat
    bias = analytical_slope_bias(s_uv, s_vv, rho_used, T)
    return beta_hat - bias, bias, rho_used