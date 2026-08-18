"""
Bayesian posteriors for the predictive slope (Table 2, specs A and B).

The two equations share the same regressors (a constant and x_t), so they form
one multivariate regression Y = X B + E, Y = [r_{t+1}, x_{t+1}]. With a flat
prior and the likelihood conditioned on x_0 (spec A), the posterior is
conjugate: Sigma | data ~ Inverse-Wishart(S, T - k) and
B | Sigma ~ matrix-normal centered at the OLS B_hat with covariance
Sigma (x) (X'X)^{-1}. We sample it directly -- no MCMC needed.

Spec B applies the stationarity prior rho in (-1, 1) by rejection: identical
draws, discarding those with |rho| >= 1. Specs C and D (exact likelihood,
treating x_0 as a draw from the stationary distribution) require
Metropolis-Hastings and are documented as an extension in the report.
"""

import numpy as np
from scipy import stats


def sample_posterior(r_next, x_lag, x_next, n_draws=20_000, seed=0,
                     restrict_rho=False):
    """Draw (beta, rho) from the flat-prior conditional posterior.

    Returns a dict with 'beta' and 'rho' arrays (post-rejection if
    restrict_rho) and 'accept_rate'.
    """
    Y = np.column_stack([r_next, x_next])
    X = np.column_stack([np.ones(len(x_lag)), x_lag])
    T, k = X.shape

    XtX_inv = np.linalg.inv(X.T @ X)
    B_hat = XtX_inv @ X.T @ Y                  # (k, 2): col 0 return eq, col 1 AR(1)
    E = Y - X @ B_hat
    S = E.T @ E                                # residual cross-product
    nu = T - k

    rng = np.random.default_rng(seed)
    L_x = np.linalg.cholesky(XtX_inv)

    betas = np.empty(n_draws)
    rhos = np.empty(n_draws)
    for d in range(n_draws):
        Sigma = stats.invwishart.rvs(df=nu, scale=S, random_state=rng)
        L_s = np.linalg.cholesky(Sigma)
        Z = rng.standard_normal((k, 2))
        B = B_hat + L_x @ Z @ L_s.T            # matrix-normal draw
        betas[d] = B[1, 0]                     # slope on x_t in return equation
        rhos[d] = B[1, 1]                      # slope on x_t in AR(1)

    if restrict_rho:
        keep = np.abs(rhos) < 1.0
        return {"beta": betas[keep], "rho": rhos[keep],
                "accept_rate": keep.mean()}
    return {"beta": betas, "rho": rhos, "accept_rate": 1.0}


def posterior_moments(beta_draws):
    """Table 2 row set: mean, sd, skewness, raw kurtosis, P(beta <= 0)."""
    a = np.asarray(beta_draws)
    z = (a - a.mean()) / a.std(ddof=0)
    return {
        "mean": a.mean(),
        "std": a.std(ddof=1),
        "skewness": (z**3).mean(),
        "kurtosis": (z**4).mean(),
        "p_beta_leq_0": np.mean(a <= 0.0),
    }