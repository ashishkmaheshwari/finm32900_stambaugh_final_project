"""
Monte Carlo engine for the finite-sample distribution of the OLS predictive
slope (Stambaugh 1999, Table 1 Part A).

Simulates the two-equation system
    r_{t+1} = alpha + beta * x_t + u_{t+1}
    x_{t+1} = theta + rho  * x_t + v_{t+1},   (u,v) ~ N(0, Sigma) iid,
and records the OLS beta_hat and rho_hat on each simulated sample. With
sigma_uv < 0 and rho near 1, beta_hat is biased upward even when beta = 0:
rho_hat is biased downward (Kendall), and E[beta_hat - beta] =
(sigma_uv/sigma_vv) * E[rho_hat - rho] flips that bias's sign into the slope.
"""

import numpy as np


def ols_slope(y, x_lag):
    """OLS slope of y on a constant and x_lag."""
    X = np.column_stack([np.ones(len(x_lag)), x_lag])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef[1]


def simulate_paths(alpha, beta, theta, rho, Sigma, T, x0, n_sims, seed=0):
    """Simulate n_sims samples of length T; return per-sample beta_hat, rho_hat.

    x0 anchors the predictor at the sample start (we use the subsample's first
    observed value -- a simplification of the paper's conditioning, documented).
    """
    rng = np.random.default_rng(seed)
    Sigma = np.asarray(Sigma, dtype=float)
    beta_hats = np.empty(n_sims)
    rho_hats = np.empty(n_sims)

    # Draw all shocks at once: (n_sims, T, 2)
    shocks = rng.multivariate_normal([0.0, 0.0], Sigma, size=(n_sims, T))

    for s in range(n_sims):
        u, v = shocks[s, :, 0], shocks[s, :, 1]
        x = np.empty(T + 1)
        x[0] = x0
        r = np.empty(T)
        for t in range(T):
            r[t] = alpha + beta * x[t] + u[t]        # r_{t+1} paired with x_t
            x[t + 1] = theta + rho * x[t] + v[t]
        beta_hats[s] = ols_slope(r, x[:-1])
        rho_hats[s] = ols_slope(x[1:], x[:-1])

    return beta_hats, rho_hats


def analytical_slope_bias(s_uv, s_vv, rho, T):
    """Kendall/Stambaugh first-order approximation:
    E[beta_hat - beta] ~= -(s_uv/s_vv) * (1 + 3*rho) / T."""
    return -(s_uv / s_vv) * (1.0 + 3.0 * rho) / T


if __name__ == "__main__":
    # THE test: true beta = 0, your estimated 1927-1996 parameters.
    s_uu, s_vv, s_uv = 31.2246e-4, 0.1118e-4, -1.6720e-4
    rho, T = 0.9731, 834
    x0 = 0.04            # a typical D/P level for the era

    beta_hats, rho_hats = simulate_paths(
        alpha=0.0, beta=0.0, theta=(1 - rho) * x0, rho=rho,
        Sigma=[[s_uu, s_uv], [s_uv, s_vv]], T=T, x0=x0,
        n_sims=10_000, seed=42,
    )

    print(f"TRUE beta = 0. Simulated OLS across {len(beta_hats):,} samples:")
    print(f"  mean beta_hat        : {beta_hats.mean():+.4f}   <- the bias")
    print(f"  analytical bias      : {analytical_slope_bias(s_uv, s_vv, rho, T):+.4f}")
    print(f"  std of beta_hat      : {beta_hats.std(ddof=1):.4f}")
    print(f"  mean rho_hat         : {rho_hats.mean():.4f}  (true rho = {rho})")
    print(f"  P(beta_hat > 0)      : {(beta_hats > 0).mean():.3f}")