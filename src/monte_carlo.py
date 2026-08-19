"""
Monte Carlo engine for the finite-sample distribution of the OLS predictive
slope (Stambaugh 1999, Table 1 Part A).

Simulates the two-equation system
    r_{t+1} = alpha + beta * x_t + u_{t+1}
    x_{t+1} = theta + rho  * x_t + v_{t+1},   (u,v) ~ N(0, Sigma) iid,
and characterizes the OLS slope across simulated samples: bias, standard
deviation, skewness, (raw) kurtosis, and the finite-sample "true" p-value --
the fraction of null (beta = 0) simulations whose slope exceeds the slope
observed in the real data.

Implementation note: the simulation loops over TIME and carries all n_sims
paths forward together as vectors, accumulating the OLS cross-products
(sum x, sum x^2, sum x*y, ...) on the fly. This avoids both a per-simulation
Python loop and storing (n_sims x T) path arrays.
"""

import numpy as np


def ols_slope(y, x_lag):
    """OLS slope of y on a constant and x_lag (small-sample helper for tests)."""
    X = np.column_stack([np.ones(len(x_lag)), x_lag])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef[1]


def simulate_slopes(alpha, beta, theta, rho, Sigma, T, x0, n_sims, seed=0):
    """Simulate n_sims samples of length T; return (beta_hats, rho_hats).

    All simulations advance together: x is an (n_sims,) vector updated T times.
    OLS slopes are computed from accumulated sums via
        slope = (T * S_xy - S_x * S_y) / (T * S_xx - S_x**2).
    """
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(np.asarray(Sigma, dtype=float))

    x = np.full(n_sims, float(x0))

    # Accumulators. Predictive regression: y = r_{t+1} on x_t.
    Sx = np.zeros(n_sims); Sxx = np.zeros(n_sims)
    Sr = np.zeros(n_sims); Sxr = np.zeros(n_sims)
    # AR(1): y = x_{t+1} on x_t (shares Sx, Sxx).
    Sxn = np.zeros(n_sims); Sxxn = np.zeros(n_sims)

    for _ in range(T):
        z = rng.standard_normal((2, n_sims))
        u, v = L @ z                       # correlated shocks, each (n_sims,)
        r = alpha + beta * x + u           # r_{t+1} paired with current x_t
        x_next = theta + rho * x + v

        Sx += x;      Sxx += x * x
        Sr += r;      Sxr += x * r
        Sxn += x_next; Sxxn += x * x_next

        x = x_next

    beta_hats = (T * Sxr - Sx * Sr) / (T * Sxx - Sx**2)
    rho_hats = (T * Sxxn - Sx * Sxn) / (T * Sxx - Sx**2)
    return beta_hats, rho_hats


def summarize_distribution(beta_hats, true_beta=0.0):
    """Part A moments: bias, std, skewness, and RAW kurtosis (normal = 3),
    matching the paper's convention (see the 3's in Table 1, Part B)."""
    a = np.asarray(beta_hats, dtype=float)
    m, s = a.mean(), a.std(ddof=1)
    z = (a - m) / a.std(ddof=0)
    return {
        "bias": m - true_beta,
        "std": s,
        "skewness": (z**3).mean(),
        "kurtosis": (z**4).mean(),
    }


def true_pvalue(beta_obs, null_beta_hats):
    """Finite-sample p-value for the test of beta = 0 vs beta > 0: the share
    of null-simulated slopes at least as large as the observed slope."""
    return float(np.mean(np.asarray(null_beta_hats) >= beta_obs))


def analytical_slope_bias(s_uv, s_vv, rho, T):
    """Kendall/Stambaugh first-order approximation:
    E[beta_hat - beta] ~= -(s_uv/s_vv) * (1 + 3*rho) / T."""
    return -(s_uv / s_vv) * (1.0 + 3.0 * rho) / T


if __name__ == "__main__":
    # Full Part A column for 1927-1996, using last night's Part C estimates.
    s_uu, s_vv, s_uv = 31.2246e-4, 0.1118e-4, -1.6720e-4
    rho, T = 0.9731, 834
    x0 = 0.04
    beta_obs = 0.2100                     # our estimated slope (Part C)

    beta_hats, rho_hats = simulate_slopes(
        alpha=0.0, beta=0.0, theta=(1 - rho) * x0, rho=rho,
        Sigma=[[s_uu, s_uv], [s_uv, s_vv]], T=T, x0=x0,
        n_sims=20_000, seed=42,
    )
    stats = summarize_distribution(beta_hats, true_beta=0.0)
    p = true_pvalue(beta_obs, beta_hats)

    print("Part A, 1927-1996 (true beta = 0):        paper:")
    print(f"  bias      : {stats['bias']:+.4f}          0.07")
    print(f"  std       : {stats['std']:.4f}           0.16")
    print(f"  skewness  : {stats['skewness']:+.4f}          0.71")
    print(f"  kurtosis  : {stats['kurtosis']:.4f}           3.84")
    print(f"  p(beta=0) : {p:.4f}           0.17")