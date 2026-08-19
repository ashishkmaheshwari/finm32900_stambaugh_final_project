"""
Bayesian posteriors for the predictive slope (Table 2, specs A-D).

The two equations share the same regressors (a constant and x_t), so they form
one multivariate regression Y = X B + E, Y = [r_{t+1}, x_{t+1}]. With a flat
prior and the likelihood conditioned on x_0 (spec A), the posterior is
conjugate: Sigma | data ~ Inverse-Wishart(S, T - k) and
B | Sigma ~ matrix-normal centered at the OLS B_hat with covariance
Sigma (x) (X'X)^{-1}. We sample it directly -- no MCMC needed.

Spec B applies the stationarity prior rho in (-1, 1) by rejection: identical
draws, discarding those with |rho| >= 1. Specs C and D use the exact likelihood, 
which treats x_0 as a draw from the predictor's stationary distribution. 
That term is nonlinear in rho and breaks conjugacy, so rather than build a new 
sampler we reweight the conditional draws by the stationary density (importance sampling); 
spec D additionally reweights by the paper's alternative prior.
"""

import numpy as np
from scipy import stats


def sample_posterior(r_next, x_lag, x_next, n_draws=20_000, seed=0,
                     restrict_rho=False):
    """Draw (beta, rho) from the flat-prior conditional posterior.

    Returns a dict with 'beta' and 'rho' arrays (post-rejection if
    restrict_rho), theta, s_vv, logdet and 'accept_rate'.
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
    thetas = np.empty(n_draws)
    s_vv = np.empty(n_draws)
    logdets = np.empty(n_draws)
    for d in range(n_draws):
        Sigma = stats.invwishart.rvs(df=nu, scale=S, random_state=rng)
        L_s = np.linalg.cholesky(Sigma)
        Z = rng.standard_normal((k, 2))
        B = B_hat + L_x @ Z @ L_s.T
        betas[d] = B[1, 0]
        rhos[d] = B[1, 1]
        thetas[d] = B[0, 1]          # AR(1) intercept
        s_vv[d] = Sigma[1, 1]        # variance of the yield innovation
        logdets[d] = np.log(np.linalg.det(Sigma))                      

    out = {"beta": betas, "rho": rhos, "theta": thetas,
           "s_vv": s_vv, "logdet": logdets, "accept_rate": 1.0}
    if restrict_rho:
        keep = np.abs(rhos) < 1.0
        out = {k: (v[keep] if isinstance(v, np.ndarray) else v)
               for k, v in out.items()}
        out["accept_rate"] = keep.mean()
    return out


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

def exact_likelihood_log_weights(draws, x0, spec="C"):
    """Log importance weights taking conditional-posterior draws to the exact
    likelihood (spec C) or to the exact likelihood under the paper's
    alternative prior (spec D).

    The conditional likelihood treats x_0 as fixed. The exact likelihood adds
    the stationary density of x_0,
        x_0 ~ N(theta/(1-rho), s_vv/(1-rho^2)),
    which is nonlinear in rho and breaks conjugacy. Rather than build a new
    sampler we reweight the conditional draws by that factor -- importance
    sampling, valid because the two posteriors differ by one observation's
    worth of information.

    Spec D additionally reweights by the prior ratio
        (1-rho^2)^{-1} * s_vv * |Sigma|^{-1},
    the paper's p(b,Sigma) ~ (1-rho^2)^{-1} sigma_v^2 |Sigma|^{-5/2} relative to
    the |Sigma|^{-3/2} used in A-C.
    """
    rho = draws["rho"]
    theta = draws["theta"]
    s_vv = draws["s_vv"]

    one_minus = 1.0 - rho**2
    valid = one_minus > 0                      # stationary draws only
    logw = np.full(rho.shape, -np.inf)

    mean0 = theta[valid] / (1.0 - rho[valid])
    var0 = s_vv[valid] / one_minus[valid]
    logw[valid] = -0.5 * np.log(2 * np.pi * var0) - (x0 - mean0) ** 2 / (2 * var0)

    if spec == "D":
        logw[valid] += (
            -np.log(one_minus[valid])
            + np.log(s_vv[valid])
            - draws["logdet"][valid]
        )
    return logw


def weighted_moments(beta_draws, log_weights):
    """Posterior moments under importance weights, plus the effective sample
    size (a diagnostic: if ESS collapses, the reweighting is unreliable)."""
    lw = log_weights - np.max(log_weights[np.isfinite(log_weights)])
    w = np.exp(lw)
    w[~np.isfinite(w)] = 0.0
    w = w / w.sum()

    a = np.asarray(beta_draws)
    mean = np.sum(w * a)
    var = np.sum(w * (a - mean) ** 2)
    sd = np.sqrt(var)
    z = (a - mean) / sd
    return {
        "mean": mean,
        "std": sd,
        "skewness": np.sum(w * z**3),
        "kurtosis": np.sum(w * z**4),
        "p_beta_leq_0": np.sum(w * (a <= 0.0)),
        "ess": 1.0 / np.sum(w**2),
    }