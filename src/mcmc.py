"""
Metropolis-Hastings sampler for the exact-likelihood posteriors (Table 2,
specifications C and D).

Why this module exists
----------------------
Specifications A and B condition the likelihood on x_0 and use a flat prior, so
their posterior is conjugate and can be sampled directly (see bayesian.py). The
exact likelihood adds the stationary density of x_0,

    x_0 ~ N( theta/(1-rho),  sigma_v^2 / (1-rho^2) ),

which is nonlinear in rho and destroys conjugacy.

We first attempted to reach C and D by importance-reweighting the conditional
draws. That works for the long, lower-persistence samples but fails in the
post-war samples: specification D's prior carries a (1-rho^2)^{-1} factor that
explodes as rho approaches one, so the weights concentrate on the handful of
draws nearest the boundary. Diagnostically, in 1952-1996 the ten heaviest draws
all had rho > 0.999 and the weighted mean of rho jumped from 0.981 to 0.989.
The proposal simply does not visit the region the target cares about, and more
draws would not fix it. This module therefore samples the exact posterior
directly.

Parameterization
----------------
The chain moves in an unconstrained space -- (alpha, beta, theta, rho,
log sigma_u, log sigma_v, atanh corr) -- so every proposal yields a valid
positive-definite Sigma. Because the priors are stated with respect to the
elements of Sigma, the change of variables contributes a Jacobian term
3*log_su + 3*log_sv + log(1 - corr^2) to the log target; omitting it would
silently alter the prior.

Proposal
--------
beta and rho are strongly correlated in this posterior -- that correlation is
the Stambaugh mechanism itself -- and the seven parameters live on very
different scales. A diagonal random walk therefore mixes badly: it must inch
along a narrow diagonal ridge, producing good acceptance rates but tiny
effective sample sizes. We instead run a short pilot chain, estimate the
posterior covariance, and propose with that covariance scaled by the standard
2.4/sqrt(d) rule. Retained draws are thinned to reduce residual autocorrelation.

Priors
------
    C: p(b, Sigma) proportional to |Sigma|^{-3/2},           rho in (-1, 1)
    D: p(b, Sigma) proportional to (1-rho^2)^{-1} sigma_v^2 |Sigma|^{-5/2},
                                                             rho in (-1, 1)
"""

import numpy as np

N_PARAMS = 7


def _unpack(params):
    """Map the unconstrained parameter vector to model quantities."""
    alpha, beta, theta, rho, log_su, log_sv, z = params
    su = np.exp(log_su)
    sv = np.exp(log_sv)
    corr = np.tanh(z)
    return alpha, beta, theta, rho, su, sv, corr


def log_target(params, r_next, x_lag, x_next, spec="C", exact=True):
    """Log posterior density (up to an additive constant).

    Parameters
    ----------
    params : array of 7 unconstrained parameters, see module docstring.
    r_next, x_lag, x_next : aligned data arrays of length T.
    spec : "C" (|Sigma|^{-3/2} prior) or "D" (the paper's alternative prior).
    exact : if True, include the stationary density of x_0; if False, use the
        conditional likelihood (used to validate the sampler against the
        conjugate specification A).
    """
    if not np.isfinite(params).all():
        return -np.inf

    alpha, beta, theta, rho, su, sv, corr = _unpack(params)

    if abs(rho) >= 1.0:
        return -np.inf          # stationarity restriction, specs B-D

    u = r_next - alpha - beta * x_lag
    v = x_next - theta - rho * x_lag
    T = len(u)

    det = su**2 * sv**2 * (1.0 - corr**2)
    if det <= 0 or not np.isfinite(det):
        return -np.inf

    zu = u / su
    zv = v / sv
    quad = (zu**2 - 2.0 * corr * zu * zv + zv**2).sum() / (1.0 - corr**2)
    loglik = -0.5 * T * np.log(det) - 0.5 * quad

    if exact:
        mean0 = theta / (1.0 - rho)
        var0 = sv**2 / (1.0 - rho**2)
        if var0 <= 0:
            return -np.inf
        x0 = x_lag[0]
        loglik += -0.5 * np.log(var0) - (x0 - mean0) ** 2 / (2.0 * var0)

    log_det_sigma = np.log(det)
    if spec == "C":
        logprior = -1.5 * log_det_sigma
    elif spec == "D":
        logprior = (
            -np.log(1.0 - rho**2)
            + 2.0 * np.log(sv)
            - 2.5 * log_det_sigma
        )
    else:
        raise ValueError("spec must be 'C' or 'D'")

    log_jacobian = 3.0 * np.log(su) + 3.0 * np.log(sv) + np.log(1.0 - corr**2)

    return loglik + logprior + log_jacobian


def initial_params(r_next, x_lag, x_next):
    """Start the chain at the OLS / conditional estimates."""
    X = np.column_stack([np.ones(len(x_lag)), x_lag])
    coef_r, *_ = np.linalg.lstsq(X, r_next, rcond=None)
    coef_x, *_ = np.linalg.lstsq(X, x_next, rcond=None)
    u = r_next - X @ coef_r
    v = x_next - X @ coef_x
    su = u.std(ddof=2)
    sv = v.std(ddof=2)
    corr = np.corrcoef(u, v)[0, 1]
    rho = min(max(coef_x[1], -0.999), 0.999)
    return np.array([coef_r[0], coef_r[1], coef_x[0], rho,
                     np.log(su), np.log(sv), np.arctanh(corr)])


def _pilot_covariance(r_next, x_lag, x_next, spec, exact, n_pilot, seed):
    """Estimate the posterior covariance from a short diagonal-proposal chain.

    The pilot need not mix well; it only has to visit enough of the posterior
    to reveal its shape, which is then used as the proposal covariance for the
    production chain.
    """
    rng = np.random.default_rng(seed)
    p = initial_params(r_next, x_lag, x_next)
    lp = log_target(p, r_next, x_lag, x_next, spec=spec, exact=exact)

    step = np.array([0.01, 0.05, 0.01, 0.002, 0.02, 0.02, 0.02])
    samples = np.empty((n_pilot, N_PARAMS))
    accepted = 0
    for i in range(n_pilot):
        prop = p + step * rng.standard_normal(N_PARAMS)
        lp_prop = log_target(prop, r_next, x_lag, x_next, spec=spec, exact=exact)
        if np.log(rng.random()) < lp_prop - lp:
            p, lp = prop, lp_prop
            accepted += 1
        if i % 100 == 99:
            rate = accepted / 100.0
            step *= np.exp(rate - 0.30)
            accepted = 0
        samples[i] = p

    cov = np.cov(samples[n_pilot // 2:].T)
    cov += np.eye(N_PARAMS) * 1e-14      # guard against singularity
    return cov


def run_mh(r_next, x_lag, x_next, spec="C", exact=True,
           n_draws=200_000, burn_in=20_000, thin=10, seed=0,
           target_accept=0.25, n_pilot=8_000):
    """Random-walk Metropolis-Hastings on the exact posterior.

    A pilot chain supplies the proposal covariance; the scale is adapted during
    burn-in toward ``target_accept`` and then held fixed, so retained draws come
    from a time-homogeneous chain. Draws are thinned by ``thin``.

    Returns
    -------
    dict with 'beta' and 'rho' arrays (post burn-in, thinned), 'accept_rate',
    and 'n_eff_beta' (effective sample size of the retained beta draws).
    """
    cov = _pilot_covariance(r_next, x_lag, x_next, spec, exact, n_pilot, seed)
    L = np.linalg.cholesky(cov)
    scale = 2.4 / np.sqrt(N_PARAMS)

    rng = np.random.default_rng(seed + 1)
    p = initial_params(r_next, x_lag, x_next)
    lp = log_target(p, r_next, x_lag, x_next, spec=spec, exact=exact)

    n_keep = n_draws // thin
    keep_beta = np.empty(n_keep)
    keep_rho = np.empty(n_keep)

    total = burn_in + n_draws
    n_accept = 0
    window_accept = 0
    k = 0

    for i in range(total):
        prop = p + scale * (L @ rng.standard_normal(N_PARAMS))
        lp_prop = log_target(prop, r_next, x_lag, x_next, spec=spec, exact=exact)
        if np.log(rng.random()) < lp_prop - lp:
            p, lp = prop, lp_prop
            n_accept += 1
            window_accept += 1

        if i < burn_in and (i + 1) % 500 == 0:
            rate = window_accept / 500.0
            scale *= np.exp(rate - target_accept)
            window_accept = 0

        if i >= burn_in:
            j = i - burn_in
            if j % thin == 0 and k < n_keep:
                keep_beta[k] = p[1]
                keep_rho[k] = p[3]
                k += 1

    return {
        "beta": keep_beta[:k],
        "rho": keep_rho[:k],
        "accept_rate": n_accept / total,
        "n_eff_beta": effective_sample_size(keep_beta[:k]),
    }


def effective_sample_size(chain, max_lag=500):
    """Effective sample size from the chain's autocorrelation function.

    Uses the initial-positive-sequence rule: sum autocorrelations until the
    first non-positive value.
    """
    a = np.asarray(chain, dtype=float)
    a = a - a.mean()
    n = len(a)
    var = a @ a / n
    if var == 0:
        return float(n)
    total = 0.0
    for lag in range(1, min(max_lag, n - 1)):
        ac = (a[lag:] @ a[:-lag]) / (n * var)
        if ac <= 0:
            break
        total += ac
    return float(n / (1.0 + 2.0 * total))


def posterior_moments_from_chain(beta_draws):
    """Same row set as bayesian.posterior_moments, for MH output."""
    a = np.asarray(beta_draws)
    z = (a - a.mean()) / a.std(ddof=0)
    return {
        "mean": a.mean(),
        "std": a.std(ddof=1),
        "skewness": (z**3).mean(),
        "kurtosis": (z**4).mean(),
        "p_beta_leq_0": np.mean(a <= 0.0),
    }