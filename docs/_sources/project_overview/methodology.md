# Methodology

**The mechanism.** In the system $r_{t+1} = \alpha + \beta x_t + u_{t+1}$,
$x_{t+1} = \theta + \rho x_t + v_{t+1}$, the AR(1) estimate of $\rho$ is biased
downward by roughly $(1+3\rho)/T$ (Kendall 1954), and that error transmits into
the slope with its sign flipped:
$E[\hat\beta - \beta] \approx -(\sigma_{uv}/\sigma_{vv})(1+3\rho)/T$. With
$\sigma_{uv} < 0$ (a price shock moves return and yield oppositely) the slope
bias is positive. Our data: $\rho = 0.99$, $\mathrm{corr}(u,v) = -0.95$.

**Table 1** simulates the null ($\beta = 0$) at each subsample's estimated
$(\rho, T, \Sigma)$, 20,000 replications, vectorized over simulations with
accumulated OLS cross-products.

**Table 2** samples four posteriors. Specifications A and B (conditional
likelihood) are conjugate: inverse-Wishart for $\Sigma$, matrix-normal for the
coefficients, with B truncating $\rho$ to $(-1,1)$ by rejection. Specifications
C and D (exact likelihood, with $x_0$ drawn from the predictor's stationary
distribution) are not conjugate and are sampled by random-walk
Metropolis-Hastings in an unconstrained parameterization, with the proposal
covariance estimated from a pilot chain. We validated the sampler by running it
on the conditional likelihood, where it reproduces the conjugate answer. An
earlier importance-sampling approach is retained in the repository, documented
as the attempt whose failure (weights concentrating on draws near $\rho = 1$)
motivated the sampler.

**Reproducibility.** `doit` rebuilds everything: WRDS pulls, the tidy panel,
every table and figure, the executed notebook, this site, and the test suite —
28 tests, split so simulation-based tests run in CI without credentials while
data-dependent tests skip cleanly when the panel is absent.