import sys; sys.path.insert(0, "src")
from calc_predictor_data import load_tidy_panel
from create_table_02 import _window_arrays
from mcmc import run_mh, posterior_moments_from_chain

p = load_tidy_panel()
r_next, x_lag, x_next = _window_arrays(p, "1927-01-01", "1996-12-31")

for spec in ("C", "D"):
    out = run_mh(r_next, x_lag, x_next, spec=spec, exact=True, seed=1,
                 n_draws=400_000, thin=1)
    print(f"\nSpec {spec}: accept {out['accept_rate']:.1%}, ESS {out['n_eff_beta']:,.0f}")
    print({k: round(v, 4) for k, v in posterior_moments_from_chain(out["beta"]).items()})
    print(f"  posterior mean rho = {out['rho'].mean():.4f}")