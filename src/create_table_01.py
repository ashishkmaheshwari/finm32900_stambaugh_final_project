"""
Assemble Table 1: finite-sample properties of the OLS predictive slope.

For each Stambaugh subsample, Part C parameters (rho, T, Sigma, beta_hat) are
estimated from the tidy panel by create_table_01_partC; this module simulates
the null (beta = 0) at those parameter values and reports Part A: bias,
standard deviation, skewness, kurtosis, and the finite-sample p-value of the
observed slope. Output: _output/table_01.tex, plus the Part A/C frames for
reuse by the figure and notebook.
"""

from pathlib import Path

import pandas as pd

from settings import config
from calc_predictor_data import load_tidy_panel
import sys
from create_table_01_partC import SUBSAMPLES, UPDATE_SUBSAMPLES, fit_subsample
from monte_carlo import simulate_slopes, summarize_distribution, true_pvalue

OUTPUT_DIR = Path(config("OUTPUT_DIR"))

N_SIMS = 20_000
SEED = 42


def build_table_01(panel=None, n_sims=N_SIMS, seed=SEED, subsamples=SUBSAMPLES):
    """Return (partA, partB, partC) DataFrames, one column per subsample."""
    if panel is None:
        panel = load_tidy_panel()

    partC_cols, partA_cols, partB_cols = {}, {}, {}
    for name, (start, end) in subsamples.items():
        c = fit_subsample(panel, start, end)
        partC_cols[name] = c

        partB_cols[name] = {
            "bias": 0.0,
            "std": c["se_beta_naive"],
            "skewness": 0.0,
            "kurtosis": 3.0,
            "p_value_beta0": c["p_naive"],
        }

        # First observed predictor value in the window anchors the simulation.
        window = panel[(panel["date"] >= start) & (panel["date"] <= end)]
        x0 = float(window["dp_ratio"].iloc[0])

        rho, T = c["rho_hat"], int(c["T"])
        Sigma = [
            [c["sigma2_u_x1e4"] * 1e-4, c["sigma_uv_x1e4"] * 1e-4],
            [c["sigma_uv_x1e4"] * 1e-4, c["sigma2_v_x1e4"] * 1e-4],
        ]
        beta_hats, _ = simulate_slopes(
            alpha=0.0, beta=0.0, theta=(1.0 - rho) * x0, rho=rho,
            Sigma=Sigma, T=T, x0=x0, n_sims=n_sims, seed=seed,
        )
        a = summarize_distribution(beta_hats, true_beta=0.0)
        a["p_value_beta0"] = true_pvalue(c["beta_hat"], beta_hats)
        partA_cols[name] = a

    return pd.DataFrame(partA_cols), pd.DataFrame(partB_cols), pd.DataFrame(partC_cols)

# Paper-style presentation: row order, labels, and per-row rounding.
PARTC_LAYOUT = [
    # (internal name,   paper label,            decimals)
    ("beta_hat",        r"$\hat\beta$",          2),
    ("T",               r"$T$",                  0),
    ("rho_hat",         r"$\rho$",               3),
    ("sigma2_u_x1e4",   r"$\sigma_u^2 \times 10^4$", 2),
    ("sigma2_v_x1e4",   r"$\sigma_v^2 \times 10^4$", 3),
    ("sigma_uv_x1e4",   r"$\sigma_{uv} \times 10^4$", 3),
]


def format_partC(partC):
    """Part C laid out as in the paper. Values are formatted to strings so the
    display (console and LaTeX) is exactly what we specify -- no dtype coercion
    of the integer T row, no floating-point residue in to_latex."""
    rows = {}
    for name, label, dec in PARTC_LAYOUT:
        vals = partC.loc[name]
        fmt = "{:.0f}" if dec == 0 else f"{{:.{dec}f}}"
        rows[label] = vals.map(fmt.format)
    return pd.DataFrame(rows).T

PARTAB_LAYOUT = [
    ("bias",          "Bias",                2),
    ("std",           "Standard deviation",  2),
    ("skewness",      "Skewness",            2),
    ("kurtosis",      "Kurtosis",            2),
    ("p_value_beta0", r"$p$-value for $\beta=0$", 2),
]


def format_partAB(part):
    """Parts A/B with the paper's row labels, two decimals, as strings."""
    rows = {}
    for name, label, dec in PARTAB_LAYOUT:
        rows[label] = part.loc[name].map(f"{{:.{dec}f}}".format)
    return pd.DataFrame(rows).T

if __name__ == "__main__":
    updated = "--updated" in sys.argv
    subsamples = UPDATE_SUBSAMPLES if updated else SUBSAMPLES
    suffix = "_updated" if updated else ""

    partA, partB, partC = build_table_01(subsamples=subsamples)

    print("Part A (simulated, true beta = 0):")
    print(partA.round(4).to_string())
    print("\nPart B (standard regression setting):")
    print(partB.round(4).to_string())
    print("\nPart C (sample characteristics):")
    print(format_partC(partC).to_string())


    combined = pd.concat(
        {"A. Simulated (true $\\beta=0$)": format_partAB(partA),
         "B. Standard regression": format_partAB(partB),
         "C. Sample characteristics": format_partC(partC)}
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if updated:
        caption = (
            "Finite-sample properties of the OLS predictive slope, samples "
            "extended through 2024. Structure follows Table~\\ref{tab:table1}: "
            "Part A simulates the null at the Part C parameters, Part B gives "
            "the standard regression model's assertions. The 1997--2024 column "
            "lies entirely outside Stambaugh's original sample. Takeaway: the "
            "bias falls as the sample lengthens (0.07 to 0.06 in the full "
            "sample, tracking the $1/T$ rate), but in the modern subsamples it "
            "grows large relative to the slope itself, and the gap between the "
            "naive and finite-sample $p$-values widens to tenfold in "
            "1997--2024 (0.01 against 0.15)."
        )
    else:
        caption = (
            "Finite-sample properties of the OLS predictive slope "
            "(replication of Stambaugh 1999, Table 1). "
            "Part A reports the simulated distribution of the OLS slope when "
            "the true slope is zero, using the parameters of Part C "
            "(20{,}000 replications): the estimator is biased upward, "
            "right-skewed, and fat-tailed. "
            "Part B reports the same properties as asserted by the standard "
            "regression model -- unbiased, symmetric, normal -- with the "
            "textbook standard error and naive $p$-value. "
            "Part C reports the sample characteristics that drive the gap: a "
            "predictor with persistence near one whose innovations are "
            "strongly negatively correlated with return innovations "
            "($\\sigma_{uv}<0$). "
            "Takeaway: the honest $p$-values in Part A are roughly three to "
            "ten times the naive ones in Part B -- most starkly in 1952--1996, "
            "where the naive test rejects no-predictability at 1.4\\% while "
            "the finite-sample test cannot reject it at 16.5\\%."
        )

    latex = combined.to_latex(
        escape=False,
        caption=caption,
        label=f"tab:table1{suffix}",
    )
    (OUTPUT_DIR / f"table_01{suffix}.tex").write_text(latex)
    print(f"\nWrote {OUTPUT_DIR / f"table_01{suffix}.tex"}")