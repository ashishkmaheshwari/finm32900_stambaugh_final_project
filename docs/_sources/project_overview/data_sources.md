# Data Sources

Two WRDS pulls supply everything.

**CRSP monthly market index** (`crsp.msi`) — the pre-aggregated value-weighted
market index. The key columns are `vwretd` (return including dividends) and
`vwretx` (return excluding dividends): their difference is the month's dividend
as a fraction of the prior month's price, which lets us reconstruct the
dividend-price ratio without a separate dividend file. Compounding `vwretx`
gives a price level, the trailing twelve months of backed-out dividends give an
annual dividend, and the predictor is their ratio.

**Fama-French monthly factors** (`ff.factors_monthly`) — the one-month
Treasury-bill rate, used to form continuously compounded excess returns
(matching the paper's specification; using simple returns overstates the slope
by ~50% in the volatile pre-war years).

**Universe note:** Stambaugh uses a NYSE-only value-weighted index; we use the
CRSP total-market VW index because our WRDS instance carries no pre-built
NYSE-only monthly index with both return columns. Both are value-weighted, so
the same mega-cap firms dominate both, and the universes coincide for the first
half of the sample. A NYSE-only rebuild from the stock file is a separate
robustness check.

The merged panel runs June 1927 to December 2024 (1,171 months). Raw data lives
only in the git-ignored `_data/` folder and never enters the repository.