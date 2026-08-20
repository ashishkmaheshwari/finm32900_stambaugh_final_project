# Project Overview

## Replicating Stambaugh (1999), "Predictive Regressions"

Does the dividend-price ratio predict stock returns? For decades the standard
test regressed next month's return on this month's dividend yield and usually
found a positive, "significant" slope. Stambaugh (1999) showed the test is
broken in a quantifiable way: because the dividend yield is highly persistent
and shares a price with the return, the OLS slope is **biased upward in finite
samples** — a positive slope is what you should expect even when the true slope
is zero.

This project rebuilds the data from CRSP, replicates the paper's Table 1,
Table 2, and Figure 1 within a stated tolerance, and extends every exhibit
through 2024 — where we find the paper's warning binds *harder* than it did in
1999: the gap between the naive and honest p-value has grown from threefold to
tenfold.

**Start here:**

- <a href="cb/notebooks/ashishkmaheshwari--stambaugh_1999_replication/01_walkthrough.html">**The walkthrough notebook**</a> —
  the guided tour: data construction, the bias mechanism, the replication, the
  update.
- <a href="playground.html">**The interactive playground**</a> — drag two
  sliders and watch a regression invent predictability out of nothing.
- **The full report** — the formal write-up with all exhibits
  (<a href="report.pdf">PDF</a>).

```{toctree}
:maxdepth: 1
:caption: Project Details
project_overview/goals
project_overview/data_sources
project_overview/methodology
```