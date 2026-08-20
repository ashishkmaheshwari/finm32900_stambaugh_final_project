# Stambaugh 1999 Replication

Last updated: {sub-ref}`today`


```{toctree}
:maxdepth: 1
:caption: Project Notes

project_overview
```



## Table of Contents

```{toctree}
:maxdepth: 1
:caption: Notebooks 📖
Walkthrough: replicating Stambaugh (1999) <cb/notebooks/ashishkmaheshwari--stambaugh_1999_replication/01_walkthrough>
```



```{toctree}
:maxdepth: 1
:caption: Pipeline Charts 📈
cb/charts.md
```

```{postlist}
:format: "{title}"
```


```{toctree}
:maxdepth: 1
:caption: Pipeline Dataframes 📊

```


```{toctree}
:maxdepth: 1
:caption: Appendix 💡
myst_markdown_demos.md
apidocs/index
```


## Pipeline Specs
| Pipeline Name                   | Stambaugh 1999 Replication                       |
|---------------------------------|--------------------------------------------------------|
| Pipeline ID                     | [ashishkmaheshwari/stambaugh_1999_replication](./index.md)              |
| Maintainer                      | Ashish Maheshwari & Omar Anabtawi               |
| Contributors                    | Ashish Maheshwari & Omar Anabtawi |
| Repository                     |                   |
| Pipeline Web Page               | <a href="file://C:/Users/ashis/Git/Full-Stack-QF/stambaugh_1999_replication/docs/index.html">Pipeline Web Page      |
| Date of Last Code Update        | 2026-08-19 23:31:05           |
| OS Compatibility                | Windows, Linux, macOS |
| Linked Dataframes               |  |


**Build Commands:**
```
pip install -r requirements.txt
doit

```


Does the dividend–price ratio predict stock returns? For decades the standard
test regressed next month's excess return on this month's dividend yield and
usually found a positive, "significant" slope. Stambaugh (1999, *Journal of
Financial Economics* 54) showed the test is broken in a quantifiable way: the
dividend yield is highly persistent and shares a price with the return, so the
OLS slope is **biased upward in finite samples**. A positive slope is what you
should expect even when the true slope is zero.

This project rebuilds the data from CRSP, replicates the paper's Table 1,
Table 2, and Figure 1 within a documented tolerance, and extends every exhibit
through 2024 — where the problem turns out to be worse than in the paper's own
sample: the gap between the naive and the finite-sample p-value has grown from
roughly threefold to tenfold.

**[Read the project site](https://ashishkmaheshwari.github.io/finm32900_stambaugh_final_project/)**
— overview, walkthrough notebook, interactive playground, and full report.

## Exhibits

| Exhibit | Content | Headline result |
|---|---|---|
| Table 1 | Finite-sample properties of the OLS slope, by subsample | Our finite-sample p-value 0.177 vs the paper's 0.17 |
| Table 2 | Bayesian posteriors under four prior/likelihood specifications | All sixteen cells within ~0.03 of the paper |
| Figure 1 | β vs ρ across methods and subperiods | Reproduces the paper's ρ > 1 overshoot in 1977–96 |
| Updates | Tables 1 and 2 on samples through 2024 | Naive p 0.015 vs honest p 0.149 in 1997–2024 |

Plus two educational products: a guided walkthrough notebook and an interactive
browser playground where two sliders show the bias growing with persistence and
shrinking with sample size.

## Repository structure

```
.
├── dodo.py                     # PyDoit build file — runs everything
├── chartbook.toml              # site configuration
├── requirements.txt
├── .env.example                # template for your .env (WRDS_USERNAME)
├── src/
│   ├── settings.py             # configuration: paths, dates, credentials
│   ├── pull_CRSP_index.py      # WRDS pull: CRSP monthly market index
│   ├── pull_fama_french.py     # WRDS pull: risk-free rate
│   ├── calc_predictor_data.py  # DATA CLEANING ONLY -> tidy monthly panel
│   ├── monte_carlo.py          # simulation engine (Table 1 Part A)
│   ├── stambaugh_bias.py       # bias-corrected estimators (Figure 1)
│   ├── bayesian.py             # conjugate posteriors (Table 2 specs A, B)
│   ├── mcmc.py                 # Metropolis-Hastings (Table 2 specs C, D)
│   ├── create_table_01_partC.py
│   ├── create_table_01.py      # Table 1 assembly -> _output/*.tex
│   ├── create_table_02.py      # Table 2 assembly -> _output/*.tex
│   ├── create_figure_01.py     # Figure 1 -> _output/figure_01.png
│   ├── 01_walkthrough.ipynb.py # guided tour notebook (jupytext source)
│   └── test_*.py               # unit tests
├── reports/report.tex          # the write-up; inputs the generated exhibits
├── docs_src/                   # site sources (edit these)
├── docs/                       # BUILT site — generated, served by Pages
├── _data/                      # pulled and cleaned data (git-ignored)
└── _output/                    # generated tables, figures (git-ignored)
```

Data cleaning lives in its own file, separate from all analysis. Raw data never
enters the repository.

## Setup

```bash
conda create -n stambaugh python=3.12 -y
conda activate stambaugh
pip install -r requirements.txt
cp .env.example .env      # then set WRDS_USERNAME=your_login
```

The first WRDS connection prompts for your password and offers to create a
`.pgpass` file so later runs are non-interactive. `.env` is git-ignored and must
never be committed.

## Running it

```bash
doit
```

That pulls from WRDS, builds the tidy panel, regenerates every table and figure,
compiles the report, executes the notebook, rebuilds the site, and runs the
tests. PyDoit tracks dependencies, so re-running rebuilds only what changed.

Individual stages:

```bash
doit pull                  # WRDS pulls
doit clean_data            # tidy panel
doit table_01 figure_01    # paper-sample exhibits
doit table_01_updated      # extended-sample exhibits
doit notebook              # execute the walkthrough
doit compile_latex_docs    # report PDF
doit build_chartbook_site  # the published site
doit run_pytest            # test suite
```

Note that `table_02` runs eight Metropolis-Hastings chains and takes several
minutes.

## Testing

```bash
pytest -q src/
```

The suite is split deliberately. Simulation-based tests verify the bias
mechanism itself — that the bias is positive when innovations are negatively
correlated, vanishes when they are not, shrinks like 1/T, and matches the
Kendall/Stambaugh analytical formula — and run anywhere, including CI without
credentials. Data-dependent tests check our estimates against the paper's
published values within stated tolerances, and skip with an explanatory message
when the panel has not been built.

## Data sources

- **CRSP Monthly Stock Market Indexes** (`crsp.msi`, WRDS) — value-weighted
  returns with (`vwretd`) and without (`vwretx`) dividends. Their difference
  gives the dividend series, which is how the dividend–price ratio is
  reconstructed without a separate dividend file.
- **Fama–French monthly factors** (`ff.factors_monthly`, WRDS) — the one-month
  risk-free rate, for continuously compounded excess returns.

Stambaugh uses a NYSE-only value-weighted index; our WRDS instance provides no
pre-built NYSE-only monthly index carrying both return columns, so we use the
CRSP total-market value-weighted index and document the choice in the report.

## A note on `docs/`

The built site is committed so GitHub Pages can serve it directly without a
build step. Edit `docs_src/`, never `docs/` — the latter is regenerated by
`doit build_chartbook_site` and hand edits are lost.

## Team

- Ashish Maheshwari
- Omar Anabtawi

FINM 32900, Full-Stack Quantitative Finance, Summer 2026.