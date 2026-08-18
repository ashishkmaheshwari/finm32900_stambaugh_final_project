"""
Unit tests for the Table 1 replication.

These tests encode our replication tolerances against Stambaugh (1999),
Table 1. Exact digit matching is impossible (CRSP data revisions since 1999;
total-market vs NYSE value-weighted index; sample begins 1927-06 due to
risk-free availability plus the trailing-12-month dividend window), so each
assertion states the tolerance we claim: tight where the quantity is stable
(persistence, signs, T), looser where it is sensitive (the 1977-96 bias,
which is hypersensitive to rho near one).
"""



import numpy as np
import pytest

from calc_predictor_data import load_tidy_panel
from create_table_01_partC import SUBSAMPLES, fit_subsample

from pathlib import Path

from settings import config

# The tidy panel is built from WRDS, which requires credentials that are
# (correctly) not available in CI. Data-dependent tests skip when the panel
# is absent; the simulation-based tests in test_monte_carlo.py always run.
PANEL_PATH = Path(config("DATA_DIR")) / "predictor_panel.parquet"

pytestmark = pytest.mark.skipif(
    not PANEL_PATH.exists(),
    reason="predictor panel not available (requires WRDS pull); run `doit` first",
)

# Paper values, Table 1 Part C, columns in SUBSAMPLES order.
PAPER = {
    "1927-1996": dict(beta=0.21, rho=0.972, T=840, s_uv=-1.621),
    "1927-1951": dict(beta=0.21, rho=0.948, T=300, s_uv=-3.360),
    "1952-1996": dict(beta=0.44, rho=0.980, T=540, s_uv=-0.651),
    "1977-1996": dict(beta=0.19, rho=0.987, T=240, s_uv=-0.715),
}


@pytest.fixture(scope="module")
def fits():
    panel = load_tidy_panel()
    return {name: fit_subsample(panel, s, e) for name, (s, e) in SUBSAMPLES.items()}


@pytest.mark.parametrize("name", list(SUBSAMPLES))
def test_persistence_matches_paper(fits, name):
    """rho is the most stable quantity; claim +/- 0.005."""
    assert abs(fits[name]["rho_hat"] - PAPER[name]["rho"]) < 0.005


@pytest.mark.parametrize("name", list(SUBSAMPLES))
def test_slope_matches_paper(fits, name):
    """beta within 0.05 of the paper (data-vintage + universe differences)."""
    assert abs(fits[name]["beta_hat"] - PAPER[name]["beta"]) < 0.05


@pytest.mark.parametrize("name", list(SUBSAMPLES))
def test_sample_size_matches_paper(fits, name):
    """T within 6 months (panel starts 1927-06; see module docstring)."""
    assert abs(fits[name]["T"] - PAPER[name]["T"]) <= 6


@pytest.mark.parametrize("name", list(SUBSAMPLES))
def test_innovation_covariance_negative_and_close(fits, name):
    """The mechanism: sigma_uv < 0 everywhere, within 10% of the paper."""
    s = fits[name]["sigma_uv_x1e4"]
    assert s < 0
    assert abs(s - PAPER[name]["s_uv"]) < 0.1 * abs(PAPER[name]["s_uv"])


def test_naive_p_less_than_true_story_holds(fits):
    """Sanity on the exhibit's punchline inputs: naive p for 1952-1996 is
    'significant' (< 0.05) while Part A's simulated p (~0.17) is not; here we
    assert the naive side, the simulated side is covered in test_monte_carlo."""
    assert fits["1952-1996"]["p_naive"] < 0.05