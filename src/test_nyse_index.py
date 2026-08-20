"""Tests for the NYSE-only stock-file robustness panel."""

from pathlib import Path

import pandas as pd
import pytest

NYSE_PANEL_PATH = Path("_data") / "predictor_panel_nyse.parquet"


def test_nyse_panel_has_expected_coverage_and_columns():
    """The NYSE robustness panel should cover the paper sample and key columns."""
    if not NYSE_PANEL_PATH.exists():
        pytest.skip("NYSE panel has not been generated locally.")

    panel = pd.read_parquet(NYSE_PANEL_PATH)

    expected_columns = {
        "date",
        "vwretd",
        "vwretx",
        "rf",
        "ret_excess",
        "totval",
        "price_index",
        "div_monthly",
        "div_ttm",
        "dp_ratio",
        "log_dp",
    }

    assert expected_columns.issubset(panel.columns)

    dates = pd.to_datetime(panel["date"])
    assert dates.min() <= pd.Timestamp("1927-06-30")
    assert dates.max() >= pd.Timestamp("1996-12-31")

    assert panel["ret_excess"].notna().all()
    assert panel["log_dp"].notna().all()
    assert panel["dp_ratio"].gt(0).all()