"""
Pull the CRSP monthly stock-market index from WRDS for the Stambaugh (1999)
"Predictive Regressions" replication.

We pull ``crsp_a_indexes.msi``: the CRSP value-weighted market index across
NYSE/AMEX/NASDAQ. Two return columns matter:

* ``vwretd`` -- value-weighted return INCLUDING dividends
* ``vwretx`` -- value-weighted return EXCLUDING dividends (price only)

Their difference, vwretd - vwretx, is the month's dividend expressed as a
fraction of last month's price. That lets us reconstruct the dividend-price
ratio (the paper's predictor) without a separate dividend file. ``totval``
(total market value) is kept for summary statistics.

Note: Stambaugh (1999) uses a NYSE-only value-weighted index. We use the CRSP
total-market VW index; both are value-weighted, so the largest (mostly NYSE)
firms dominate both, and the series are nearly identical. A NYSE-only rebuild
from the stock file is provided separately as a robustness check.
"""

from pathlib import Path

import pandas as pd
import wrds

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
WRDS_USERNAME = config("WRDS_USERNAME")
START_DATE = config("START_DATE")
END_DATE = config("END_DATE")


def pull_CRSP_index(
    start_date=START_DATE, end_date=END_DATE, wrds_username=WRDS_USERNAME
):
    """Pull monthly value-weighted market returns (with and without dividends)
    and total market value from crsp_a_indexes.msi."""
    query = f"""
        SELECT date, vwretd, vwretx, totval
        FROM crsp.msi
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY date
    """
    db = wrds.Connection(wrds_username=wrds_username)
    df = db.raw_sql(query, date_cols=["date"])
    db.close()
    return df


def load_CRSP_index(data_dir=DATA_DIR):
    """Load the previously pulled CRSP index file from disk."""
    path = Path(data_dir) / "crsp_msi.parquet"
    return pd.read_parquet(path)


if __name__ == "__main__":
    df_msi = pull_CRSP_index(start_date=START_DATE, end_date=END_DATE)
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    df_msi.to_parquet(Path(DATA_DIR) / "crsp_msi.parquet")
    print(f"Saved {len(df_msi)} rows: {df_msi['date'].min()} to {df_msi['date'].max()}")