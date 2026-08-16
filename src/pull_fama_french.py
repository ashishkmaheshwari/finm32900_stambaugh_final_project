"""
Pull the monthly Fama-French factors from WRDS (ff.factors_monthly).

We need only ``rf`` -- the one-month Treasury-bill rate -- to form EXCESS
market returns (vwretd - rf), which are the dependent variable in Stambaugh's
predictive regression. We keep ``mktrf`` as a free cross-check: it should be
close to our vwretd - rf. Same series used in the Fama-French homework.
"""

from pathlib import Path

import pandas as pd
import wrds

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
WRDS_USERNAME = config("WRDS_USERNAME")
START_DATE = config("START_DATE")
END_DATE = config("END_DATE")


def pull_fama_french_factors(
    start_date=START_DATE, end_date=END_DATE, wrds_username=WRDS_USERNAME
):
    """Pull monthly FF factors; rf is the column we need."""
    query = f"""
        SELECT date, mktrf, smb, hml, rf
        FROM ff.factors_monthly
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY date
    """
    db = wrds.Connection(wrds_username=wrds_username)
    df = db.raw_sql(query, date_cols=["date"])
    db.close()
    return df


def load_fama_french_factors(data_dir=DATA_DIR):
    """Load the previously pulled FF factor file from disk."""
    path = Path(data_dir) / "ff_factors.parquet"
    return pd.read_parquet(path)


if __name__ == "__main__":
    df_ff = pull_fama_french_factors(start_date=START_DATE, end_date=END_DATE)
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    df_ff.to_parquet(Path(DATA_DIR) / "ff_factors.parquet")
    print(f"Saved {len(df_ff)} rows: {df_ff['date'].min()} to {df_ff['date'].max()}")