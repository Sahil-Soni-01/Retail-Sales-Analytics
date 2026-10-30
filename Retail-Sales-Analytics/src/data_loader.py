"""
data_loader.py
--------------
Loading, validating and cleaning the retail sales dataset.

Key data-quality problem handled here
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Order/Ship dates arrive in mixed formats (DD/MM/YYYY and MM/DD/YYYY) within
the same column. A date such as 05/06/2017 is ambiguous on its own, so instead
of guessing one format globally we use *business rules* as evidence:

  1. Ship Date must be on/after Order Date and within MAX_SHIP_DAYS of it.
  2. The year embedded in Order ID (e.g. CA-2017-152156) must match the
     Order Date year (when the ID follows that pattern).

Some dates stay *truly* ambiguous (e.g. 05/06/2017 shipped the same day:
both readings pass every rule). These are given the day-first reading and
flagged `Date_Ambiguous`, so analyses can test sensitivity by excluding them.

For every row we test each (order-format, ship-format) combination and keep
the first one that satisfies the rules. Rows with no valid reading are flagged
(`Date_Suspect`) and excluded from time-based analysis, but kept for
non-date analysis because their Sales values are still valid.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from . import config

log = logging.getLogger(__name__)

DAY_FIRST = "%d/%m/%Y"
MONTH_FIRST = "%m/%d/%Y"
# Priority order of (order format, ship format) readings.
_COMBOS = [
    (DAY_FIRST, DAY_FIRST),
    (MONTH_FIRST, MONTH_FIRST),
    (DAY_FIRST, MONTH_FIRST),
    (MONTH_FIRST, DAY_FIRST),
]


def _parse(series: pd.Series, fmt: str) -> pd.Series:
    return pd.to_datetime(series.astype("string").str.strip(), format=fmt, errors="coerce")


def resolve_dates(order_raw: pd.Series, ship_raw: pd.Series,
                  order_id: pd.Series | None = None,
                  max_ship_days: int = config.MAX_SHIP_DAYS) -> pd.DataFrame:
    """
    Resolve mixed-format dates using business-rule validation.

    Returns a DataFrame (same index) with columns:
      Order Date, Ship Date  - parsed datetimes (NaT if unresolved)
      Date_Ambiguous         - True if more than one reading satisfied the rules
      Date_Suspect           - True if NO reading satisfied the rules
    """
    id_year = None
    if order_id is not None:
        id_year = pd.to_numeric(
            order_id.astype("string").str.extract(r"-(\d{4})-")[0], errors="coerce"
        )

    n = len(order_raw)
    order_out = pd.Series(pd.NaT, index=order_raw.index, dtype="datetime64[ns]")
    ship_out = pd.Series(pd.NaT, index=order_raw.index, dtype="datetime64[ns]")
    resolved = pd.Series(False, index=order_raw.index)
    valid_count = pd.Series(0, index=order_raw.index)

    for o_fmt, s_fmt in _COMBOS:
        o, s = _parse(order_raw, o_fmt), _parse(ship_raw, s_fmt)
        gap = (s - o).dt.days
        ok = gap.between(0, max_ship_days)
        if id_year is not None:
            ok &= id_year.isna() | (o.dt.year == id_year)
        ok = ok.fillna(False)

        valid_count += ok.astype(int)
        take = ok & ~resolved
        order_out[take] = o[take]
        ship_out[take] = s[take]
        resolved |= take

    out = pd.DataFrame({"Order Date": order_out, "Ship Date": ship_out})
    out["Date_Ambiguous"] = valid_count > 1
    out["Date_Suspect"] = ~resolved

    log.info("Dates resolved for %d/%d rows (%d ambiguous, %d suspect)",
             int(resolved.sum()), n, int(out["Date_Ambiguous"].sum()), int(out["Date_Suspect"].sum()))
    return out


def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in config.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning steps to a raw DataFrame and return a new one."""
    validate_schema(raw)
    df = raw.copy()
    df.columns = df.columns.str.strip()

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    log.info("Removed %d exact duplicate rows", before - len(df))

    # Sales must be numeric and non-negative
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    bad_sales = df["Sales"].isna() | (df["Sales"] < 0)
    if bad_sales.any():
        log.warning("Dropping %d rows with invalid Sales", int(bad_sales.sum()))
        df = df.loc[~bad_sales].reset_index(drop=True)

    # Dates
    dates = resolve_dates(df["Order Date"], df["Ship Date"], df["Order ID"])
    for col in dates.columns:
        df[col] = dates[col].astype(bool) if col.startswith("Date_") else dates[col]

    df["Ship Days"] = (df["Ship Date"] - df["Order Date"]).dt.days
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.to_period("M").dt.to_timestamp()

    # Tidy text columns & optional columns
    for col in ["Ship Mode", "Segment", "Region", "Category", "Sub-Category"]:
        df[col] = df[col].astype("string").str.strip().astype("category")
    if "Postal Code" in df.columns:
        df["Postal Code"] = df["Postal Code"].astype("string")  # an identifier, not a number

    return df


def load_data(filepath: str | Path | None = None) -> pd.DataFrame:
    """Load and clean the dataset. Falls back to the bundled synthetic sample."""
    path = Path(filepath) if filepath else config.DATA_PATH
    if not path.exists():
        if config.SAMPLE_DATA_PATH.exists():
            log.info("%s not found - using the bundled synthetic dataset.", path)
            path = config.SAMPLE_DATA_PATH
        else:
            raise FileNotFoundError(
                f"Dataset not found at {path}. See README.md ('Dataset') for setup."
            )
    raw = pd.read_csv(path, encoding="latin1", dtype=str)
    # read as str so date strings are untouched; Sales is converted in clean_data
    df = clean_data(raw)
    df.attrs["source"] = str(path)
    return df


def data_quality_report(df: pd.DataFrame) -> dict:
    """Summary of data-quality metrics, used by the CLI and the dashboard."""
    n = len(df)
    missing = df.isna().sum()
    return {
        "rows": n,
        "columns": df.shape[1],
        "missing_by_column": missing[missing > 0].to_dict(),
        "date_suspect_rows": int(df["Date_Suspect"].sum()),
        "date_suspect_pct": round(df["Date_Suspect"].mean() * 100, 2),
        "date_ambiguous_rows": int(df["Date_Ambiguous"].sum()),
        "date_min": df["Order Date"].min(),
        "date_max": df["Order Date"].max(),
        "unique_customers": int(df["Customer ID"].nunique()),
        "unique_orders": int(df["Order ID"].nunique()),
        "source": df.attrs.get("source", "?"),
    }


def dated(df: pd.DataFrame, exclude_ambiguous: bool = False) -> pd.DataFrame:
    """Rows with a usable order date (use for any time-based analysis).

    exclude_ambiguous=True additionally drops rows whose day/month order could
    not be determined - handy as a sensitivity check.
    """
    keep = ~df["Date_Suspect"]
    if exclude_ambiguous:
        keep &= ~df["Date_Ambiguous"]
    return df.loc[keep]
