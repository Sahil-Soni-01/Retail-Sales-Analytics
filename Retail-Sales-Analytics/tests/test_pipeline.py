import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src import analysis as A, forecast as F, rfm as R
from src.data_loader import clean_data, resolve_dates


def _s(*v):
    return pd.Series(list(v))


# ---------- date resolution ----------
def test_unambiguous_day_first():
    d = resolve_dates(_s("25/03/2017"), _s("28/03/2017"))
    assert d.loc[0, "Order Date"] == pd.Timestamp("2017-03-25")
    assert d.loc[0, "Ship Date"] == pd.Timestamp("2017-03-28")
    assert not d.loc[0, "Date_Suspect"]


def test_unambiguous_month_first():
    d = resolve_dates(_s("03/25/2017"), _s("03/28/2017"))
    assert d.loc[0, "Order Date"] == pd.Timestamp("2017-03-25")


def test_ship_window_disambiguates():
    # 05/06 -> 08/06: day-first = 3 days; month-first = 92 days (invalid)
    d = resolve_dates(_s("05/06/2017"), _s("08/06/2017"))
    assert d.loc[0, "Order Date"] == pd.Timestamp("2017-06-05")
    assert not d.loc[0, "Date_Ambiguous"]


def test_truly_ambiguous_is_flagged():
    d = resolve_dates(_s("05/06/2017"), _s("05/06/2017"))   # same-day shipping
    assert d.loc[0, "Date_Ambiguous"] and not d.loc[0, "Date_Suspect"]


def test_order_id_year_used_as_evidence():
    # 05/06/2016 vs ID year 2017 -> invalid both ways -> suspect
    d = resolve_dates(_s("05/06/2016"), _s("07/06/2016"), order_id=_s("CA-2017-100001"))
    assert d.loc[0, "Date_Suspect"]


def test_impossible_dates_are_suspect():
    d = resolve_dates(_s("31/02/2017", "garbage"), _s("02/03/2017", "x"))
    assert d["Date_Suspect"].all()


# ---------- cleaning / analysis ----------
@pytest.fixture
def small_df():
    rng = np.random.default_rng(0)
    n = 300
    od = pd.to_datetime("2016-01-01") + pd.to_timedelta(rng.integers(0, 1000, n), unit="D")
    raw = pd.DataFrame({
        "Order ID": [f"CA-{d.year}-{i}" for i, d in enumerate(od)],
        "Order Date": od.strftime("%d/%m/%Y"),
        "Ship Date": (od + pd.Timedelta(days=3)).strftime("%d/%m/%Y"),
        "Ship Mode": "Standard Class",
        "Customer ID": [f"C{i % 40}" for i in range(n)],
        "Segment": rng.choice(["Consumer", "Corporate"], n),
        "Region": rng.choice(["East", "West"], n),
        "Category": rng.choice(["Furniture", "Technology"], n),
        "Sub-Category": "X", "Product Name": "P", "Sales": rng.uniform(5, 500, n).round(2),
    })
    return clean_data(pd.concat([raw, raw.iloc[:5]]))   # 5 exact duplicates


def test_duplicates_removed(small_df):
    assert len(small_df) == 300


def test_missing_column_rejected():
    with pytest.raises(ValueError):
        clean_data(pd.DataFrame({"Sales": [1]}))


def test_kpis_consistent(small_df):
    k = A.kpis(small_df)
    assert k["total_sales"] == pytest.approx(small_df["Sales"].sum())
    assert k["avg_order_value"] == pytest.approx(k["total_sales"] / small_df["Order ID"].nunique())


def test_sales_by_shares_sum_to_100(small_df):
    assert A.sales_by(small_df, "Region")["Share %"].sum() == pytest.approx(100, abs=0.1)


def test_monthly_has_no_gaps(small_df):
    m = A.monthly_sales(small_df)
    assert (m["Month"].diff().dropna().dt.days.between(28, 31)).all()


def test_rfm_scores_in_range(small_df):
    r = R.compute_rfm(small_df)
    assert r[["R", "F", "M"]].min().min() >= 1 and r[["R", "F", "M"]].max().max() <= 5
    assert len(r) == small_df["Customer ID"].nunique()


def test_forecast_shape(small_df):
    # 300 rows over ~33 months is enough for 24+6
    res = F.forecast_sales(small_df, horizon=3, holdout=3)
    assert len(res["forecast"]) == 3 and (res["forecast"] >= 0).all()


def test_pareto_bounds(small_df):
    p = A.pareto(small_df, "Customer ID")
    assert 20 <= p["top20_share_pct"] <= 100
