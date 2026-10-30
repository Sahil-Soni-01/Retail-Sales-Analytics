"""
analysis.py
-----------
Descriptive and statistical analysis. Every function takes a cleaned
DataFrame and returns a DataFrame / Series / dict (no printing, no plotting),
which makes them easy to unit-test and reuse from the CLI and the dashboard.
"""
from __future__ import annotations

import pandas as pd
from scipy import stats

from .data_loader import dated


def kpis(df: pd.DataFrame) -> dict:
    orders = df["Order ID"].nunique()
    total = df["Sales"].sum()
    return {
        "total_sales": total,
        "orders": orders,
        "customers": df["Customer ID"].nunique(),
        "avg_order_value": total / orders if orders else 0.0,
        "avg_ship_days": df["Ship Days"].mean(),
    }


def sales_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Sales, order count and share of total for any categorical column."""
    g = (df.groupby(col, observed=True)
           .agg(Sales=("Sales", "sum"), Orders=("Order ID", "nunique"))
           .sort_values("Sales", ascending=False))
    g["Share %"] = (g["Sales"] / g["Sales"].sum() * 100).round(2)
    return g.reset_index()


def monthly_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly sales with month-over-month growth. Uses only trustworthy dates."""
    m = (dated(df).groupby("Month")["Sales"].sum().rename("Sales").to_frame())
    m = m.asfreq("MS", fill_value=0.0)          # make sure no month is skipped
    m["MoM %"] = (m["Sales"].pct_change() * 100).round(2)
    m["Rolling 3M"] = m["Sales"].rolling(3).mean()
    return m.reset_index()


def yearly_growth(df: pd.DataFrame) -> pd.DataFrame:
    y = dated(df).groupby("Year")["Sales"].sum().to_frame("Sales")
    y["YoY %"] = (y["Sales"].pct_change() * 100).round(2)
    return y.reset_index()


def seasonality_index(df: pd.DataFrame) -> pd.DataFrame:
    """Average sales of each calendar month relative to the overall monthly mean (1.0 = average)."""
    m = monthly_sales(df)
    m["MonthNo"] = m["Month"].dt.month
    idx = m.groupby("MonthNo")["Sales"].mean() / m["Sales"].mean()
    return idx.round(3).rename("Index").reset_index()


def top_n(df: pd.DataFrame, col: str, n: int = 10) -> pd.DataFrame:
    return sales_by(df, col).head(n)


def pareto(df: pd.DataFrame, col: str = "Customer ID") -> dict:
    """Pareto (80/20) analysis: what share of sales comes from the top 20% of entities?"""
    s = df.groupby(col)["Sales"].sum().sort_values(ascending=False)
    cum = s.cumsum() / s.sum()
    top20 = max(int(round(len(s) * 0.2)), 1)
    return {
        "entities": len(s),
        "top20_share_pct": round(cum.iloc[top20 - 1] * 100, 1),
        "entities_for_80pct": int((cum < 0.8).sum() + 1),
        "curve": cum.reset_index(drop=True),
    }


def delivery_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Average / median / max shipping days per ship mode (trustworthy dates only)."""
    return (dated(df).groupby("Ship Mode", observed=True)["Ship Days"]
            .agg(["count", "mean", "median", "max"]).round(2).reset_index())


def category_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Region x Category sales matrix (for heatmaps)."""
    return df.pivot_table(index="Region", columns="Category", values="Sales",
                          aggfunc="sum", observed=True).round(0)


# ---- Statistical tests -----------------------------------------------------
def test_order_value_by_group(df: pd.DataFrame, group_col: str = "Segment") -> dict:
    """
    Kruskal-Wallis H-test: does the typical order value differ between groups?
    (Non-parametric - order values are right-skewed, so ANOVA's normality
    assumption would not hold.)
    """
    ov = df.groupby(["Order ID", group_col], observed=True)["Sales"].sum().reset_index()
    groups = [g["Sales"].values for _, g in ov.groupby(group_col, observed=True) if len(g) > 1]
    if len(groups) < 2:
        return {"test": "Kruskal-Wallis", "note": "Not enough groups"}
    h, p = stats.kruskal(*groups)
    return {
        "test": "Kruskal-Wallis H",
        "group_col": group_col,
        "statistic": round(float(h), 3),
        "p_value": float(p),
        "significant_at_5pct": bool(p < 0.05),
        "medians": ov.groupby(group_col, observed=True)["Sales"].median().round(2).to_dict(),
    }


def test_category_region_independence(df: pd.DataFrame) -> dict:
    """Chi-square test: is the number of orders per category independent of region?"""
    ct = pd.crosstab(df["Region"], df["Category"])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    return {"test": "Chi-square independence", "chi2": round(float(chi2), 3), "dof": int(dof),
            "p_value": float(p), "significant_at_5pct": bool(p < 0.05)}
