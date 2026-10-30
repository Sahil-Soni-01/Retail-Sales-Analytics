"""charts.py - static Matplotlib/Seaborn figures saved to reports/figures/."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns

from . import analysis as A
from . import config

PALETTE = ["#2a6f97", "#e07a5f", "#81b29a", "#f2cc8f", "#6d597a", "#b56576"]
sns.set_theme(style="whitegrid", palette=PALETTE)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "figure.dpi": 100})
USD = mtick.FuncFormatter(lambda v, _: f"${v/1e6:.1f}M" if abs(v) >= 1e6 else f"${v/1e3:.0f}K")


def _save(fig, name: str, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def monthly_trend(df, out=config.FIGURES_DIR):
    m = A.monthly_sales(df)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(m["Month"], m["Sales"], color=PALETTE[0], alpha=.45, label="Monthly sales")
    ax.plot(m["Month"], m["Rolling 3M"], color=PALETTE[1], lw=2.4, label="3-month average")
    ax.yaxis.set_major_formatter(USD)
    span = f"{m['Month'].min():%b %Y} - {m['Month'].max():%b %Y}"
    ax.set(title=f"Monthly Sales Trend ({span})", xlabel="", ylabel="Sales")
    ax.legend(frameon=False)
    return _save(fig, "01_monthly_trend.png", out)


def yearly_growth(df, out=config.FIGURES_DIR):
    y = A.yearly_growth(df)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.bar(y["Year"].astype(str), y["Sales"], color=PALETTE[0])
    for b, g in zip(bars, y["YoY %"]):
        if pd.notna(g):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{g:+.1f}%", ha="center", va="bottom", fontsize=9)
    ax.yaxis.set_major_formatter(USD)
    ax.set(title="Yearly Sales and YoY Growth", xlabel="", ylabel="Sales")
    return _save(fig, "02_yearly_growth.png", out)


def seasonality(df, out=config.FIGURES_DIR):
    s = A.seasonality_index(df)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    colors = [PALETTE[1] if v > 1 else PALETTE[0] for v in s["Index"]]
    ax.bar(s["MonthNo"], s["Index"], color=colors)
    ax.axhline(1, color="grey", ls="--", lw=1)
    ax.set_xticks(range(1, 13), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set(title="Seasonality Index (1.0 = average month)", xlabel="", ylabel="Index")
    return _save(fig, "03_seasonality.png", out)


def breakdown(df, col, filename, title, out=config.FIGURES_DIR, top=None):
    d = A.sales_by(df, col)
    if top:
        d = d.head(top)
    d = d.sort_values("Sales")
    fig, ax = plt.subplots(figsize=(8, max(3, .45 * len(d) + 1.5)))
    ax.barh(d[col].astype(str), d["Sales"], color=PALETTE[0])
    for y_, (v, p) in enumerate(zip(d["Sales"], d["Share %"])):
        ax.text(v, y_, f"  {p:.1f}%", va="center", fontsize=9)
    ax.xaxis.set_major_formatter(USD)
    ax.set(title=title, xlabel="Sales", ylabel="")
    ax.margins(x=.12)
    return _save(fig, filename, out)


def category_region_heatmap(df, out=config.FIGURES_DIR):
    m = A.category_matrix(df)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(m, annot=True, fmt=",.0f", cmap="Blues", cbar=False, ax=ax)
    ax.set(title="Sales: Region x Category ($)", xlabel="", ylabel="")
    return _save(fig, "07_region_category_heatmap.png", out)


def pareto_curve(df, out=config.FIGURES_DIR):
    p = A.pareto(df, "Customer ID")
    x = (pd.Series(range(1, len(p["curve"]) + 1)) / len(p["curve"])) * 100
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(x, p["curve"] * 100, color=PALETTE[0], lw=2.4)
    ax.axvline(20, color="grey", ls=":"); ax.axhline(p["top20_share_pct"], color="grey", ls=":")
    ax.annotate(f"Top 20% of customers\n= {p['top20_share_pct']}% of sales", (20, p["top20_share_pct"]),
                xytext=(35, p["top20_share_pct"] - 22), arrowprops=dict(arrowstyle="->"))
    ax.set(title="Customer Pareto Curve", xlabel="% of customers (ranked by sales)", ylabel="Cumulative % of sales")
    return _save(fig, "08_customer_pareto.png", out)


def delivery_boxplot(df, out=config.FIGURES_DIR):
    d = A.dated(df)
    order = d.groupby("Ship Mode", observed=True)["Ship Days"].median().sort_values().index
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sns.boxplot(data=d, x="Ship Mode", y="Ship Days", order=order, color=PALETTE[2], ax=ax)
    ax.set(title="Delivery Time by Ship Mode", xlabel="", ylabel="Days to ship")
    return _save(fig, "09_delivery_time.png", out)


def rfm_segments(summary: pd.DataFrame, out=config.FIGURES_DIR):
    d = summary.sort_values("Total_Sales")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(d["Label"], d["Total_Sales"], color=PALETTE[4])
    for y_, (v, c) in enumerate(zip(d["Total_Sales"], d["Customers"])):
        ax.text(v, y_, f"  {c} customers", va="center", fontsize=9)
    ax.xaxis.set_major_formatter(USD); ax.margins(x=.2)
    ax.set(title="Sales by RFM Customer Segment", xlabel="Sales", ylabel="")
    return _save(fig, "10_rfm_segments.png", out)


def rfm_elbow(scores: pd.DataFrame, chosen_k: int, out=config.FIGURES_DIR):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.plot(scores["k"], scores["Inertia"], marker="o", color=PALETTE[0]); a1.set(title="Elbow Method", xlabel="k", ylabel="Inertia")
    a2.plot(scores["k"], scores["Silhouette"], marker="o", color=PALETTE[1]); a2.set(title="Silhouette Score", xlabel="k", ylabel="Score")
    for a in (a1, a2):
        a.axvline(chosen_k, color="grey", ls="--", lw=1)
    return _save(fig, "11_kmeans_k_selection.png", out)


def forecast_chart(res: dict, out=config.FIGURES_DIR):
    h, bt, fc = res["history"], res["backtest"], res["forecast"]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(h.index, h.values, color=PALETTE[0], label="Actual")
    ax.plot(bt.index, bt[res["best_model"]], color=PALETTE[2], ls="--", lw=2, label=f"Backtest ({res['best_model']})")
    link = pd.concat([h.iloc[[-1]], fc])
    ax.plot(link.index, link.values, color=PALETTE[1], lw=2.4, marker="o", label="Forecast")
    ax.yaxis.set_major_formatter(USD)
    ax.set(title=f"Sales Forecast - next {len(fc)} months", xlabel="", ylabel="Sales")
    ax.legend(frameon=False)
    return _save(fig, "12_forecast.png", out)


def generate_all(df, rfm_summary, elbow, k, forecast_result, out=config.FIGURES_DIR) -> list[Path]:
    return [
        monthly_trend(df, out), yearly_growth(df, out), seasonality(df, out),
        breakdown(df, "Category", "04_category_sales.png", "Sales by Category", out),
        breakdown(df, "Region", "05_region_sales.png", "Sales by Region", out),
        breakdown(df, "Product Name", "06_top_products.png", "Top 10 Products by Sales", out, top=10),
        category_region_heatmap(df, out), pareto_curve(df, out), delivery_boxplot(df, out),
        rfm_segments(rfm_summary, out), rfm_elbow(elbow, k, out), forecast_chart(forecast_result, out),
    ]
