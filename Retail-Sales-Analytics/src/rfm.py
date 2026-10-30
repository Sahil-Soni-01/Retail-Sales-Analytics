"""
rfm.py
------
Customer segmentation using RFM analysis (Recency, Frequency, Monetary) plus
K-Means clustering as a data-driven comparison.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from . import config
from .data_loader import dated


def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """One row per customer with Recency (days), Frequency (orders), Monetary (sales)."""
    d = dated(df)
    snapshot = d["Order Date"].max() + pd.Timedelta(days=1)
    rfm = d.groupby("Customer ID").agg(
        Recency=("Order Date", lambda s: (snapshot - s.max()).days),
        Frequency=("Order ID", "nunique"),
        Monetary=("Sales", "sum"),
    )
    # Rank-based quintiles: robust when many customers tie on Frequency.
    rfm["R"] = pd.qcut(rfm["Recency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["F"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M"] = pd.qcut(rfm["Monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["RFM Score"] = rfm["R"] + rfm["F"] + rfm["M"]
    rfm["Label"] = rfm.apply(_label, axis=1)
    return rfm.reset_index()


def _label(row) -> str:
    r, f, m = row["R"], row["F"], row["M"]
    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    if f >= 4 and r >= 3:
        return "Loyal"
    if r >= 4 and f <= 2:
        return "New / Promising"
    if r <= 2 and (f >= 4 or m >= 4):
        return "At Risk"
    if r <= 2:
        return "Lost / Hibernating"
    return "Needs Attention"


def segment_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    s = rfm.groupby("Label").agg(
        Customers=("Customer ID", "count"),
        Avg_Recency=("Recency", "mean"),
        Avg_Frequency=("Frequency", "mean"),
        Avg_Monetary=("Monetary", "mean"),
        Total_Sales=("Monetary", "sum"),
    ).round(1)
    s["Sales Share %"] = (s["Total_Sales"] / s["Total_Sales"].sum() * 100).round(1)
    return s.sort_values("Total_Sales", ascending=False).reset_index()


def kmeans_segments(rfm: pd.DataFrame, k: int = config.RFM_CLUSTERS) -> tuple[pd.DataFrame, float]:
    """
    Cluster customers with K-Means on log-scaled, standardised R/F/M.
    (log1p tames the heavy right skew of Frequency/Monetary.)
    Returns the DataFrame with a 'Cluster' column and the silhouette score.
    """
    X = np.log1p(rfm[["Recency", "Frequency", "Monetary"]])
    X = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=k, n_init=10, random_state=config.RANDOM_STATE).fit(X)
    out = rfm.copy()
    out["Cluster"] = km.labels_
    return out, float(silhouette_score(X, km.labels_))


def elbow_scores(rfm: pd.DataFrame, k_range=range(2, 9)) -> pd.DataFrame:
    """Inertia and silhouette for each k, to justify the chosen number of clusters."""
    X = StandardScaler().fit_transform(np.log1p(rfm[["Recency", "Frequency", "Monetary"]]))
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=config.RANDOM_STATE).fit(X)
        rows.append({"k": k, "Inertia": km.inertia_, "Silhouette": silhouette_score(X, km.labels_)})
    return pd.DataFrame(rows)
