"""
run_analysis.py - end-to-end pipeline (CLI).

    python run_analysis.py                 # uses data/sales_data.csv (or sample fallback)
    python run_analysis.py --data my.csv   # custom file

Outputs: figures in reports/figures/, tables in reports/tables/, and a
plain-text summary in reports/summary.txt.
"""
import argparse
import json
import logging

import pandas as pd

from src import analysis as A, charts, config, forecast as F, rfm as R
from src.data_loader import data_quality_report, load_data


def main():
    ap = argparse.ArgumentParser(description="Retail sales analytics pipeline")
    ap.add_argument("--data", help="path to the sales CSV")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    df = load_data(args.data)
    q = data_quality_report(df)
    k = A.kpis(df)
    tables = config.REPORTS_DIR / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    # --- analysis ---
    yearly = A.yearly_growth(df)
    by_cat, by_reg, by_seg = (A.sales_by(df, c) for c in ("Category", "Region", "Segment"))
    by_sub = A.sales_by(df, "Sub-Category")
    top_prod = A.top_n(df, "Product Name", 10)
    par_c, par_p = A.pareto(df, "Customer ID"), A.pareto(df, "Product ID") if "Product ID" in df else None
    deliv = A.delivery_performance(df)
    t1, t2 = A.test_order_value_by_group(df, "Segment"), A.test_category_region_independence(df)

    rfm = R.compute_rfm(df)
    rfm_sum = R.segment_summary(rfm)
    elbow = R.elbow_scores(rfm)
    _, sil = R.kmeans_segments(rfm)
    fc = F.forecast_sales(df)

    # --- outputs ---
    figs = charts.generate_all(df, rfm_sum, elbow, config.RFM_CLUSTERS, fc)
    for name, t in {"yearly": yearly, "category": by_cat, "region": by_reg, "segment": by_seg,
                    "sub_category": by_sub, "top_products": top_prod, "delivery": deliv,
                    "rfm_segments": rfm_sum, "rfm_customers": rfm, "forecast_metrics": fc["metrics"].reset_index(),
                    "forecast": fc["forecast"].reset_index()}.items():
        t.to_csv(tables / f"{name}.csv", index=False)

    lines = [
        f"DATA SOURCE : {q['source']}",
        f"ROWS        : {q['rows']:,}   ORDERS: {q['unique_orders']:,}   CUSTOMERS: {q['unique_customers']:,}",
        f"PERIOD      : {q['date_min']:%d %b %Y} -> {q['date_max']:%d %b %Y}",
        f"DATE QUALITY: {q['date_suspect_rows']} unresolved, {q['date_ambiguous_rows']} ambiguous rows ({q['date_ambiguous_rows']/q['rows']:.1%})",
        "",
        f"Total sales        : ${k['total_sales']:,.2f}",
        f"Avg order value    : ${k['avg_order_value']:,.2f}",
        f"Avg shipping days  : {k['avg_ship_days']:.2f}",
        f"Top category       : {by_cat.iloc[0]['Category']} ({by_cat.iloc[0]['Share %']}%)",
        f"Top region         : {by_reg.iloc[0]['Region']} ({by_reg.iloc[0]['Share %']}%)",
        f"Top 20% customers  : {par_c['top20_share_pct']}% of sales",
        "",
        "YEARLY GROWTH", yearly.to_string(index=False), "",
        "STATISTICAL TESTS", json.dumps(t1, indent=2, default=str), json.dumps(t2, indent=2, default=str), "",
        "RFM SEGMENTS", rfm_sum.to_string(index=False), f"K-Means (k={config.RFM_CLUSTERS}) silhouette: {sil:.3f}", "",
        f"FORECAST (best model: {fc['best_model']})", fc["metrics"].to_string(), "", fc["forecast"].round(0).to_string(),
    ]
    summary = "\n".join(lines)
    (config.REPORTS_DIR / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"\n{len(figs)} figures -> {config.FIGURES_DIR}\nTables  -> {tables}")


if __name__ == "__main__":
    main()
