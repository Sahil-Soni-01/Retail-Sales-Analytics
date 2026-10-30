# Retail Sales Analytics

An end-to-end **data analytics project** on retail (Superstore-style) sales data: data cleaning with
business-rule validation, descriptive analytics, statistical testing, **RFM customer segmentation
(+ K-Means)**, **time-series forecasting** with a proper back-test, and an interactive **Streamlit dashboard**.

> MCA (AI/ML) project — Chandigarh University · Author: Gaurav Singh

## Problem statement
A retailer (simulated here) wants to know *where sales come from, how they change over time, which customers matter most,
and what to expect next*. The raw export is messy (mixed date formats, duplicates, missing values), so the
project must first make the data trustworthy, then answer those questions with reproducible analysis.

## Objectives
1. Clean and validate the data, handling the mixed `DD/MM/YYYY` / `MM/DD/YYYY` date problem rigorously.
2. Measure performance by time, category, region, product and delivery speed.
3. Test whether observed differences are statistically significant (not just visible in a chart).
4. Segment customers (RFM + K-Means) and quantify the 80/20 (Pareto) effect.
5. Forecast the next 6 months of sales and compare against a baseline.
6. Deliver the findings as static figures, tables and an interactive dashboard.

## What's new vs. a basic EDA project
| Area | Technique |
|---|---|
| Data quality | Row-level date resolution using *ship-window* (0–14 days) and *Order-ID year* rules; ambiguous/unresolved rows flagged, never silently guessed |
| Descriptive | YoY / MoM growth, 3-month rolling mean, seasonality index, Pareto curve, Region×Category heatmap |
| Statistics | Kruskal-Wallis (order value vs. segment), Chi-square (category vs. region independence) |
| Machine learning | RFM scoring, K-Means with elbow + silhouette for choosing *k* |
| Forecasting | Holt-Winters vs. seasonal-naive baseline, 6-month hold-out, MAE / RMSE / MAPE |
| Engineering | Modular `src/` package, pure functions, unit tests (pytest), logging, CLI + dashboard |

## Project structure
```
Retail-Sales-Analytics/
├── app.py                  # Streamlit dashboard (5 tabs)
├── run_analysis.py         # CLI: runs everything, writes figures/tables/summary
├── requirements.txt
├── src/
│   ├── config.py           # paths & constants
│   ├── data_loader.py      # loading, cleaning, date resolution, quality report
│   ├── analysis.py         # KPIs, growth, Pareto, statistical tests
│   ├── rfm.py              # RFM scoring + K-Means
│   ├── forecast.py         # Holt-Winters / seasonal-naive with back-testing
│   └── charts.py           # Matplotlib/Seaborn figures
├── tests/test_pipeline.py  # 14 unit tests
├── tools/make_sample_data.py
├── data/
│   └── sample_sales_data.csv   # synthetic dataset used by the project
├── reports/                # generated: figures/, tables/, summary.txt
└── docs/REPORT.md          # project report template (fill with your real results)
```

## Dataset
This project uses a **synthetic (computer-generated) dataset** that follows the schema of the well-known
*Sample Superstore* retail data. It is produced by `tools/make_sample_data.py` (fixed random seed, so it is
reproducible) and stored in `data/sample_sales_data.csv` (~5,500 rows, 2015-2018).

**Why synthetic?** A public file with the required columns (order dates, customer IDs, product names) was not
used; generating the data makes the project fully reproducible and lets known data-quality problems be
injected deliberately - mixed `DD/MM/YYYY` / `MM/DD/YYYY` dates, exact duplicate rows and missing postal codes -
so that the cleaning pipeline can be demonstrated and tested.

**Important:** all figures, segments and forecasts are properties of the *generated* data, not of a real
business. Treat the results as a demonstration of the methodology.

**Using real data instead:** save any CSV with the required columns as `data/sales_data.csv`; it takes
priority automatically (or pass `--data path.csv`, or set `RETAIL_DATA_PATH`).
Required columns: `Order ID, Order Date, Ship Date, Ship Mode, Customer ID, Segment, Region, Category,
Sub-Category, Product Name, Sales` (optional: `Product ID`, `State`, `City`, ...).

## How to run
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run_analysis.py        # figures -> reports/figures, tables -> reports/tables, summary -> reports/summary.txt
streamlit run app.py          # interactive dashboard
pytest                        # run the unit tests
```

## Method notes

**Date resolution.** For each row, all four (order-format, ship-format) readings are tried; the first
where `0 ≤ Ship − Order ≤ 14 days` *and* the order year equals the year in `Order ID` wins. If several
readings pass (e.g. same-day shipping on `05/06/2017`) the row is flagged `Date_Ambiguous` and gets the
day-first reading; if none pass it is `Date_Suspect` and is excluded from time-based analysis (its Sales
are still used for non-date analysis). The dashboard has a toggle to exclude ambiguous rows as a
sensitivity check.

**Forecasting.** The last 6 months are held out. Holt-Winters (additive trend + seasonality) is compared
with a seasonal-naive baseline ("same month last year"); the lower-MAE model is refit on all data to
forecast ahead. A model that cannot beat the baseline is not worth using — the output says which won.

**Statistics.** Order values are right-skewed, so a non-parametric test (Kruskal-Wallis) is used instead of ANOVA.

## Limitations & future work
- Only `Sales` is analysed (the public file has no Profit/Quantity). With `Profit`/`Discount`, add margin analysis.
- ~4 years of monthly data limits forecast reliability; prediction intervals and ARIMA/Prophet would be a next step.
- RFM snapshot is the last order date in the data; results change with the chosen period.
- Ambiguous dates cannot be resolved from the data alone; a source-system fix is the real solution.

## Tech stack
Python · pandas · NumPy · SciPy · scikit-learn · statsmodels · Matplotlib · Seaborn · Plotly · Streamlit · pytest
