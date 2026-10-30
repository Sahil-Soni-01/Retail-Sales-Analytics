"""
app.py - interactive Streamlit dashboard.

Run:  streamlit run app.py
"""
import pandas as pd
import plotly.express as px
import streamlit as st

from src import analysis as A, config, forecast as F, rfm as R
from src.data_loader import data_quality_report, dated, load_data

st.set_page_config(page_title="Retail Sales Analytics", page_icon="📊", layout="wide")


@st.cache_data(show_spinner="Loading and cleaning data...")
def get_data() -> pd.DataFrame:
    return load_data()


df_all = get_data()
if "sample_sales_data" in df_all.attrs.get("source", ""):
    st.info("Dataset: **synthetic data** generated for this project (see README). "
            "Results demonstrate the methodology, not a real business.", icon="ℹ️")

# ---------------- Sidebar filters ----------------
st.sidebar.header("Filters")


def multi(label, col):
    opts = sorted(df_all[col].dropna().astype(str).unique())
    return st.sidebar.multiselect(label, opts, default=opts)


cats, regs, segs = multi("Category", "Category"), multi("Region", "Region"), multi("Segment", "Segment")
valid = dated(df_all)
lo, hi = valid["Order Date"].min().date(), valid["Order Date"].max().date()
date_range = st.sidebar.date_input("Order date range", (lo, hi), min_value=lo, max_value=hi)
drop_amb = st.sidebar.checkbox("Exclude rows with ambiguous dates", value=False,
                               help="Rows where DD/MM vs MM/DD could not be determined. Use to test sensitivity.")

df = df_all[df_all["Category"].astype(str).isin(cats) & df_all["Region"].astype(str).isin(regs)
            & df_all["Segment"].astype(str).isin(segs)]
df = df[~df["Date_Suspect"]]
if drop_amb:
    df = df[~df["Date_Ambiguous"]]
if len(date_range) == 2:
    df = df[(df["Order Date"] >= pd.Timestamp(date_range[0])) & (df["Order Date"] <= pd.Timestamp(date_range[1]))]
if df.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

st.title("📊 Retail Sales Analytics Dashboard")
k = A.kpis(df)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total sales", f"${k['total_sales']:,.0f}")
c2.metric("Orders", f"{k['orders']:,}")
c3.metric("Customers", f"{k['customers']:,}")
c4.metric("Avg order value", f"${k['avg_order_value']:,.0f}")
c5.metric("Avg ship days", f"{k['avg_ship_days']:.1f}")

tab_over, tab_prod, tab_cust, tab_fc, tab_dq = st.tabs(
    ["Overview", "Products & Geography", "Customer segments (RFM)", "Forecast", "Data quality"])

# ---------------- Overview ----------------
with tab_over:
    m = A.monthly_sales(df)
    st.subheader("Monthly sales")
    fig = px.line(m, x="Month", y=["Sales", "Rolling 3M"], markers=False)
    fig.update_layout(legend_title_text="", yaxis_title="Sales ($)", xaxis_title="")
    st.plotly_chart(fig, width="stretch")

    a, b = st.columns(2)
    with a:
        st.subheader("Year-over-year growth")
        y = A.yearly_growth(df)
        fig = px.bar(y, x="Year", y="Sales", text=y["YoY %"].map(lambda v: "" if pd.isna(v) else f"{v:+.1f}%"))
        fig.update_xaxes(type="category")
        st.plotly_chart(fig, width="stretch")
    with b:
        st.subheader("Seasonality index")
        s = A.seasonality_index(df)
        fig = px.bar(s, x="MonthNo", y="Index")
        fig.add_hline(y=1, line_dash="dash")
        fig.update_xaxes(dtick=1, title="Month")
        st.plotly_chart(fig, width="stretch")

# ---------------- Products & geography ----------------
with tab_prod:
    a, b = st.columns(2)
    with a:
        st.subheader("Sales by category")
        d = A.sales_by(df, "Category")
        st.plotly_chart(px.bar(d, x="Category", y="Sales", color="Category", text="Share %"), width="stretch")
    with b:
        st.subheader("Sales by region")
        d = A.sales_by(df, "Region")
        st.plotly_chart(px.bar(d, x="Region", y="Sales", color="Region", text="Share %"), width="stretch")

    st.subheader("Region × Category heatmap")
    st.plotly_chart(px.imshow(A.category_matrix(df), text_auto=",.0f", aspect="auto",
                              color_continuous_scale="Blues"), width="stretch")

    st.subheader("Top 10 products")
    tp = A.top_n(df, "Product Name", 10)
    fig = px.bar(tp, x="Sales", y="Product Name", orientation="h")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, width="stretch")

    a, b = st.columns(2)
    with a:
        st.subheader("Sub-category sales")
        st.dataframe(A.sales_by(df, "Sub-Category"), hide_index=True, width="stretch")
    with b:
        st.subheader("Delivery performance")
        st.dataframe(A.delivery_performance(df), hide_index=True, width="stretch")

# ---------------- RFM ----------------
with tab_cust:
    if df["Customer ID"].nunique() < 20:
        st.info("Select more data (at least 20 customers) to run customer segmentation.")
    else:
        rfm = R.compute_rfm(df)
        summ = R.segment_summary(rfm)
        p = A.pareto(df, "Customer ID")
        st.metric("Top 20% of customers generate", f"{p['top20_share_pct']}% of sales")
        a, b = st.columns(2)
        with a:
            st.subheader("Sales by RFM segment")
            st.plotly_chart(px.bar(summ, x="Total_Sales", y="Label", orientation="h", text="Customers"),
                            width="stretch")
        with b:
            st.subheader("Segment profile")
            st.dataframe(summ, hide_index=True, width="stretch")
        st.subheader("Frequency vs Monetary (each dot = a customer)")
        st.plotly_chart(px.scatter(rfm, x="Frequency", y="Monetary", color="Label", size="Monetary",
                                   hover_data=["Customer ID", "Recency"], log_y=True), width="stretch")
        with st.expander("What is RFM?"):
            st.markdown("**R**ecency = days since last order, **F**requency = number of orders, "
                        "**M**onetary = total spend. Each is scored 1–5 by quintile; combinations "
                        "map to labels such as *Champions* or *At Risk*.")

# ---------------- Forecast ----------------
with tab_fc:
    st.caption("Forecast uses the *full* dataset (filters ignored) so seasonality can be learned.")
    try:
        res = F.forecast_sales(df_all)
    except ValueError as e:
        st.info(str(e))
    else:
        hist = res["history"].reset_index()
        hist.columns = ["Month", "Sales"]
        fc = res["forecast"].reset_index()
        fc.columns = ["Month", "Sales"]
        fig = px.line(hist, x="Month", y="Sales")
        fig.add_scatter(x=pd.concat([hist["Month"].tail(1), fc["Month"]]),
                        y=pd.concat([hist["Sales"].tail(1), fc["Sales"]]), mode="lines+markers", name="Forecast")
        st.plotly_chart(fig, width="stretch")
        st.markdown(f"**Selected model:** {res['best_model']} (lowest MAE on a {config.BACKTEST_MONTHS}-month hold-out)")
        st.dataframe(res["metrics"], width="stretch")

# ---------------- Data quality ----------------
with tab_dq:
    q = data_quality_report(df_all)
    a, b, c = st.columns(3)
    a.metric("Rows (after de-duplication)", f"{q['rows']:,}")
    b.metric("Ambiguous-date rows", f"{q['date_ambiguous_rows']:,}", f"{q['date_ambiguous_rows']/q['rows']:.1%}", delta_color="off")
    c.metric("Unresolved-date rows", f"{q['date_suspect_rows']:,}")
    st.markdown(
        "Dates arrive in mixed **DD/MM/YYYY** and **MM/DD/YYYY** formats. Each row is parsed under every "
        "possible reading and kept only if *Ship Date* falls 0–14 days after *Order Date* and the order "
        "year matches the year inside *Order ID*. Rows where two readings both pass are **ambiguous** "
        "(given the day-first reading); rows where none pass are **unresolved** and excluded from "
        "time-based charts.")
    if q["missing_by_column"]:
        st.write("Missing values:", q["missing_by_column"])
    st.write(f"Data source: `{q['source']}`")
    with st.expander("View filtered data"):
        st.dataframe(df, width="stretch")
