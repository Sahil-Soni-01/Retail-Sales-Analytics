"""Central configuration: paths and tunable constants."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Real dataset goes here (see README). Override with env var RETAIL_DATA_PATH.
DATA_PATH = Path(os.environ.get("RETAIL_DATA_PATH", DATA_DIR / "sales_data.csv"))
SAMPLE_DATA_PATH = DATA_DIR / "sample_sales_data.csv"

# A real order ships within this many days; used to validate parsed dates.
MAX_SHIP_DAYS = 14

# Forecasting / backtesting
FORECAST_HORIZON = 6      # months to forecast
BACKTEST_MONTHS = 6       # months held out for model evaluation

# RFM / clustering
RFM_CLUSTERS = 4
RANDOM_STATE = 42

REQUIRED_COLUMNS = [
    "Order ID", "Order Date", "Ship Date", "Ship Mode", "Customer ID",
    "Segment", "Region", "Category", "Sub-Category", "Product Name", "Sales",
]
