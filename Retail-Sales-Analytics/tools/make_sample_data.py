"""
Generates a SYNTHETIC dataset with the same schema as the Kaggle 'Sample
Superstore' file, so the project can be run/tested without the real CSV.

The numbers are random - do NOT report them as real findings. For your
submission, place the real dataset at data/sales_data.csv.

Includes deliberately mixed DD/MM/YYYY and MM/DD/YYYY dates to exercise the
cleaning pipeline.
"""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
N_ORDERS, N_CUST = 2200, 420

catalog = {
    "Furniture": {"Chairs": (150, 90), "Tables": (300, 180), "Bookcases": (200, 120), "Furnishings": (40, 25)},
    "Office Supplies": {"Paper": (25, 15), "Binders": (30, 20), "Storage": (70, 40), "Art": (20, 12), "Labels": (15, 8)},
    "Technology": {"Phones": (250, 160), "Accessories": (60, 40), "Machines": (500, 350), "Copiers": (900, 500)},
}
regions = {"West": ["California", "Washington"], "East": ["New York", "Pennsylvania"],
           "Central": ["Texas", "Illinois"], "South": ["Florida", "Georgia"]}
region_w = {"West": 0.32, "East": 0.29, "Central": 0.22, "South": 0.17}
ship_modes = {"Standard Class": (0.6, (4, 7)), "Second Class": (0.2, (2, 5)),
              "First Class": (0.15, (1, 3)), "Same Day": (0.05, (0, 0))}

products = []
for cat, subs in catalog.items():
    for sub, (mu, sd) in subs.items():
        for i in range(1, 9):
            products.append((f"{cat[:3].upper()}-{sub[:2].upper()}-{i:04d}", cat, sub, f"{sub} Model {chr(64+i)}", max(mu + rng.normal(0, sd * .3), 5)))

segs = rng.choice(["Consumer", "Corporate", "Home Office"], N_CUST, p=[.52, .30, .18])
cust = pd.DataFrame({"Customer ID": [f"CU-{i:05d}" for i in range(N_CUST)],
                     "Customer Name": [f"Customer {i}" for i in range(N_CUST)], "Segment": segs})
cust_w = rng.pareto(1.6, N_CUST) + 1; cust_w /= cust_w.sum()  # few customers buy a lot

rows = []
days = pd.date_range("2015-01-01", "2018-12-30")
month_season = {1: .7, 2: .65, 3: .9, 4: .85, 5: .9, 6: .95, 7: .9, 8: .95, 9: 1.3, 10: 1.0, 11: 1.6, 12: 1.7}
day_w = np.array([month_season[d.month] * (1 + 0.22 * (d.year - 2015)) for d in days]); day_w /= day_w.sum()

for k in range(N_ORDERS):
    od = rng.choice(days, p=day_w)
    od = pd.Timestamp(od)
    mode = rng.choice(list(ship_modes), p=[v[0] for v in ship_modes.values()])
    lo, hi = ship_modes[mode][1]
    sd_ = od + pd.Timedelta(days=int(rng.integers(lo, hi + 1)))
    c = cust.iloc[rng.choice(N_CUST, p=cust_w)]
    reg = rng.choice(list(regions), p=list(region_w.values()))
    state = rng.choice(regions[reg])
    oid = f"{reg[:2].upper()}-{od.year}-{100000 + k}"
    for _ in range(int(rng.integers(1, 5))):
        pid, cat, sub, name, base = products[rng.integers(len(products))]
        sales = round(float(base * rng.lognormal(0, 0.55) * rng.integers(1, 4)), 2)
        # ~45% of rows use MM/DD/YYYY, others DD/MM/YYYY  -> mixed-format column
        mdy = rng.random() < 0.45
        f = "%m/%d/%Y" if mdy else "%d/%m/%Y"
        rows.append({
            "Row ID": len(rows) + 1, "Order ID": oid, "Order Date": od.strftime(f), "Ship Date": sd_.strftime(f),
            "Ship Mode": mode, "Customer ID": c["Customer ID"], "Customer Name": c["Customer Name"],
            "Segment": c["Segment"], "Country": "United States", "City": f"{state} City", "State": state,
            "Postal Code": rng.integers(10000, 99999), "Region": reg, "Product ID": pid, "Category": cat,
            "Sub-Category": sub, "Product Name": name, "Sales": sales})

df = pd.DataFrame(rows)
# inject a few data-quality problems
df["Postal Code"] = df["Postal Code"].astype("Int64")
df.loc[rng.choice(len(df), 12, replace=False), "Postal Code"] = pd.NA
df = pd.concat([df, df.sample(8, random_state=1)], ignore_index=True)  # exact duplicates

out = Path(__file__).resolve().parent.parent / "data" / "sample_sales_data.csv"
df.to_csv(out, index=False)
print(f"Wrote {len(df)} rows -> {out}")
