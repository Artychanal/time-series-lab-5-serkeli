from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_steam(path: Path):
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.sort_values("date").reset_index(drop=True)
    columns = ["positive_reviews", "negative_reviews", "total_reviews", "positive_share"]
    frame[columns] = frame[columns].apply(pd.to_numeric, errors="raise")
    if frame[columns].isna().any().any() or not np.isfinite(frame[columns]).all().all():
        raise ValueError("Nonfinite Steam data")
    if frame.date.duplicated().any():
        raise ValueError("Duplicate months")
    expected = pd.date_range(frame.date.min(), frame.date.max(), freq="MS")
    if not frame.date.equals(pd.Series(expected, name="date")):
        raise ValueError("Steam calendar must contain consecutive month starts")
    counts = frame[columns[:3]].to_numpy(float)
    if (counts < 0).any() or not np.equal(counts, np.floor(counts)).all():
        raise ValueError("Invalid review counts")
    if not np.array_equal(counts[:, 0] + counts[:, 1], counts[:, 2]):
        raise ValueError("Review sum identity failed")
    if not np.allclose(frame.positive_share, frame.positive_reviews / frame.total_reviews, atol=1e-9, rtol=0):
        raise ValueError("Review share identity failed")
    selected = frame.loc[frame.date <= "2026-08-01"].copy().reset_index(drop=True)
    if len(selected) < 60:
        raise ValueError("At least 60 complete months are required")
    metadata = json.loads(path.with_name("cyberpunk_2077_metadata.json").read_text(encoding="utf-8"))
    return selected, {"rows": len(selected), "raw_rows": len(frame), "start": str(selected.date.min().date()),
                      "end": str(selected.date.max().date()), "excluded": frame.loc[frame.date > "2026-08-01", "date"].astype(str).tolist(),
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "missing": 0,
                      "duplicates": 0, "source": metadata}


def money(values):
    return pd.to_numeric(values.astype(str).str.replace("$", "", regex=False)
                         .str.replace("\u00a0", "", regex=False).str.replace(" ", "", regex=False)
                         .str.replace(",", ".", regex=False), errors="raise")


def aggregate_sales(frame):
    return frame.groupby("date").agg(revenue=("revenue", "sum"), quantity=("quantity", "sum"),
                                    shipping=("shipping", "sum"), orders=("order_id", "size")).sort_index()


def load_sales(path: Path):
    raw = pd.read_csv(path, skiprows=2, encoding="utf-8-sig")
    if raw.isna().any().any() or raw["Order ID"].duplicated().any():
        raise ValueError("Missing cells or duplicate order IDs")
    frame = pd.DataFrame({"order_id": raw["Order ID"], "date": pd.to_datetime(raw["Order Date"], format="%m/%d/%y"),
                          "shipped_date": pd.to_datetime(raw["Shipped Date"], format="%m/%d/%y"),
                          "product": raw["Product Name"], "category": raw.Category, "region": raw.Region,
                          "price": money(raw["Unit Price"]), "quantity": pd.to_numeric(raw.Quantity),
                          "revenue": money(raw.Revenue), "shipping": money(raw["Shipping Fee"])})
    frame["delivery_days"] = (frame.shipped_date - frame.date).dt.days
    if not np.isfinite(frame[["price", "quantity", "revenue", "shipping"]]).all().all():
        raise ValueError("Nonfinite sales data")
    if (frame[["price", "quantity", "revenue"]] <= 0).any().any() or (frame.shipping < 0).any():
        raise ValueError("Invalid sales magnitudes")
    if not np.equal(frame.quantity, np.floor(frame.quantity)).all() or (frame.delivery_days < 0).any():
        raise ValueError("Invalid quantities or delivery dates")
    discrepancy = np.abs(frame.price * frame.quantity - frame.revenue)
    if discrepancy.max() > 0.010001:
        raise ValueError("Revenue identity failed")
    frame["shipping_rate"] = frame.shipping / frame.revenue
    daily = aggregate_sales(frame)
    return frame, daily, {"rows": len(frame), "raw_columns": len(raw.columns), "unique_dates": len(daily),
                          "start": str(frame.date.min().date()), "end": str(frame.date.max().date()),
                          "title": path.read_text(encoding="utf-8-sig").splitlines()[0].split(",")[0],
                          "date_title_conflict": True, "missing": 0, "duplicates": 0,
                          "revenue_identity_max_error": float(discrepancy.max()),
                          "constant_delivery_days": int(frame.delivery_days.nunique()) == 1,
                          "calendar_policy": "Absent dates are unknown; daily analysis uses the nine observed dates",
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
