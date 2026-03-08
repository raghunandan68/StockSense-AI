"""
Smart Inventory Guardian – Analytics Engine
Handles risk detection, forecasting, and restock recommendations.
"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple

# ── Optional heavy learners ──────────────────────────────────────────────────
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder


# ═══════════════════════════════════════════════════════════════════════════
# 1.  RISK DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def detect_risks(inventory: pd.DataFrame,
                 sales_history: pd.DataFrame,
                 dead_stock_days: int = 30,
                 expiry_warning_days: int = 7) -> pd.DataFrame:
    """
    Returns inventory with risk columns appended:
        risk_level  : CRITICAL | HIGH | MEDIUM | LOW | OK
        risk_reason : human-readable explanation
        days_to_stockout
        days_to_expiry
    """
    df = inventory.copy()
    today = datetime.today().date()

    # ── Days to stockout ────────────────────────────────────────────────
    df["days_to_stockout"] = (
        df["Current_Stock"] / df["Daily_Sales_Avg"].replace(0, np.nan)
    ).round(1)

    # ── Days to expiry ──────────────────────────────────────────────────
    if "Expiry_Date" in df.columns:
        df["Expiry_Date"] = pd.to_datetime(df["Expiry_Date"], errors="coerce")
        df["days_to_expiry"] = (df["Expiry_Date"].dt.date.apply(
            lambda x: (x - today).days if pd.notna(x) else 9999))
    else:
        df["days_to_expiry"] = 9999

    # ── Dead-stock detection ─────────────────────────────────────────────
    if sales_history is not None and len(sales_history):
        recent_cut = (datetime.today() - timedelta(days=dead_stock_days)).date()
        recent = sales_history[
            pd.to_datetime(sales_history["Date"]).dt.date >= recent_cut
        ]
        sold_ids = recent[recent["Units_Sold"] > 0]["Product_ID"].unique()
        df["is_dead_stock"] = ~df["Product_ID"].isin(sold_ids)
    else:
        df["is_dead_stock"] = False

    # ── Assign risk ──────────────────────────────────────────────────────
    def _risk(row):
        reasons = []
        level = "OK"

        if row["days_to_expiry"] <= 3:
            level = "CRITICAL"; reasons.append(f"Expires in {int(row['days_to_expiry'])} day(s)!")
        elif row["days_to_expiry"] <= expiry_warning_days:
            level = "HIGH";     reasons.append(f"Expiring soon ({int(row['days_to_expiry'])} days)")

        if row["Current_Stock"] == 0:
            level = "CRITICAL"; reasons.append("OUT OF STOCK")
        elif row["days_to_stockout"] <= row["Lead_Time_Days"]:
            level = max(level, "CRITICAL", key=_level_order)
            reasons.append(f"Stockout in {row['days_to_stockout']}d (lead={int(row['Lead_Time_Days'])}d)")

        if row["is_dead_stock"]:
            lvl = "HIGH" if row["Current_Stock"] > row["Reorder_Point"] else "MEDIUM"
            level = max(level, lvl, key=_level_order)
            reasons.append(f"No sales in {dead_stock_days}+ days (dead stock)")

        if (row["Current_Stock"] <= row["Reorder_Point"] and level == "OK"):
            level = "MEDIUM"; reasons.append("Below reorder point")

        return pd.Series({"risk_level": level, "risk_reason": "; ".join(reasons) or "All good"})

    df[["risk_level", "risk_reason"]] = df.apply(_risk, axis=1)
    return df


def _level_order(level: str) -> int:
    return {"OK": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}.get(level, 0)


# ═══════════════════════════════════════════════════════════════════════════
# 2.  RESTOCK RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════

def compute_restock(inventory_risk: pd.DataFrame,
                    forecast_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Returns rows that need restocking with order quantities.
    Uses forecast if available, else falls back to daily average × coverage window.
    """
    needs_restock = inventory_risk[
        inventory_risk["risk_level"].isin(["CRITICAL", "HIGH", "MEDIUM"])
        & ~inventory_risk["is_dead_stock"]
    ].copy()

    coverage_days = 14  # order enough for 2 weeks

    if forecast_df is not None and len(forecast_df):
        # sum next 14-day forecast per product
        fc_sum = (forecast_df
                  .groupby("Product_ID")["Forecast_Units"]
                  .sum()
                  .rename("forecast_14d"))
        needs_restock = needs_restock.merge(fc_sum, on="Product_ID", how="left")
        needs_restock["order_qty"] = (
            needs_restock["forecast_14d"].fillna(
                needs_restock["Daily_Sales_Avg"] * coverage_days
            ) - needs_restock["Current_Stock"]
        ).clip(lower=0).round().astype(int)
    else:
        needs_restock["order_qty"] = (
            needs_restock["Daily_Sales_Avg"] * coverage_days
            - needs_restock["Current_Stock"]
        ).clip(lower=0).round().astype(int)

    needs_restock = needs_restock[needs_restock["order_qty"] > 0].copy()
    needs_restock["estimated_cost"] = (
        needs_restock["order_qty"] * needs_restock["Cost_Price"]
    ).round(2)

    # Delivery deadline
    today = datetime.today().date()
    needs_restock["order_by_date"] = needs_restock.apply(
        lambda r: (today + timedelta(days=max(0, int(r["days_to_stockout"] - r["Lead_Time_Days"] - 1)))).isoformat()
        if pd.notna(r["days_to_stockout"]) and r["days_to_stockout"] < 30
        else today.isoformat(),
        axis=1
    )

    needs_restock["action_text"] = needs_restock.apply(
        lambda r: (
            f"Order {int(r['order_qty'])} units of {r['Product_Name']} "
            f"(${r['estimated_cost']:.2f}) — deliver by {r['order_by_date']}"
        ), axis=1
    )

    return needs_restock.sort_values(
        "risk_level",
        key=lambda s: s.map({"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "OK": 4})
    )


# ═══════════════════════════════════════════════════════════════════════════
# 3.  DEMAND FORECASTING
# ═══════════════════════════════════════════════════════════════════════════

def _build_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray | None]:
    """Engineer time features from a daily sales df for one product."""
    df = df.copy().sort_values("Date")
    df["dayofweek"]  = df["Date"].dt.dayofweek
    df["dayofmonth"] = df["Date"].dt.day
    df["month"]      = df["Date"].dt.month
    df["lag1"]       = df["Units_Sold"].shift(1).fillna(0)
    df["lag7"]       = df["Units_Sold"].shift(7).fillna(0)
    df["roll7"]      = df["Units_Sold"].shift(1).rolling(7, min_periods=1).mean().fillna(0)
    df["roll14"]     = df["Units_Sold"].shift(1).rolling(14, min_periods=1).mean().fillna(0)
    le = LabelEncoder()
    df["event_enc"]  = le.fit_transform(df["Event"].fillna("Weekday"))

    feature_cols = ["dayofweek","dayofmonth","month","lag1","lag7","roll7","roll14","event_enc"]
    X = df[feature_cols].values
    y = df["Units_Sold"].values if "Units_Sold" in df.columns else None
    return X, y, feature_cols


def forecast_product(product_history: pd.DataFrame,
                     horizon: int = 7,
                     model_type: str = "auto") -> pd.DataFrame:
    """
    Forecast `horizon` days of sales for a single product.
    Returns DataFrame with columns: Date, Forecast_Units.
    """
    hist = product_history.copy()
    hist["Date"] = pd.to_datetime(hist["Date"])
    hist = hist.sort_values("Date").reset_index(drop=True)

    if len(hist) < 14:
        # Not enough history — return moving average
        avg = hist["Units_Sold"].tail(7).mean() if len(hist) else 0
        dates = [datetime.today().date() + timedelta(days=i+1) for i in range(horizon)]
        return pd.DataFrame({"Date": dates, "Forecast_Units": round(avg, 1)})

    X, y, _ = _build_features(hist)

    # Choose model
    if model_type == "auto":
        if XGB_AVAILABLE and len(hist) >= 30:
            model_type = "xgb"
        elif LGB_AVAILABLE and len(hist) >= 30:
            model_type = "lgb"
        else:
            model_type = "linear"

    if model_type == "xgb" and XGB_AVAILABLE:
        model = XGBRegressor(n_estimators=100, max_depth=4,
                             learning_rate=0.1, random_state=42, verbosity=0)
    elif model_type == "lgb" and LGB_AVAILABLE:
        model = LGBMRegressor(n_estimators=100, max_depth=4,
                              learning_rate=0.1, random_state=42, verbose=-1)
    else:
        model = LinearRegression()

    model.fit(X, y)

    # Roll forward
    last_sales = list(hist["Units_Sold"].values)
    preds = []
    today = datetime.today().date()

    for i in range(1, horizon + 1):
        fc_date = today + timedelta(days=i)
        fc_dt   = datetime.combine(fc_date, datetime.min.time())
        event   = _get_event(fc_dt)
        event_enc = {"Weekday": 0, "Weekend": 1, "Public Holiday": 2,
                     "Pre-Holiday": 3, "Rainy Day": 4}.get(event, 0)
        lag1  = last_sales[-1] if last_sales else 0
        lag7  = last_sales[-7] if len(last_sales) >= 7 else np.mean(last_sales)
        roll7 = np.mean(last_sales[-7:])  if len(last_sales) >= 7  else np.mean(last_sales)
        roll14= np.mean(last_sales[-14:]) if len(last_sales) >= 14 else np.mean(last_sales)
        row = np.array([[fc_date.weekday(), fc_date.day, fc_date.month,
                         lag1, lag7, roll7, roll14, event_enc]])
        pred = max(0, float(model.predict(row)[0]))
        preds.append({"Date": fc_date, "Forecast_Units": round(pred, 1), "Event": event})
        last_sales.append(pred)

    return pd.DataFrame(preds)


def forecast_all(inventory: pd.DataFrame,
                 sales_history: pd.DataFrame,
                 horizon: int = 7) -> pd.DataFrame:
    """Runs forecasting for every product and returns combined DataFrame."""
    results = []
    for pid in inventory["Product_ID"].unique():
        prod_hist = sales_history[sales_history["Product_ID"] == pid].copy()
        fc = forecast_product(prod_hist, horizon=horizon)
        fc["Product_ID"] = pid
        prod_name = inventory.loc[inventory["Product_ID"] == pid, "Product_Name"].values
        fc["Product_Name"] = prod_name[0] if len(prod_name) else pid
        results.append(fc)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def _get_event(dt: datetime) -> str:
    holidays = {(1,1),(25,12),(26,12),(1,5),(25,3)}
    if (dt.day, dt.month) in holidays:
        return "Public Holiday"
    if dt.weekday() == 4:
        return "Pre-Holiday"
    if dt.weekday() >= 5:
        return "Weekend"
    return "Weekday"


# ═══════════════════════════════════════════════════════════════════════════
# 4.  SUMMARY KPIs
# ═══════════════════════════════════════════════════════════════════════════

def compute_kpis(inventory_risk: pd.DataFrame,
                 sales_history: pd.DataFrame) -> dict:
    kpis = {}
    kpis["total_products"]   = len(inventory_risk)
    kpis["critical_count"]   = int((inventory_risk["risk_level"] == "CRITICAL").sum())
    kpis["high_count"]       = int((inventory_risk["risk_level"] == "HIGH").sum())
    kpis["out_of_stock"]     = int((inventory_risk["Current_Stock"] == 0).sum())
    kpis["dead_stock_count"] = int(inventory_risk["is_dead_stock"].sum())
    kpis["expiring_soon"]    = int((inventory_risk["days_to_expiry"] <= 7).sum())

    # Inventory value
    kpis["inventory_value"] = round(
        (inventory_risk["Current_Stock"] * inventory_risk["Cost_Price"]).sum(), 2
    )
    kpis["potential_revenue"] = round(
        (inventory_risk["Current_Stock"] * inventory_risk["Selling_Price"]).sum(), 2
    )

    # Dead stock loss risk
    dead = inventory_risk[inventory_risk["is_dead_stock"]]
    kpis["dead_stock_value"] = round(
        (dead["Current_Stock"] * dead["Cost_Price"]).sum(), 2
    )

    # 7-day revenue from history
    if sales_history is not None and len(sales_history):
        last7 = (datetime.today() - timedelta(days=7)).date()
        recent = sales_history[pd.to_datetime(sales_history["Date"]).dt.date >= last7]
        kpis["revenue_last_7d"] = round(recent["Revenue"].sum(), 2)
    else:
        kpis["revenue_last_7d"] = 0.0

    return kpis
