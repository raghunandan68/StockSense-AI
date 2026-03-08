"""
Sample Data Generator for Smart Inventory Guardian
Generates realistic retail inventory and sales data for demonstration.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

random.seed(42)
np.random.seed(42)

PRODUCTS = [
    # (Product Name, Category, Cost Price, Sell Price, Shelf Life Days, Min Stock, Max Stock)
    ("Organic Whole Milk 1L",     "Dairy",      1.20, 2.50,   7,  20, 100),
    ("Greek Yogurt 500g",         "Dairy",      0.90, 1.80,  14,  15,  80),
    ("Cheddar Cheese 250g",       "Dairy",      1.50, 3.20,  30,  10,  60),
    ("Fresh Bread Loaf",          "Bakery",     0.80, 1.50,   3,  25, 120),
    ("Croissants x4",             "Bakery",     1.00, 2.20,   2,  20, 100),
    ("Brown Eggs x12",            "Eggs",       1.80, 3.50,  21,  15,  70),
    ("Free Range Eggs x6",        "Eggs",       1.20, 2.80,  21,  10,  50),
    ("Paracetamol 500mg x16",     "Pharmacy",   0.60, 2.99, 730,   5,  40),
    ("Ibuprofen 200mg x24",       "Pharmacy",   0.90, 3.99, 730,   5,  40),
    ("Vitamin C 1000mg x30",      "Pharmacy",   2.00, 7.50, 365,   8,  50),
    ("Multivitamin Tablets x60",  "Pharmacy",   3.00,10.99, 365,   5,  30),
    ("Antiseptic Cream 50g",      "Pharmacy",   1.20, 4.50, 730,   5,  25),
    ("Chicken Breast 500g",       "Meat",       2.80, 5.50,   5,  10,  60),
    ("Ground Beef 500g",          "Meat",       2.20, 4.80,   5,  10,  60),
    ("Salmon Fillet 300g",        "Seafood",    3.50, 7.99,   4,   8,  40),
    ("Tinned Tomatoes 400g",      "Canned",     0.30, 0.89, 730,  30, 150),
    ("Baked Beans 415g",          "Canned",     0.35, 0.99, 730,  30, 150),
    ("Pasta Penne 500g",          "Dry Goods",  0.50, 1.29, 730,  20, 120),
    ("Basmati Rice 1kg",          "Dry Goods",  0.90, 2.10, 730,  15,  90),
    ("Olive Oil 500ml",           "Oils",       2.50, 5.99, 365,  10,  60),
    ("Orange Juice 1L",           "Beverages",  0.70, 1.80,  14,  20,  90),
    ("Sparkling Water 1.5L",      "Beverages",  0.20, 0.89, 365,  30, 150),
    ("Cola 2L",                   "Beverages",  0.55, 1.50, 365,  25, 120),
    ("Cornflakes 500g",           "Cereals",    0.80, 2.20, 180,  10,  60),
    ("Rolled Oats 1kg",           "Cereals",    0.90, 2.50, 365,  10,  50),
    ("Frozen Pizza Margherita",   "Frozen",     1.80, 3.99, 180,  12,  60),
    ("Ice Cream Vanilla 500ml",   "Frozen",     1.20, 3.50, 180,  10,  50),
    ("Shampoo 400ml",             "Personal",   1.50, 4.99, 730,   8,  40),
    ("Toothpaste 100ml",          "Personal",   0.80, 2.99, 730,  10,  50),
    ("Hand Sanitizer 250ml",      "Personal",   0.60, 2.49, 730,  15,  60),
]

SPECIAL_EVENTS = {
    "Weekend":       1.25,
    "Public Holiday":1.50,
    "Weekday":       1.00,
    "Pre-Holiday":   1.35,
    "Rainy Day":     0.85,
}


def get_event_for_date(d: datetime) -> str:
    holidays = {(1,1),(25,12),(26,12),(1,5),(25,3)}
    if (d.day, d.month) in holidays:
        return "Public Holiday"
    if d.weekday() == 4:           # Friday
        return "Pre-Holiday"
    if d.weekday() >= 5:           # Sat/Sun
        return "Weekend"
    return "Weekday"


def generate_inventory(save_path: str = "inventory.csv") -> pd.DataFrame:
    today = datetime.today().date()
    rows = []
    for pid, (name, cat, cost, sell, shelf, min_s, max_s) in enumerate(PRODUCTS, 1):
        stock = random.randint(max(0, min_s - 5), max_s)
        daily_sales = round(random.uniform(2, 25), 1)

        # Inject interesting edge cases
        if pid in [5, 17]:      # dead stock candidates
            stock = random.randint(60, 100)
            daily_sales = round(random.uniform(0.1, 0.5), 1)
        if pid in [1, 13, 21]:  # low stock / restock needed
            stock = random.randint(0, min_s - 1)
        if pid in [3, 9, 26]:   # near expiry
            shelf = random.randint(3, 8)

        expiry = (today + timedelta(days=shelf)).isoformat() if shelf < 730 else (today + timedelta(days=shelf)).isoformat()
        lead_time = random.randint(1, 5)
        reorder_point = round(daily_sales * lead_time * 1.3)

        rows.append({
            "Product_ID":       f"P{pid:03d}",
            "Product_Name":     name,
            "Category":         cat,
            "Current_Stock":    stock,
            "Reorder_Point":    reorder_point,
            "Max_Capacity":     max_s,
            "Daily_Sales_Avg":  daily_sales,
            "Cost_Price":       cost,
            "Selling_Price":    sell,
            "Expiry_Date":      expiry,
            "Lead_Time_Days":   lead_time,
            "Supplier":         f"Supplier_{chr(64 + (pid % 8) + 1)}",
        })
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    return df


def generate_sales_history(inventory_df: pd.DataFrame,
                           days: int = 90,
                           save_path: str = "sales_history.csv") -> pd.DataFrame:
    today = datetime.today().date()
    rows = []
    for _, prod in inventory_df.iterrows():
        base_sales = prod["Daily_Sales_Avg"]
        for d in range(days, 0, -1):
            sale_date = today - timedelta(days=d)
            event = get_event_for_date(datetime.combine(sale_date, datetime.min.time()))
            multiplier = SPECIAL_EVENTS[event]
            # Add weekly seasonality and random noise
            weekly = 1 + 0.15 * np.sin(2 * np.pi * sale_date.weekday() / 7)
            noise = np.random.normal(1.0, 0.12)
            units = max(0, int(round(base_sales * multiplier * weekly * noise)))
            revenue = round(units * prod["Selling_Price"], 2)
            rows.append({
                "Date":         sale_date.isoformat(),
                "Product_ID":   prod["Product_ID"],
                "Product_Name": prod["Product_Name"],
                "Category":     prod["Category"],
                "Units_Sold":   units,
                "Revenue":      revenue,
                "Event":        event,
            })
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    return df


if __name__ == "__main__":
    import os, pathlib
    out = pathlib.Path(__file__).parent
    inv = generate_inventory(str(out / "inventory.csv"))
    generate_sales_history(inv, days=90, save_path=str(out / "sales_history.csv"))
    print(f"Generated {len(inv)} products and 90-day sales history.")
