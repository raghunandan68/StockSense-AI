# 🛡️ Smart Inventory Guardian
**AI-Powered Restock & Risk Detection System**

Built for the Smart Inventory Hackathon.

---

## 📦 Tech Stack
| Layer | Technology |
|---|---|
| Data Manipulation | Python, Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost, LightGBM |
| Dashboard / UI | Streamlit + Plotly |
| Data Format | CSV / Excel (.xlsx) |

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## 📁 Project Structure
```
smart_inventory_guardian/
├── app.py                        # Main Streamlit dashboard
├── requirements.txt
├── README.md
├── data/
│   ├── generate_sample_data.py   # Demo data generator (30 products, 90-day history)
│   ├── inventory.csv             # Auto-generated on first run
│   └── sales_history.csv         # Auto-generated on first run
└── utils/
    └── analytics.py              # Core AI engine
        ├── detect_risks()        # Risk scoring (CRITICAL / HIGH / MEDIUM / LOW / OK)
        ├── compute_restock()     # Order quantity calculator
        ├── forecast_product()    # XGBoost/LightGBM/Linear demand forecasting
        ├── forecast_all()        # Runs forecast for all products
        └── compute_kpis()        # Dashboard KPI metrics
```

---

## 🧠 How the AI Works

### Risk Detection (`detect_risks`)
Each product is scored based on:
- **Days to stockout** = `Current Stock ÷ Daily Sales` vs Lead Time
- **Days to expiry** — triggers warning at ≤7 days, critical at ≤3 days
- **Dead stock detection** — no sales in configurable window (default 30 days)
- **Reorder point breach** — stock below calculated safety level

Risk levels: `CRITICAL` → `HIGH` → `MEDIUM` → `LOW` → `OK`

### Demand Forecasting (`forecast_product`)
Uses **feature engineering** + **gradient boosting**:
- Features: day of week, day of month, month, lag-1, lag-7, 7-day rolling avg, 14-day rolling avg, event type
- Model auto-selection: XGBoost (if ≥30 days history) → LightGBM → Linear Regression fallback
- Event-aware: weekends, public holidays, and pre-holiday spikes are encoded

### Restock Calculator (`compute_restock`)
- Order quantity = `(14-day forecast OR 14× daily avg) − current stock`
- Calculates `order_by_date` from days-to-stockout minus lead time
- Generates human-readable action text: *"Order 20 units of Milk to last until Tuesday"*

---

## 📊 Dashboard Features

| Tab | Contents |
|---|---|
| 📊 Guardian Dashboard | Risk donut, category heatmap, stock scatter, 7-day revenue |
| 🚨 Smart Alerts | Critical/High/Medium item cards, dead stock table |
| 📈 Forecast View | Per-product forecast chart + aggregate daily units |
| 🛒 Restock Planner | Actionable order list, cost breakdown, CSV download |
| 🔍 Data Explorer | Full styled inventory table + 30-day sales trend |

---

## 📤 Upload Your Own Data

**Inventory CSV** must include:
`Product_ID, Product_Name, Category, Current_Stock, Reorder_Point, Max_Capacity, Daily_Sales_Avg, Cost_Price, Selling_Price, Expiry_Date, Lead_Time_Days, Supplier`

**Sales History CSV** must include:
`Date, Product_ID, Product_Name, Category, Units_Sold, Revenue, Event`

---

## 🏆 Hackathon Evaluation Mapping

| Criterion | Implementation |
|---|---|
| Data Analysis | Pandas pipeline with aggregation, rolling averages, lag features |
| Predictive Logic | XGBoost / LightGBM / Linear Regression with automatic fallback |
| UX/UI Design | Dark-mode Streamlit dashboard, colour-coded risk badges, KPI cards |
| Business Logic | Reorder points, lead times, dead stock, expiry, margin-aware ordering |
| Actionable Insights | Plain-English action text per product with cost & deadline |
