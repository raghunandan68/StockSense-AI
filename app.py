"""
╔══════════════════════════════════════════════════════════╗
║        SMART INVENTORY GUARDIAN  – Main App              ║
║   AI-Powered Restock & Risk Detection System             ║
╚══════════════════════════════════════════════════════════╝
Run:  streamlit run app.py
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io, os

from data.generate_sample_data import generate_inventory, generate_sales_history
from utils.analytics import (
    detect_risks, compute_restock, forecast_all, compute_kpis
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Inventory Guardian",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Sidebar */
  [data-testid="stSidebar"] { background: #0f172a; }
  [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
  /* KPI cards */
  .kpi-card {
    background: linear-gradient(135deg,#1e293b,#0f172a);
    border-radius:12px; padding:18px 22px;
    border-left: 4px solid #3b82f6;
    margin-bottom: 8px;
  }
  .kpi-value { font-size:2rem; font-weight:700; color:#60a5fa; }
  .kpi-label { font-size:.85rem; color:#94a3b8; margin-top:2px; }
  /* Risk badges */
  .badge-critical { background:#ef4444;color:#fff;padding:3px 10px;border-radius:99px;font-size:.78rem;font-weight:600; }
  .badge-high     { background:#f97316;color:#fff;padding:3px 10px;border-radius:99px;font-size:.78rem;font-weight:600; }
  .badge-medium   { background:#eab308;color:#000;padding:3px 10px;border-radius:99px;font-size:.78rem;font-weight:600; }
  .badge-low      { background:#22c55e;color:#fff;padding:3px 10px;border-radius:99px;font-size:.78rem;font-weight:600; }
  .badge-ok       { background:#3b82f6;color:#fff;padding:3px 10px;border-radius:99px;font-size:.78rem;font-weight:600; }
  /* Alert boxes */
  .alert-critical { background:#450a0a;border-left:4px solid #ef4444;padding:12px 16px;border-radius:8px;margin:6px 0;color:#fca5a5 !important; }
  .alert-critical b { color:#ffffff !important; }
  .alert-critical small { color:#fca5a5 !important; }
  .alert-high { background:#431407;border-left:4px solid #f97316;padding:12px 16px;border-radius:8px;margin:6px 0;color:#fdba74 !important; }
  .alert-high b { color:#ffffff !important; }
  .alert-high small { color:#fdba74 !important; }
  .action-box { background:#0c2340;border-left:4px solid #3b82f6;padding:12px 16px;border-radius:8px;margin:6px 0;font-size:.9rem;color:#e2e8f0 !important; }
  .action-box * { color:#e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# SESSION-STATE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_demo_data():
    """Generate and return demo data."""
    inv  = generate_inventory()
    sal  = generate_sales_history(inv, days=90)
    return inv, sal


def load_uploaded(inv_file, sal_file):
    inv = pd.read_csv(inv_file) if inv_file.name.endswith(".csv") else pd.read_excel(inv_file)
    sal = pd.read_csv(sal_file) if sal_file.name.endswith(".csv") else pd.read_excel(sal_file)
    return inv, sal


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🛡️ Inventory Guardian")
    st.markdown("---")

    data_source = st.radio("**Data Source**",
                           ["📦 Use Demo Data", "📤 Upload My Data"],
                           index=0)

    if data_source == "📤 Upload My Data":
        inv_upload = st.file_uploader("Inventory CSV/Excel", type=["csv","xlsx"])
        sal_upload = st.file_uploader("Sales History CSV/Excel", type=["csv","xlsx"])
        if inv_upload and sal_upload:
            inventory_raw, sales_raw = load_uploaded(inv_upload, sal_upload)
            st.success("✅ Files loaded!")
        else:
            st.info("Upload both files to proceed.")
            inventory_raw, sales_raw = load_demo_data()
    else:
        inventory_raw, sales_raw = load_demo_data()

    st.markdown("---")
    st.markdown("**⚙️ Settings**")
    dead_stock_days    = st.slider("Dead Stock Threshold (days)", 14, 60, 30)
    expiry_warn_days   = st.slider("Expiry Warning Window (days)",  3, 21,  7)
    forecast_horizon   = st.slider("Forecast Horizon (days)",        3, 14,  7)
    cat_filter         = st.multiselect("Filter by Category",
                                        sorted(inventory_raw["Category"].unique()),
                                        default=[])
    st.markdown("---")
    st.caption("Built for the Smart Inventory Hackathon 🚀")


# ═══════════════════════════════════════════════════════════════════════════
# DATA PROCESSING
# ═══════════════════════════════════════════════════════════════════════════

with st.spinner("🧠 Analysing inventory…"):
    inv_risk   = detect_risks(inventory_raw, sales_raw,
                              dead_stock_days=dead_stock_days,
                              expiry_warning_days=expiry_warn_days)
    forecast   = forecast_all(inventory_raw, sales_raw, horizon=forecast_horizon)
    restock_df = compute_restock(inv_risk, forecast)
    kpis       = compute_kpis(inv_risk, sales_raw)

if cat_filter:
    inv_risk   = inv_risk[inv_risk["Category"].isin(cat_filter)]
    restock_df = restock_df[restock_df["Category"].isin(cat_filter)]


# ═══════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown(
    "<h1 style='margin-bottom:0'>🛡️ StockSense AI</h1>"
    "<p style='color:#94a3b8;margin-top:2px'>AI-Powered Restock & Risk Detection · "
    f"Last updated: {datetime.now().strftime('%d %b %Y, %H:%M')}</p>",
    unsafe_allow_html=True
)
st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════
# KPI ROW
# ═══════════════════════════════════════════════════════════════════════════

c1,c2,c3,c4,c5,c6 = st.columns(6)
kpi_data = [
    (c1, "🔴 Critical Alerts",   kpis["critical_count"],   "#ef4444"),
    (c2, "🟠 High Risk",          kpis["high_count"],        "#f97316"),
    (c3, "🚫 Out of Stock",       kpis["out_of_stock"],      "#dc2626"),
    (c4, "💀 Dead Stock",         kpis["dead_stock_count"],  "#7c3aed"),
    (c5, "⏰ Expiring ≤7d",       kpis["expiring_soon"],     "#d97706"),
    (c6, "📦 Inventory Value",    f"${kpis['inventory_value']:,.0f}", "#3b82f6"),
]
for col, label, val, color in kpi_data:
    col.markdown(
        f"<div class='kpi-card' style='border-color:{color}'>"
        f"<div class='kpi-value' style='color:{color}'>{val}</div>"
        f"<div class='kpi-label'>{label}</div></div>",
        unsafe_allow_html=True
    )

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Guardian Dashboard",
    "🚨 Smart Alerts",
    "📈 Forecast View",
    "🛒 Restock Planner",
    "🔍 Data Explorer",
])


# ───────────────────────────────────────────────────────────────────────────
#  TAB 1 – GUARDIAN DASHBOARD
# ───────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("📊 Inventory Health Overview")

    # Risk distribution donut
    risk_counts = inv_risk["risk_level"].value_counts().reset_index()
    risk_counts.columns = ["Risk Level","Count"]
    color_map = {"CRITICAL":"#ef4444","HIGH":"#f97316","MEDIUM":"#eab308",
                 "LOW":"#22c55e","OK":"#3b82f6"}

    col_l, col_r = st.columns([1,1])

    with col_l:
        fig_donut = px.pie(
            risk_counts, values="Count", names="Risk Level",
            color="Risk Level", color_discrete_map=color_map,
            hole=0.55, title="Risk Distribution"
        )
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", legend_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=40,b=0,l=0,r=0)
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_r:
        # Category health heatmap
        cat_risk = inv_risk.groupby("Category")["risk_level"].apply(
            lambda x: (x.isin(["CRITICAL","HIGH"]).sum() / len(x) * 100)
        ).reset_index()
        cat_risk.columns = ["Category","Risk %"]
        cat_risk = cat_risk.sort_values("Risk %", ascending=True)
        fig_bar = px.bar(
            cat_risk, x="Risk %", y="Category", orientation="h",
            title="Category Risk Score (%)", color="Risk %",
            color_continuous_scale=["#22c55e","#eab308","#ef4444"],
            range_color=[0,100]
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", coloraxis_showscale=False,
            margin=dict(t=40,b=0,l=0,r=0)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # Stock level vs reorder point scatter
    st.subheader("📉 Stock Levels vs Reorder Points")
    fig_scatter = px.scatter(
        inv_risk, x="Product_Name", y="Current_Stock",
        color="risk_level", color_discrete_map=color_map,
        size="Daily_Sales_Avg", hover_data=["Category","days_to_stockout","risk_reason"],
        title="Current Stock (bubble size = daily sales velocity)"
    )
    # Add reorder point line as a bar underneath
    fig_scatter.add_scatter(
        x=inv_risk["Product_Name"], y=inv_risk["Reorder_Point"],
        mode="markers", marker=dict(symbol="line-ew", size=12, color="#f97316",
                                     line=dict(width=2,color="#f97316")),
        name="Reorder Point"
    )
    fig_scatter.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0", xaxis_tickangle=-45,
        margin=dict(t=50,b=120,l=0,r=0), height=420
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Revenue last 7 days by category
    st.markdown("---")
    st.subheader("💰 Revenue – Last 7 Days by Category")
    sales_raw["Date"] = pd.to_datetime(sales_raw["Date"])
    last7 = sales_raw[sales_raw["Date"] >= (datetime.now() - timedelta(days=7))]
    rev_cat = last7.groupby("Category")["Revenue"].sum().reset_index().sort_values("Revenue",ascending=False)
    fig_rev = px.bar(rev_cat, x="Category", y="Revenue",
                     color="Revenue", color_continuous_scale="blues",
                     title="7-Day Revenue by Category")
    fig_rev.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0", coloraxis_showscale=False,
        margin=dict(t=40,b=0,l=0,r=0)
    )
    st.plotly_chart(fig_rev, use_container_width=True)


# ───────────────────────────────────────────────────────────────────────────
#  TAB 2 – SMART ALERTS
# ───────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("🚨 Smart Alerts & Risk Warnings")

    col_crit, col_high = st.columns(2)

    with col_crit:
        st.markdown("#### 🔴 CRITICAL – Immediate Action Required")
        crit = inv_risk[inv_risk["risk_level"]=="CRITICAL"]
        if crit.empty:
            st.success("✅ No critical items — great job!")
        for _, r in crit.iterrows():
            st.markdown(
                f"<div class='alert-critical'>"
                f"<b>{r['Product_Name']}</b> ({r['Category']})<br>"
                f"<small>{r['risk_reason']}</small><br>"
                f"<small>Stock: <b>{int(r['Current_Stock'])}</b> | "
                f"Stockout in: <b>{r['days_to_stockout']:.0f}d</b> | "
                f"Expiry: <b>{int(r['days_to_expiry'])}d</b></small>"
                f"</div>", unsafe_allow_html=True
            )

    with col_high:
        st.markdown("#### 🟠 HIGH – Act This Week")
        high = inv_risk[inv_risk["risk_level"]=="HIGH"]
        if high.empty:
            st.success("✅ No high-risk items right now.")
        for _, r in high.iterrows():
            st.markdown(
                f"<div class='alert-high'>"
                f"<b>{r['Product_Name']}</b> ({r['Category']})<br>"
                f"<small>{r['risk_reason']}</small><br>"
                f"<small>Stock: <b>{int(r['Current_Stock'])}</b> | "
                f"Stockout in: <b>{r['days_to_stockout']:.0f}d</b></small>"
                f"</div>", unsafe_allow_html=True
            )

    st.markdown("---")
    st.markdown("#### 🟡 MEDIUM – Monitor Closely")
    med = inv_risk[inv_risk["risk_level"]=="MEDIUM"]
    if med.empty:
        st.success("✅ No medium-risk items.")
    else:
        display_cols = ["Product_Name","Category","Current_Stock","Reorder_Point",
                        "days_to_stockout","days_to_expiry","risk_reason"]
        st.dataframe(med[display_cols].reset_index(drop=True), use_container_width=True)

    st.markdown("---")
    st.markdown("#### 💀 Dead Stock Warning")
    dead = inv_risk[inv_risk["is_dead_stock"]]
    if dead.empty:
        st.success("✅ No dead stock detected.")
    else:
        dead_display = dead[["Product_Name","Category","Current_Stock",
                              "Cost_Price","Selling_Price"]].copy()
        dead_display["Capital_at_Risk"] = (dead_display["Current_Stock"] * dead_display["Cost_Price"]).round(2)
        dead_display = dead_display.sort_values("Capital_at_Risk", ascending=False)
        st.warning(f"⚠️  {len(dead)} product(s) with **no sales in {dead_stock_days}+ days** "
                   f"— ${dead_display['Capital_at_Risk'].sum():,.2f} capital at risk")
        st.dataframe(dead_display.reset_index(drop=True), use_container_width=True)


# ───────────────────────────────────────────────────────────────────────────
#  TAB 3 – FORECAST VIEW
# ───────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader(f"📈 {forecast_horizon}-Day Demand Forecast")

    if forecast.empty:
        st.warning("Not enough sales history to generate forecasts.")
    else:
        # Selector
        prod_names = sorted(forecast["Product_Name"].unique())
        selected   = st.selectbox("Choose a product to inspect:", prod_names)

        pid_sel  = inventory_raw.loc[inventory_raw["Product_Name"]==selected,"Product_ID"].values[0]
        hist_sel = sales_raw[sales_raw["Product_ID"]==pid_sel].copy()
        hist_sel["Date"] = pd.to_datetime(hist_sel["Date"])
        fc_sel   = forecast[forecast["Product_Name"]==selected].copy()
        fc_sel["Date"] = pd.to_datetime(fc_sel["Date"])

        fig_fc = go.Figure()
        # Historical (last 30 days)
        hist30 = hist_sel[hist_sel["Date"] >= (datetime.now()-timedelta(days=30))]
        fig_fc.add_trace(go.Scatter(
            x=hist30["Date"], y=hist30["Units_Sold"],
            mode="lines+markers", name="Historical Sales",
            line=dict(color="#60a5fa",width=2)
        ))
        # Forecast
        fig_fc.add_trace(go.Scatter(
            x=fc_sel["Date"], y=fc_sel["Forecast_Units"],
            mode="lines+markers", name="Forecast",
            line=dict(color="#f59e0b",width=2,dash="dash"),
            marker=dict(symbol="diamond",size=8)
        ))
        # Shaded forecast region
        fig_fc.add_vrect(
            x0=fc_sel["Date"].min(), x1=fc_sel["Date"].max(),
            fillcolor="#f59e0b", opacity=0.06, line_width=0
        )
        fig_fc.update_layout(
            title=f"Sales Forecast — {selected}",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", legend_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True,gridcolor="#1e293b"),
            margin=dict(t=50,b=0,l=0,r=0), height=380
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        # Forecast table with event column
        fc_display = fc_sel[["Date","Forecast_Units","Event"]].copy()
        fc_display["Date"] = fc_display["Date"].dt.strftime("%a %d %b")
        st.dataframe(fc_display.reset_index(drop=True), use_container_width=True)

        st.markdown("---")
        # Aggregate forecast bar
        st.subheader("📊 Aggregate Forecast – All Products")
        agg_fc = forecast.groupby("Date")["Forecast_Units"].sum().reset_index()
        agg_fc["Date"] = pd.to_datetime(agg_fc["Date"])
        fig_agg = px.bar(agg_fc, x="Date", y="Forecast_Units",
                         title=f"Total Units Forecasted per Day (All Products)",
                         color="Forecast_Units",
                         color_continuous_scale=["#1e3a5f","#3b82f6","#93c5fd"])
        fig_agg.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", coloraxis_showscale=False,
            margin=dict(t=50,b=0,l=0,r=0)
        )
        st.plotly_chart(fig_agg, use_container_width=True)


# ───────────────────────────────────────────────────────────────────────────
#  TAB 4 – RESTOCK PLANNER
# ───────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("🛒 Restock Planner – Actionable Orders")

    if restock_df.empty:
        st.success("✅ All products are sufficiently stocked. No orders needed!")
    else:
        total_order_cost = restock_df["estimated_cost"].sum()
        n_orders = len(restock_df)
        st.info(f"📋 **{n_orders} products** need restocking — "
                f"Estimated total order cost: **${total_order_cost:,.2f}**")

        st.markdown("#### 📌 Action Items")
        for _, r in restock_df.iterrows():
            rl = r['risk_level']
            badge = f"<span class='badge-{rl.lower()}'>{rl}</span>"
            st.markdown(
                f"<div class='action-box'>"
                f"{badge} &nbsp; {r['action_text']}"
                f"</div>", unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown("#### 📊 Order Summary Table")
        order_table = restock_df[[
            "Product_Name","Category","Supplier","Current_Stock",
            "order_qty","estimated_cost","order_by_date","risk_level"
        ]].rename(columns={
            "order_qty":"Order Qty",
            "estimated_cost":"Est. Cost ($)",
            "order_by_date":"Order By",
            "risk_level":"Priority"
        })
        st.dataframe(order_table.reset_index(drop=True), use_container_width=True)

        # Download button
        csv_buf = io.StringIO()
        order_table.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇️  Download Restock Order (CSV)",
            csv_buf.getvalue(),
            file_name=f"restock_order_{datetime.today().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

        # Cost by category bar
        cost_cat = restock_df.groupby("Category")["estimated_cost"].sum().reset_index()
        fig_cost = px.bar(cost_cat, x="Category", y="estimated_cost",
                          color="estimated_cost",
                          color_continuous_scale=["#1e3a5f","#3b82f6"],
                          title="Restock Cost Breakdown by Category",
                          labels={"estimated_cost":"Est. Cost ($)"})
        fig_cost.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", coloraxis_showscale=False,
            margin=dict(t=50,b=0,l=0,r=0)
        )
        st.plotly_chart(fig_cost, use_container_width=True)


# ───────────────────────────────────────────────────────────────────────────
#  TAB 5 – DATA EXPLORER
# ───────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("🔍 Full Inventory Data Explorer")

    risk_filter = st.multiselect("Filter by Risk Level",
                                 ["CRITICAL","HIGH","MEDIUM","LOW","OK"],
                                 default=["CRITICAL","HIGH","MEDIUM","LOW","OK"])
    display = inv_risk[inv_risk["risk_level"].isin(risk_filter)].copy()

    def _color_risk(val):
        palette = {"CRITICAL":"background-color:#450a0a;color:#fca5a5",
                   "HIGH":    "background-color:#431407;color:#fdba74",
                   "MEDIUM":  "background-color:#422006;color:#fde68a",
                   "LOW":     "background-color:#052e16;color:#86efac",
                   "OK":      "background-color:#172554;color:#93c5fd"}
        return palette.get(val,"")

    cols_show = ["Product_ID","Product_Name","Category","Current_Stock",
                 "Reorder_Point","Daily_Sales_Avg","days_to_stockout",
                 "days_to_expiry","risk_level","risk_reason"]
    styled = (display[cols_show]
              .reset_index(drop=True)
              .style.applymap(_color_risk, subset=["risk_level"]))
    st.dataframe(styled, use_container_width=True, height=520)

    st.markdown("---")
    st.subheader("📅 Sales History (Last 30 Days)")
    sales_raw["Date"] = pd.to_datetime(sales_raw["Date"])
    recent_sales = sales_raw[sales_raw["Date"] >= (datetime.now() - timedelta(days=30))]
    daily_total  = recent_sales.groupby("Date")[["Units_Sold","Revenue"]].sum().reset_index()

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Bar(x=daily_total["Date"], y=daily_total["Revenue"],
                               name="Revenue ($)", marker_color="#3b82f6", yaxis="y"))
    fig_trend.add_trace(go.Scatter(x=daily_total["Date"], y=daily_total["Units_Sold"],
                                   name="Units Sold", line=dict(color="#f59e0b",width=2),
                                   yaxis="y2"))
    fig_trend.update_layout(
        title="30-Day Sales Trend (All Products)",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0", legend_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(title="Revenue ($)", showgrid=True, gridcolor="#1e293b"),
        yaxis2=dict(title="Units Sold", overlaying="y", side="right", showgrid=False),
        margin=dict(t=50,b=0,l=0,r=0), height=380
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # Download full inventory risk
    inv_csv = io.StringIO()
    inv_risk.to_csv(inv_csv, index=False)
    st.download_button(
        "⬇️  Download Full Inventory Report (CSV)",
        inv_csv.getvalue(),
        file_name=f"inventory_report_{datetime.today().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
