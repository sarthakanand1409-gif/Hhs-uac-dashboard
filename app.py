import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib

st.set_page_config(
    page_title="UAC Program — Operations Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊",
)

st.markdown("""
<style>
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
[data-testid="stMetricValue"] { font-size: 1.6rem; }
[data-testid="stMetricLabel"] { font-size: 0.85rem; color: #6b7280; }
div[data-testid="stSidebar"] { background-color: #f8f9fb; }
h1, h2, h3 { color: #1f2937; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] { padding: 8px 18px; border-radius: 8px 8px 0 0; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_kpi():
    return pd.read_csv("day3_kpi.csv", parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)

@st.cache_data
def load_forecast():
    return pd.read_csv("forecast_output.csv", parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)

@st.cache_data
def load_ml_predictions():
    return pd.read_csv("ml_predictions.csv", parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)

@st.cache_data
def load_model_comparison():
    try:
        return pd.read_csv("model_comparison.csv")
    except FileNotFoundError:
        return None

@st.cache_resource
def load_models():
    prophet_model = joblib.load("prophet_model.pkl")
    rf_model = joblib.load("rf_model.pkl")
    return prophet_model, rf_model

kpi_full = load_kpi()
forecast = load_forecast()
ml_preds = load_ml_predictions()
comparison = load_model_comparison()
prophet_model, rf_model = load_models()

min_date, max_date = kpi_full["Date"].min().date(), kpi_full["Date"].max().date()

with st.sidebar:
    st.title("⚙️ Filters")

    preset = st.radio(
        "Quick range",
        ["All time", "Last 30 days", "Last 90 days", "Last 6 months", "Last 1 year", "Custom"],
        index=2,
    )

    if preset == "All time":
        start_default, end_default = min_date, max_date
    elif preset == "Last 30 days":
        start_default, end_default = max_date - pd.Timedelta(days=30), max_date
    elif preset == "Last 90 days":
        start_default, end_default = max_date - pd.Timedelta(days=90), max_date
    elif preset == "Last 6 months":
        start_default, end_default = max_date - pd.Timedelta(days=180), max_date
    elif preset == "Last 1 year":
        start_default, end_default = max_date - pd.Timedelta(days=365), max_date
    else:
        start_default, end_default = max_date - pd.Timedelta(days=90), max_date

    date_range = st.date_input(
        "Date range",
        value=(start_default, end_default),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = start_default, end_default

    st.divider()

    bottleneck_filter = st.multiselect(
        "Bottleneck stage",
        options=sorted(kpi_full["bottleneck"].unique().tolist()),
        default=sorted(kpi_full["bottleneck"].unique().tolist()),
    )

    stress_only = st.checkbox("Show stress days only", value=False)

    st.divider()

    show_rolling = st.checkbox("Show rolling averages", value=True)
    forecast_horizon = st.slider("Forecast horizon (days)", 7, 60, 30)

    st.divider()
    st.caption(f"📅 Data range: {min_date} → {max_date}")
    st.caption(f"📊 Total records: {len(kpi_full):,}")

mask = (
    (kpi_full["Date"].dt.date >= start_date)
    & (kpi_full["Date"].dt.date <= end_date)
    & (kpi_full["bottleneck"].isin(bottleneck_filter))
)
if stress_only:
    mask &= (kpi_full["stress_flag"] == 1)
kpi = kpi_full.loc[mask].reset_index(drop=True)

st.title("🏛️ HHS Unaccompanied Alien Children Program")
st.caption("CBP Intake → CBP Custody → Transfer → HHS Care → Discharge")

if len(kpi) == 0:
    st.warning("No data matches the current filters. Widen the date range or adjust filters in the sidebar.")
    st.stop()

latest = kpi.iloc[-1]
prev = kpi.iloc[-2] if len(kpi) > 1 else latest

def delta(curr, prev_val):
    if prev_val == 0 or pd.isna(prev_val):
        return None
    return f"{(curr - prev_val) / prev_val * 100:+.1f}%"

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Load", f"{int(latest['total_load']):,}", delta(latest["total_load"], prev["total_load"]))
c2.metric("CBP Custody", f"{int(latest['cbp_custody']):,}", delta(latest["cbp_custody"], prev["cbp_custody"]))
c3.metric("HHS Care", f"{int(latest['hhs_care']):,}", delta(latest["hhs_care"], prev["hhs_care"]))
c4.metric("Pressure Index", f"{latest['pressure_index']:.1f}", delta(latest["pressure_index"], prev["pressure_index"]), delta_color="inverse")
c5.metric("Stability Score", f"{latest['stability_score']:.1f}", delta(latest["stability_score"], prev["stability_score"]))

s1, s2, s3 = st.columns(3)
stress_days = int(kpi["stress_flag"].sum())
avg_load = kpi["total_load"].mean()
cbp_pct = (kpi["bottleneck"] == "CBP").mean() * 100
s1.metric("Stress Days (in range)", f"{stress_days}")
s2.metric("Avg Total Load", f"{avg_load:,.0f}")
s3.metric("CBP-Bottleneck Days", f"{cbp_pct:.1f}%")

st.divider()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Overview", "📈 KPIs", "🔮 Forecast", "🤖 ML Validation", "📥 Data", "🔍 Insights"])

def style_axis(ax):
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))

with tab1:
    st.subheader("Pipeline Flow")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(kpi["Date"], kpi["cbp_custody"], label="CBP Custody", color="#c0392b", lw=1.8)
    ax.plot(kpi["Date"], kpi["hhs_care"], label="HHS Care", color="#2980b9", lw=1.8)
    ax.plot(kpi["Date"], kpi["total_load"], label="Total Load", color="black", lw=2)
    if show_rolling:
        ax.plot(kpi["Date"], kpi["rolling_7"], label="Rolling 7d", color="orange", ls="--", lw=1.2, alpha=0.8)
    ax.legend(loc="upper left", frameon=False)
    ax.set_ylabel("Children")
    style_axis(ax)
    st.pyplot(fig)

    colA, colB = st.columns(2)
    with colA:
        st.subheader("Intakes vs Discharges")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(kpi["Date"], kpi["transfers"], label="Transfers", color="green", lw=1.5)
        ax.plot(kpi["Date"], kpi["discharges"], label="Discharges", color="orange", lw=1.5)
        ax.fill_between(kpi["Date"], kpi["transfers"], kpi["discharges"],
                        where=(kpi["transfers"] > kpi["discharges"]),
                        alpha=0.15, color="red", label="Net +")
        ax.legend(frameon=False)
        style_axis(ax)
        st.pyplot(fig)

    with colB:
        st.subheader("Growth Rate")
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = ["green" if v >= 0 else "red" for v in kpi["growth_rate"]]
        ax.bar(kpi["Date"], kpi["growth_rate"], color=colors, width=1.0, alpha=0.7)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_ylabel("% change")
        style_axis(ax)
        st.pyplot(fig)

with tab2:
    colA, colB = st.columns([2, 1])

    with colA:
        st.subheader("Pressure vs Stability")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(kpi["Date"], kpi["pressure_index"], label="Pressure", color="#e74c3c", lw=1.8)
        ax.plot(kpi["Date"], kpi["stability_score"], label="Stability", color="#27ae60", lw=1.8)
        ax.fill_between(kpi["Date"], kpi["pressure_index"], alpha=0.15, color="#e74c3c")
        ax.legend(frameon=False)
        style_axis(ax)
        st.pyplot(fig)

    with colB:
        st.subheader("Current Pressure")
        pressure = float(latest["pressure_index"])
        fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={"projection": "polar"})
        theta = np.linspace(np.pi, 0, 100)
        ax.plot(theta, [1] * 100, color="#e5e7eb", lw=20, solid_capstyle="round")
        filled = np.linspace(np.pi, np.pi - (pressure / 100) * np.pi, 100)
        color = "#27ae60" if pressure < 40 else "#f39c12" if pressure < 70 else "#e74c3c"
        ax.plot(filled, [1] * 100, color=color, lw=20, solid_capstyle="round")
        ax.set_ylim(0, 1.2)
        ax.set_yticks([]); ax.set_xticks([])
        ax.spines["polar"].set_visible(False)
        ax.text(np.pi / 2, 0.3, f"{pressure:.0f}", ha="center", va="center", fontsize=28, fontweight="bold", color=color)
        ax.text(np.pi / 2, 0.05, "Pressure", ha="center", va="center", fontsize=10, color="#6b7280")
        st.pyplot(fig)

    st.subheader("Bottleneck & Backlog")
    colC, colD = st.columns(2)
    with colC:
        counts = kpi["bottleneck"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(counts.index, counts.values, color=["#c0392b", "#2980b9"])
        for i, v in enumerate(counts.values):
            ax.text(i, v, f"{v}", ha="center", va="bottom", fontweight="bold")
        ax.set_ylabel("Days")
        style_axis(ax)
        st.pyplot(fig)
    with colD:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(kpi["Date"], kpi["backlog"], color="darkorange", lw=1.8)
        ax.fill_between(kpi["Date"], kpi["backlog"], alpha=0.2, color="darkorange")
        ax.set_ylabel("Cumulative backlog")
        style_axis(ax)
        st.pyplot(fig)

with tab3:
    st.subheader(f"Prophet Forecast — next {forecast_horizon} days")

    history_tail = kpi_full[["Date", "total_load"]].tail(180)
    future_only = forecast[forecast["Date"] > kpi_full["Date"].max()].head(forecast_horizon)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(history_tail["Date"], history_tail["total_load"], label="Historical", color="black", lw=1.8)
    ax.plot(future_only["Date"], future_only["yhat"], label="Forecast", color="#e74c3c", lw=2)
    ax.fill_between(future_only["Date"], future_only["yhat_lower"], future_only["yhat_upper"],
                    color="#e74c3c", alpha=0.2, label="Confidence interval")
    ax.axvline(kpi_full["Date"].max(), color="gray", ls=":", alpha=0.7)
    ax.legend(loc="upper left", frameon=False)
    ax.set_ylabel("Total Load")
    style_axis(ax)
    st.pyplot(fig)

    colA, colB, colC = st.columns(3)
    colA.metric("Forecast start", future_only["Date"].iloc[0].strftime("%Y-%m-%d") if len(future_only) else "—")
    colB.metric("Peak forecast", f"{future_only['yhat'].max():,.0f}" if len(future_only) else "—")
    colC.metric("Avg forecast", f"{future_only['yhat'].mean():,.0f}" if len(future_only) else "—")

    with st.expander("📋 Forecast table"):
        st.dataframe(future_only.round(1), use_container_width=True)

with tab4:
    if comparison is not None:
        st.subheader("Model Comparison")
        st.dataframe(comparison, use_container_width=True)

    preds_in_range = ml_preds[(ml_preds["Date"].dt.date >= start_date) & (ml_preds["Date"].dt.date <= end_date)]
    if len(preds_in_range) == 0:
        preds_in_range = ml_preds

    mae = np.mean(np.abs(preds_in_range["actual"] - preds_in_range["predicted"]))
    rmse = np.sqrt(np.mean((preds_in_range["actual"] - preds_in_range["predicted"]) ** 2))
    mape = np.mean(np.abs((preds_in_range["actual"] - preds_in_range["predicted"]) / preds_in_range["actual"])) * 100

    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{mae:.2f}")
    m2.metric("RMSE", f"{rmse:.2f}")
    m3.metric("MAPE", f"{mape:.2f}%")

    st.subheader("Actual vs Predicted")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(preds_in_range["Date"], preds_in_range["actual"], label="Actual", color="black", lw=1.8)
    if "rf_predicted" in preds_in_range.columns:
        ax.plot(preds_in_range["Date"], preds_in_range["rf_predicted"], label="RF", color="steelblue", ls="--", alpha=0.8)
    if "xgb_predicted" in preds_in_range.columns:
        ax.plot(preds_in_range["Date"], preds_in_range["xgb_predicted"], label="XGB", color="#e74c3c", ls="--", alpha=0.8)
    ax.plot(preds_in_range["Date"], preds_in_range["predicted"], label="Winner", color="#27ae60", lw=2)
    ax.legend(frameon=False)
    style_axis(ax)
    st.pyplot(fig)

    st.subheader("Residuals")
    residuals = preds_in_range["actual"] - preds_in_range["predicted"]
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(preds_in_range["Date"], residuals, color="purple", lw=1.2)
    ax.axhline(0, color="black", lw=0.6)
    ax.fill_between(preds_in_range["Date"], residuals, 0, alpha=0.2, color="purple")
    style_axis(ax)
    st.pyplot(fig)

with tab5:
    st.subheader("Filtered KPI Data")
    st.dataframe(kpi, use_container_width=True, height=400)

    colA, colB, colC = st.columns(3)
    colA.download_button("⬇️ Download filtered KPIs", kpi.to_csv(index=False).encode(), "filtered_kpi.csv", "text/csv")
    colB.download_button("⬇️ Download forecast", forecast.to_csv(index=False).encode(), "forecast_output.csv", "text/csv")
    colC.download_button("⬇️ Download ML predictions", ml_preds.to_csv(index=False).encode(), "ml_predictions.csv", "text/csv")

with tab6:
    st.subheader("Final Insights Report")
    try:
        with open("insights_report.md", "r") as f:
            report_md = f.read()
        st.markdown(report_md)
        st.download_button("⬇️ Download report (.md)", report_md.encode(), "insights_report.md", "text/markdown")
    except FileNotFoundError:
        st.warning("Run `python generate_insights.py` first to generate the report.")

st.divider()
st.caption("Built with Streamlit · Prophet + RF/XGB ensemble · Data source: HHS UAC Program")