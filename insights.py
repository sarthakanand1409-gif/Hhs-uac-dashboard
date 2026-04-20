import pandas as pd
import numpy as np
from datetime import datetime

kpi = pd.read_csv("day3_kpi.csv", parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
forecast = pd.read_csv("forecast_output.csv", parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
ml_preds = pd.read_csv("ml_predictions.csv", parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)

try:
    comparison = pd.read_csv("model_comparison.csv")
except FileNotFoundError:
    comparison = None

date_min, date_max = kpi["Date"].min(), kpi["Date"].max()
total_days = len(kpi)

peak_row = kpi.loc[kpi["total_load"].idxmax()]
low_row = kpi.loc[kpi["total_load"].idxmin()]
avg_load = kpi["total_load"].mean()
median_load = kpi["total_load"].median()
std_load = kpi["total_load"].std()

stress_days = int(kpi["stress_flag"].sum())
stress_pct = stress_days / total_days * 100

cbp_days = int((kpi["bottleneck"] == "CBP").sum())
hhs_days = int((kpi["bottleneck"] == "HHS").sum())
cbp_pct = cbp_days / total_days * 100
hhs_pct = hhs_days / total_days * 100

avg_pressure = kpi["pressure_index"].mean()
avg_stability = kpi["stability_score"].mean()
peak_pressure = kpi["pressure_index"].max()
peak_pressure_date = kpi.loc[kpi["pressure_index"].idxmax(), "Date"]

total_intake = kpi["transfers"].sum()
total_discharge = kpi["discharges"].sum()
net_flow = total_intake - total_discharge
final_backlog = kpi["backlog"].iloc[-1]

avg_discharge_ratio = kpi["discharge_ratio"].mean()
days_below_1 = int((kpi["discharge_ratio"] < 1).sum())

last_30 = kpi.tail(30)
prev_30 = kpi.iloc[-60:-30] if len(kpi) >= 60 else kpi.head(30)
recent_avg = last_30["total_load"].mean()
prev_avg = prev_30["total_load"].mean()
recent_trend = (recent_avg - prev_avg) / prev_avg * 100 if prev_avg else 0

future_only = forecast[forecast["Date"] > date_max]
forecast_peak = future_only["yhat"].max()
forecast_avg = future_only["yhat"].mean()
forecast_start = future_only["yhat"].iloc[0] if len(future_only) else np.nan
forecast_end = future_only["yhat"].iloc[-1] if len(future_only) else np.nan
forecast_direction = "upward" if forecast_end > forecast_start else "downward"
forecast_change_pct = (forecast_end - forecast_start) / forecast_start * 100 if forecast_start else 0

ml_mae = np.mean(np.abs(ml_preds["actual"] - ml_preds["predicted"]))
ml_rmse = np.sqrt(np.mean((ml_preds["actual"] - ml_preds["predicted"]) ** 2))
ml_mape = np.mean(np.abs((ml_preds["actual"] - ml_preds["predicted"]) / ml_preds["actual"])) * 100
baseline_mae = ml_preds["actual"].std()
improvement = (1 - ml_mae / baseline_mae) * 100

kpi["year"] = kpi["Date"].dt.year
yearly = kpi.groupby("year")["total_load"].agg(["mean", "max", "min"]).round(0)

kpi["month"] = kpi["Date"].dt.month
monthly_avg = kpi.groupby("month")["total_load"].mean().round(0)
peak_month = monthly_avg.idxmax()
low_month = monthly_avg.idxmin()
month_names = {1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",
               7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"}

report = f"""# HHS Unaccompanied Alien Children Program
## Data Analytics & Forecasting — Final Report

**Report generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Data coverage:** {date_min.date()} → {date_max.date()} ({total_days:,} days)

---

## Executive Summary

This report analyzes the operational pipeline for the HHS Unaccompanied Alien Children (UAC) Program — tracking the flow from CBP intake through HHS discharge. Across {total_days:,} days of data, the system processed a total of {int(total_intake):,} transfers into HHS care and {int(total_discharge):,} discharges, with an average daily total load (CBP custody + HHS care) of **{avg_load:,.0f} children**.

The system spent **{stress_pct:.1f}%** of days under operational stress. Bottleneck analysis shows the **{"CBP" if cbp_pct > hhs_pct else "HHS"}** stage was the dominant constraint, accounting for **{max(cbp_pct, hhs_pct):.1f}%** of days. Over the 30-day forecast horizon, total load is projected to trend **{forecast_direction}** by **{forecast_change_pct:+.1f}%**.

---

## 1. Historical Performance

### Load Statistics
| Metric | Value |
|---|---|
| Peak load | {int(peak_row['total_load']):,} children on {peak_row['Date'].date()} |
| Minimum load | {int(low_row['total_load']):,} children on {low_row['Date'].date()} |
| Average daily load | {avg_load:,.0f} |
| Median daily load | {median_load:,.0f} |
| Standard deviation | {std_load:,.0f} |
| Final backlog | {int(final_backlog):,} (cumulative net flow) |

### Recent Trend (last 30 days vs prior 30 days)
- Recent 30-day average: **{recent_avg:,.0f}**
- Prior 30-day average: **{prev_avg:,.0f}**
- Change: **{recent_trend:+.1f}%**

### Seasonality
- Historically heaviest month: **{month_names[peak_month]}** (avg {monthly_avg[peak_month]:,.0f})
- Historically lightest month: **{month_names[low_month]}** (avg {monthly_avg[low_month]:,.0f})

### Year-over-Year
{yearly.to_markdown()}

---

## 2. Operational Health

| KPI | Value |
|---|---|
| Average pressure index | {avg_pressure:.1f} / 100 |
| Average stability score | {avg_stability:.1f} / 100 |
| Peak pressure | {peak_pressure:.1f} on {peak_pressure_date.date()} |
| Stress days | {stress_days} ({stress_pct:.1f}% of total) |
| Avg discharge-to-transfer ratio | {avg_discharge_ratio:.2f} |
| Days where discharges lagged transfers | {days_below_1} ({days_below_1/total_days*100:.1f}%) |

**Interpretation:** A discharge ratio below 1.0 means children are entering HHS care faster than they're being released, directly increasing backlog. This occurred on **{days_below_1/total_days*100:.0f}%** of days in the dataset.

---

## 3. Bottleneck Analysis

| Stage | Days as bottleneck | % of total |
|---|---|---|
| CBP Custody | {cbp_days:,} | {cbp_pct:.1f}% |
| HHS Care | {hhs_days:,} | {hhs_pct:.1f}% |

**Finding:** The **{"CBP intake stage" if cbp_pct > hhs_pct else "HHS care stage"}** is the system's primary constraint. {"This suggests border apprehension volume is the limiting factor — downstream HHS capacity is generally able to absorb transfers." if cbp_pct > hhs_pct else "This suggests HHS shelter capacity is the limiting factor — children are accumulating downstream faster than they can be released to sponsors."}

---

## 4. Forecast (Next 30 Days — Prophet)

| Metric | Value |
|---|---|
| Forecast start | {future_only['Date'].iloc[0].date() if len(future_only) else 'N/A'} |
| Forecast end | {future_only['Date'].iloc[-1].date() if len(future_only) else 'N/A'} |
| Peak forecast | {forecast_peak:,.0f} |
| Average forecast | {forecast_avg:,.0f} |
| Starting load | {forecast_start:,.0f} |
| Ending load | {forecast_end:,.0f} |
| Direction | **{forecast_direction.upper()}** ({forecast_change_pct:+.1f}%) |

**Model:** Facebook Prophet with weekly and yearly seasonality, changepoint prior scale = 0.1.

---

## 5. Machine Learning Validation

### Winner Model Performance (90-day holdout)
| Metric | Value |
|---|---|
| MAE | {ml_mae:.2f} |
| RMSE | {ml_rmse:.2f} |
| MAPE | {ml_mape:.2f}% |
| Improvement vs naive baseline | {improvement:.1f}% |

"""

if comparison is not None:
    report += f"### Model Comparison (GridSearchCV with TimeSeriesSplit)\n\n{comparison.to_markdown(index=False)}\n\n"

report += f"""**Features used:** `lag_1`, `lag_7`, `lag_14`, `net_intake`, `rolling_7`, `rolling_14`
**Validation strategy:** 5-fold expanding-window time-series cross-validation on the training set; final evaluation on untouched last 90 days.

---

## 6. Key Findings

1. **System operates near capacity {stress_pct:.0f}% of the time.** With {stress_days} stress days out of {total_days}, capacity headroom is consistently tight.
2. **{"CBP" if cbp_pct > hhs_pct else "HHS"} is the dominant bottleneck** ({max(cbp_pct, hhs_pct):.0f}% of days). {"Investment in border processing capacity would have the largest system-wide impact." if cbp_pct > hhs_pct else "Investment in HHS shelter capacity and sponsor-matching throughput would have the largest system-wide impact."}
3. **Discharge throughput lags intake on {days_below_1/total_days*100:.0f}% of days**, which is the direct driver of backlog growth.
4. **Recent trend is {("accelerating" if recent_trend > 5 else "decelerating" if recent_trend < -5 else "stable")}** ({recent_trend:+.1f}% vs prior 30 days).
5. **Forward outlook:** Prophet projects a **{forecast_direction}** 30-day trajectory ({forecast_change_pct:+.1f}%), peaking at {forecast_peak:,.0f}.
6. **ML model reliability:** MAPE of {ml_mape:.1f}% means short-term predictions are within ±{ml_mape:.0f}% on average — suitable for operational planning.

---

## 7. Recommendations

- **Operational:** Monitor pressure index daily; escalate when it exceeds 70. The gauge in the dashboard highlights this automatically.
- **Capacity planning:** Prioritize {"CBP intake processing" if cbp_pct > hhs_pct else "HHS shelter expansion and discharge throughput"} based on the bottleneck analysis.
- **Forecasting cadence:** Retrain Prophet weekly and RF/XGB monthly as new data arrives; the existing `.pkl` pipeline supports this with no code changes.
- **Alerting:** Trigger stress-day alerts when `total_load` crosses the 90th percentile ({kpi['total_load'].quantile(0.90):,.0f}) OR `volatility` exceeds {kpi['volatility'].quantile(0.90):,.0f}.

---

## 8. Technical Stack

| Component | Tool |
|---|---|
| Data processing | pandas, numpy |
| Visualization | matplotlib |
| Time-series forecast | Prophet |
| ML forecast | scikit-learn Random Forest, XGBoost |
| Hyperparameter tuning | GridSearchCV + TimeSeriesSplit |
| Dashboard | Streamlit |
| Persistence | joblib (.pkl), CSV |

### Pipeline Artifacts
- `day1_cleaned.csv` → cleaned daily series
- `day2_enhanced.csv` → + rolling features, volatility, stress flags
- `day3_kpi.csv` → + pressure, backlog, bottleneck, stability
- `forecast_output.csv` → Prophet 30-day forecast with CI bounds
- `ml_predictions.csv` → Actual vs predicted (90-day holdout)
- `model_comparison.csv` → RF vs XGB scorecard
- `prophet_model.pkl`, `rf_model.pkl`, `all_models.pkl` → serialized models

---

## 9. Project Timeline

| Day | Deliverable |
|---|---|
| 1 | Data loading, cleaning, `total_load` + `net_intake` |
| 2 | Rolling averages, growth rate, volatility, stress flags |
| 3 | Discharge ratio, backlog, pressure index, bottleneck detection, stability score |
| 4A | Prophet forecasting (weekly + yearly seasonality, 30-day horizon) |
| 4B | RF vs XGBoost with GridSearchCV + TimeSeriesSplit |
| 5 | Streamlit dashboard (baseline) |
| 6 | Dashboard polish: sidebar filters, gauge, deltas, downloads |
| 7 | Final insights report (this document) |

---

*End of report.*
"""

with open("insights_report.md", "w") as f:
    f.write(report)

print("✅ Report saved to insights_report.md")
print(f"📊 {total_days:,} days analyzed | {stress_days} stress days | Winner MAE = {ml_mae:.2f}")
print(f"🔮 Forecast direction: {forecast_direction} ({forecast_change_pct:+.1f}%)")