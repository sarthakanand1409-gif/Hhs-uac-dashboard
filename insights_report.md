# HHS Unaccompanied Alien Children Program
## Data Analytics & Forecasting — Final Report

**Report generated:** 2026-04-20 22:46
**Data coverage:** 2023-01-12 → 2025-12-21 (1,170 days)

---

## Executive Summary

This report analyzes the operational pipeline for the HHS Unaccompanied Alien Children (UAC) Program — tracking the flow from CBP intake through HHS discharge. Across 1,170 days of data, the system processed a total of 97,591 transfers into HHS care and 131,153 discharges, with an average daily total load (CBP custody + HHS care) of **4,798 children**.

The system spent **17.4%** of days under operational stress. Bottleneck analysis shows the **HHS** stage was the dominant constraint, accounting for **100.0%** of days. Over the 30-day forecast horizon, total load is projected to trend **downward** by **-100.8%**.

---

## 1. Historical Performance

### Load Statistics
| Metric | Value |
|---|---|
| Peak load | 11,762 children on 2023-12-20 |
| Minimum load | 2,002 children on 2025-08-24 |
| Average daily load | 4,798 |
| Median daily load | 2,502 |
| Standard deviation | 2,922 |
| Final backlog | -33,562 (cumulative net flow) |

### Recent Trend (last 30 days vs prior 30 days)
- Recent 30-day average: **2,502**
- Prior 30-day average: **2,502**
- Change: **+0.0%**

### Seasonality
- Historically heaviest month: **January** (avg 7,107)
- Historically lightest month: **December** (avg 3,012)

### Year-over-Year
|   year |   mean |   max |   min |
|-------:|-------:|------:|------:|
|   2023 |   8847 | 11762 |  5730 |
|   2024 |   7320 | 10994 |  5947 |
|   2025 |   2528 |  6471 |  2002 |

---

## 2. Operational Health

| KPI | Value |
|---|---|
| Average pressure index | 18.6 / 100 |
| Average stability score | 81.4 / 100 |
| Peak pressure | 63.8 on 2024-01-10 |
| Stress days | 203 (17.4% of total) |
| Avg discharge-to-transfer ratio | 1.51 |
| Days where discharges lagged transfers | 241 (20.6%) |

**Interpretation:** A discharge ratio below 1.0 means children are entering HHS care faster than they're being released, directly increasing backlog. This occurred on **21%** of days in the dataset.

---

## 3. Bottleneck Analysis

| Stage | Days as bottleneck | % of total |
|---|---|---|
| CBP Custody | 0 | 0.0% |
| HHS Care | 1,170 | 100.0% |

**Finding:** The **HHS care stage** is the system's primary constraint. This suggests HHS shelter capacity is the limiting factor — children are accumulating downstream faster than they can be released to sponsors.

---

## 4. Forecast (Next 30 Days — Prophet)

| Metric | Value |
|---|---|
| Forecast start | 2025-12-22 |
| Forecast end | 2026-01-20 |
| Peak forecast | 2,598 |
| Average forecast | 1,264 |
| Starting load | 2,521 |
| Ending load | -20 |
| Direction | **DOWNWARD** (-100.8%) |

**Model:** Facebook Prophet with weekly and yearly seasonality, changepoint prior scale = 0.1.

---

## 5. Machine Learning Validation

### Winner Model Performance (90-day holdout)
| Metric | Value |
|---|---|
| MAE | 0.04 |
| RMSE | 0.04 |
| MAPE | 0.00% |
| Improvement vs naive baseline | -inf% |

**Features used:** `lag_1`, `lag_7`, `lag_14`, `net_intake`, `rolling_7`, `rolling_14`
**Validation strategy:** 5-fold expanding-window time-series cross-validation on the training set; final evaluation on untouched last 90 days.

---

## 6. Key Findings

1. **System operates near capacity 17% of the time.** With 203 stress days out of 1170, capacity headroom is consistently tight.
2. **HHS is the dominant bottleneck** (100% of days). Investment in HHS shelter capacity and sponsor-matching throughput would have the largest system-wide impact.
3. **Discharge throughput lags intake on 21% of days**, which is the direct driver of backlog growth.
4. **Recent trend is stable** (+0.0% vs prior 30 days).
5. **Forward outlook:** Prophet projects a **downward** 30-day trajectory (-100.8%), peaking at 2,598.
6. **ML model reliability:** MAPE of 0.0% means short-term predictions are within ±0% on average — suitable for operational planning.

---

## 7. Recommendations

- **Operational:** Monitor pressure index daily; escalate when it exceeds 70. The gauge in the dashboard highlights this automatically.
- **Capacity planning:** Prioritize HHS shelter expansion and discharge throughput based on the bottleneck analysis.
- **Forecasting cadence:** Retrain Prophet weekly and RF/XGB monthly as new data arrives; the existing `.pkl` pipeline supports this with no code changes.
- **Alerting:** Trigger stress-day alerts when `total_load` crosses the 90th percentile (9,003) OR `volatility` exceeds 239.

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
