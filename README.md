# UAC Program — Operations Dashboard

An interactive operations dashboard for the HHS Unaccompanied Alien Children (UAC) Program. This application is built with Streamlit and provides insights, machine learning forecasts, and data validations for operations tracking.

## Features

- **Pipeline Overview:** Interactive charts showing pipeline flow across total loads, CBP custody, and HHS care.
- **Key Performance Indicators (KPIs):** View growth rates, bottleneck stages, backlogs, pressure index, and stability scores.
- **Forecasting:** See total load forecasts powered by Prophet models with adjustable forecast horizons.
- **ML Validation:** Check Actual vs Predicted load metrics by comparing Random Forest (RF) and XGBoost models against real data.
- **Data Filtering:** Extensive filtering by date range and bottleneck stage, with quick stats download.

## Project Structure

- `app.py`: The main Streamlit dashboard application script.
- `requirements.txt`: Python package dependencies needed to run the app.
- `*.csv` (e.g. `day3_kpi.csv`, `forecast_output.csv`): Real-time metrics and datasets.
- `*.pkl` (e.g. `prophet_model.pkl`, `rf_model.pkl`): Serialized, pre-trained forecasting models.
- `Project1.ipynb`: Original Jupyter Notebook containing the data exploration and model training logic.

## Prerequisites & Installation

To run this dashboard locally, ensure you have Python 3 installed. Then, follow these steps:

1. **Clone or Download the Repository:**
   ```bash
   git clone <your-repo-url>
   cd intern_project
   ```

2. **Create a Virtual Environment (Recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On MacOS/Linux
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Start the Streamlit application by running the following command in your terminal:

```bash
streamlit run app.py
```

This will automatically launch the dashboard in your default web browser (typically accessible at `http://localhost:8501`).

## Contributing
Feel free to open issues or submit pull requests for potential feature additions and optimizations.
