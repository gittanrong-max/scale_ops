import io
from datetime import datetime
from typing import cast

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(
    page_title="ScaleOps - Forecasting App",
    layout="wide",
    initial_sidebar_state="expanded",
)

HEADER = "Forecast Your Inventory, Headcount or Volume"
DESCRIPTION = (
    "Upload a CSV, paste tabular data, or enter values manually. "
    "Include a date/time column plus a numeric volume/headcount column, then predict the next period." 
)

st.title(HEADER)
st.write(DESCRIPTION)

with st.sidebar:
    st.header("How to use")
    st.markdown(
        "1. Upload a CSV file with a date/time column and one numeric column.\n"
        "2. Or paste CSV/text data into the box.\n"
        "3. Or edit the sample manual table directly.\n"
        "4. Choose your forecast horizon and model.\n"
        "5. Download the forecast CSV when ready."
    )
    st.markdown("---")
    st.markdown("**Default forecast horizon:** 12 months")


def parse_csv(file) -> pd.DataFrame:
    try:
        return pd.read_csv(file)
    except Exception:
        return pd.DataFrame()


def parse_text(text) -> pd.DataFrame:
    try:
        return pd.read_csv(io.StringIO(text))
    except Exception:
        return pd.DataFrame()


def infer_frequency(ts: pd.Series) -> str:
    try:
        freq = pd.infer_freq(ts)
        return freq if freq is not None else "unknown"
    except Exception:
        return "unknown"


def build_forecast(df, date_col, value_col, periods, model_name, seasonal_periods=None):
    series = df.set_index(date_col)[value_col].astype(float)
    if model_name == "Naive (repeat last)":
        forecast_index = pd.date_range(start=series.index[-1], periods=periods + 1, freq=series.index.freq or pd.infer_freq(series.index))
        if len(forecast_index) > 0:
            forecast_index = forecast_index[1:]
        return pd.Series([series.iloc[-1]] * periods, index=forecast_index, name="forecast")

    if model_name == "Linear trend":
        x = np.arange(len(series)).reshape(-1, 1)
        y = series.values.reshape(-1, 1)
        reg = LinearRegression().fit(x, y)
        future_x = np.arange(len(series), len(series) + periods).reshape(-1, 1)
        pred = reg.predict(future_x).flatten()
        freq = series.index.freq or pd.infer_freq(series.index)
        forecast_index = pd.date_range(start=series.index[-1], periods=periods + 1, freq=freq)[1:]
        return pd.Series(pred, index=forecast_index, name="forecast")

    if model_name == "Exponential smoothing":
        fit = ExponentialSmoothing(
            series,
            trend="add",
            seasonal="add" if seasonal_periods else None,
            seasonal_periods=seasonal_periods,
            initialization_method="estimated",
        ).fit(optimized=True)
        pred = fit.forecast(periods)
        return pred.rename("forecast")

    raise ValueError(f"Unknown model {model_name}")


def create_sample_data() -> pd.DataFrame:
    dates = pd.date_range(start="2023-01-01", periods=24, freq="M")
    values = np.round(np.linspace(120, 220, len(dates)) + np.random.normal(0, 8, len(dates)), 0)
    return pd.DataFrame({"date": dates, "value": values})


uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], accept_multiple_files=False)
source = None

if uploaded_file is not None:
    source = parse_csv(uploaded_file)
    st.success("CSV file loaded.")
else:
    raw_text = st.text_area(
        "Paste CSV text here",
        value="date,value\n2024-01-01,100\n2024-02-01,115\n2024-03-01,125\n",
        height=200,
    )
    if raw_text.strip():
        source = parse_text(raw_text)

if source is None or source.empty:
    st.info("No data loaded yet. Use upload, paste text, or edit the sample table below.")
    source = create_sample_data()

st.markdown("---")
st.subheader("Edit or confirm your series")
# Prefer the stable `data_editor` API, fall back to the experimental name, else show a warning
editor = getattr(st, "data_editor", None) or getattr(st, "experimental_data_editor", None)
if callable(editor):
    editable = cast(pd.DataFrame, editor(source, num_rows="dynamic"))
else:
    st.warning(
        "This Streamlit release does not include the interactive data editor. "
        "You can paste CSV into the text box above to edit your data."
    )
    editable = source

if editable is None or editable.empty:
    st.error("Please provide a table with a date column and a numeric value column.")
    st.stop()

editable_cols = editable.columns.tolist()
date_col = st.selectbox("Select date/time column", options=editable_cols, index=0)
value_col = st.selectbox("Select numeric column", options=[c for c in editable_cols if c != date_col], index=0 if len(editable_cols) > 1 else 0)

try:
    editable[date_col] = pd.to_datetime(editable[date_col])
except Exception:
    st.error("Could not parse the selected date/time column. Please use valid dates.")
    st.stop()

if not is_numeric_dtype(editable[value_col]):
    try:
        editable[value_col] = pd.to_numeric(editable[value_col], errors="raise")
    except Exception:
        st.error("The selected numeric column contains non-numeric values. Please clean the data.")
        st.stop()

editable = editable.sort_values(date_col)
editable = editable.reset_index(drop=True)

freq_label = infer_frequency(editable[date_col])
if freq_label == "unknown":
    st.warning("Could not infer a regular date frequency from the series. Forecasts may be less accurate.")
else:
    st.info(f"Detected frequency: {freq_label}")

horizon_units = st.selectbox("Forecast horizon unit", ["months", "weeks", "days", "years"], index=0)
horizon_value = st.number_input("Forecast horizon", min_value=1, max_value=60, value=12, step=1)
model_name = st.selectbox("Forecast model", ["Exponential smoothing", "Linear trend", "Naive (repeat last)"])
seasonal_periods = None
if model_name == "Exponential smoothing":
    if freq_label and freq_label != "unknown":
        if freq_label.startswith("M"):
            seasonal_periods = 12
        elif freq_label.startswith("W"):
            seasonal_periods = 52
        elif freq_label.startswith("D"):
            seasonal_periods = 7
        else:
            seasonal_periods = None
    seasonal_periods = st.number_input("Seasonal period (optional)", min_value=0, max_value=52, value=seasonal_periods or 0, step=1)
    if seasonal_periods == 0:
        seasonal_periods = None

if st.button("Run forecast"):
    if editable[date_col].duplicated().any():
        st.error("Duplicate dates were found. Each date/time value must be unique.")
    else:
        dates = editable[date_col]
        freq = pd.infer_freq(dates)
        if freq is None:
            if freq_label != "unknown":
                freq = freq_label
            else:
                st.warning("Using a generic daily frequency due to irregular dates.")
                freq = "D"

        unit_map = {"days": "D", "weeks": "W", "months": "M", "years": "Y"}
        requested_unit = unit_map[horizon_units]

        try:
            forecast_periods = horizon_value
            if requested_unit != freq[0]:
                if requested_unit == "M" and freq.startswith("M"):
                    forecast_periods = horizon_value
                elif requested_unit == "Y" and freq.startswith("M"):
                    forecast_periods = horizon_value * 12
                elif requested_unit == "W" and freq.startswith("D"):
                    forecast_periods = int(horizon_value * 7)
                elif requested_unit == "D" and freq.startswith("D"):
                    forecast_periods = horizon_value
                else:
                    forecast_periods = horizon_value
        except Exception:
            forecast_periods = horizon_value

        if forecast_periods <= 0:
            st.error("Forecast horizon must be positive.")
        else:
            try:
                forecast = build_forecast(editable, date_col, value_col, forecast_periods, model_name, seasonal_periods)
                history = editable.set_index(date_col)[value_col].astype(float)
                combined = pd.concat([history.rename("actual"), forecast], axis=1)

                st.subheader("Forecast results")
                fig = px.line(
                    combined.reset_index().melt(id_vars=date_col, var_name="series", value_name="value"),
                    x=date_col,
                    y="value",
                    color="series",
                    title="Actual vs Forecast",
                )
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(combined.tail(20).reset_index())

                csv_bytes = combined.reset_index().to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download forecast as CSV",
                    data=csv_bytes,
                    file_name="forecast_output.csv",
                    mime="text/csv",
                )
            except Exception as exc:
                st.error(f"Forecast failed: {exc}")
