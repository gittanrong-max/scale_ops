# Forecasting Portfolio App

A simple Streamlit app for product management portfolio use: upload a CSV, paste series data, or manually edit a table with a date/time column and numeric values, then forecast the next period.

## Features

- CSV upload, text paste, or manual table entry
- Date/time plus numeric series input
- Forecast horizon input with default `12 months`
- Models: Exponential smoothing, linear trend, naive last-value
- Interactive Plotly chart and downloadable CSV output

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run forecast_app.py
```

## Data format

The CSV should include at least two columns: one date/time column and one numeric column, for example:

```csv
date,value
2024-01-01,100
2024-02-01,115
2024-03-01,125
```
