# Mongolbank Macro Data Automation & Dashboard

## 📌 Overview
This project automates macroeconomic data collection from Mongolbank,
stores it in Google BigQuery, and visualizes it using an interactive Streamlit dashboard.

## 🧱 Architecture
- **data_automation.py**: ETL pipeline (API → BigQuery)
- **GitHub Actions**: Scheduled automation
- **BigQuery**: Central data warehouse
- **Streamlit**: Interactive dashboard

## 🚀 How it works
1. GitHub Actions runs `data_automation.py` daily
2. Data is appended / updated in BigQuery
3. Streamlit app (`app.py`) queries BigQuery
4. Dashboard is available via public URL

## 🛠 Tech Stack
- Python
- Google BigQuery
- GitHub Actions
- Streamlit

## 📊 Dashboard
The dashboard supports:
- Indicator filtering
- Time-series visualization
- QoQ / YoY analysis (planned)

---
