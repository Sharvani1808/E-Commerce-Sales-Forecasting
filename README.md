# E-Commerce Sales & Demand Forecasting

## 📌 Project Overview

This project analyzes historical e-commerce transaction data and uses time-series forecasting techniques to predict future sales.

The project focuses on analyzing daily sales trends, building forecasting models, evaluating their performance, comparing models, and generating a 30-day future sales forecast.

## 🎯 Objectives

- Clean and preprocess e-commerce transaction data
- Analyze daily sales patterns
- Build time-series forecasting models
- Compare ARIMA and Exponential Smoothing
- Evaluate forecasting performance using MAE and RMSE
- Generate a 30-day future sales forecast
- Save forecasting results as CSV files
- Visualize historical and forecasted sales

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Statsmodels
- OpenPyXL
- VS Code

## 📊 Dataset

The project uses the Online Retail dataset containing e-commerce transaction records.

Important fields include:

- Invoice Number
- Product Description
- Quantity
- Invoice Date
- Unit Price
- Customer ID
- Country

The transaction-level data was cleaned and aggregated into daily total sales for time-series forecasting.

## 🔄 Project Workflow

```text
Raw Dataset
     ↓
Data Cleaning
     ↓
Daily Sales Aggregation
     ↓
Exploratory Data Analysis
     ↓
Train-Test Split
     ↓
ARIMA Model
     ↓
Exponential Smoothing Model
     ↓
Model Evaluation
     ↓
Model Comparison
     ↓
30-Day Future Forecast
     ↓
Visualization & Results

## 📈 Model Results

The forecasting models were evaluated using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

| Metric | Result |
|---|---:|
| MAE | 22,129.47 |
| RMSE | 31,111.03 |
| Average Actual Daily Sales | 46,712.45 |
| Forecast Accuracy | 52.63% |

The project also generated a 30-day future sales forecast using the trained time-series model.

## 📁 Project Files

- `analysis.py` – Main data analysis and forecasting code
- `30_day_sales_forecast.csv` – 30-day future sales predictions
- `arima_forecast_results.csv` – ARIMA forecasting results
- `model_comparison.csv` – Model performance comparison
- `project_summary.csv` – Project summary
- `requirements.txt` – Python dependencies
- `data/` – Dataset files

## 🚀 How to Run

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install the required libraries:

```bash
pip install -r requirements.txt
 👩‍💻 Author
Sharvani