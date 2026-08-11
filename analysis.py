import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_excel("data/Online Retail.xlsx")
print(df.head())
print(df.shape)
print(df.columns)
print(df.isnull().sum())
df = df.dropna(subset=["Description"])
print(df.shape)
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
df = df[df["Quantity"] > 0]
df = df[df["Quantity"] > 0]

df["TotalSales"] = df["Quantity"] * df["UnitPrice"]

print(df[["Quantity", "UnitPrice", "TotalSales"]].head())
daily_sales = df.groupby(df["InvoiceDate"].dt.date)["TotalSales"].sum()

print(daily_sales.head())
daily_sales = daily_sales.reset_index()
print(daily_sales.head())
daily_sales["InvoiceDate"] = pd.to_datetime(daily_sales["InvoiceDate"])

print(daily_sales.dtypes)
daily_sales = daily_sales.sort_values("InvoiceDate")

print(daily_sales.head())
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(daily_sales["InvoiceDate"], daily_sales["TotalSales"])
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.title("Daily Sales Trend")
plt.show()
daily_sales = daily_sales.set_index("InvoiceDate")

print(daily_sales.head())
print(daily_sales.index.to_series().diff().value_counts().head())
daily_sales = daily_sales.asfreq("D", fill_value=0)
print(daily_sales.head())
print(daily_sales.shape)
plt.figure(figsize=(12, 6))
plt.plot(daily_sales.index, daily_sales["TotalSales"])
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.title("Daily Sales Trend After Filling Missing Dates")
plt.show()
daily_sales["MovingAverage7"] = daily_sales["TotalSales"].rolling(window=7).mean()

print(daily_sales.head(10))
plt.figure(figsize=(12, 6))
plt.plot(daily_sales.index, daily_sales["TotalSales"], label="Daily Sales")
plt.plot(daily_sales.index, daily_sales["MovingAverage7"], label="7-Day Moving Average")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.title("Daily Sales vs 7-Day Moving Average")
plt.legend()
plt.show()
train_size = int(len(daily_sales) * 0.8)

train = daily_sales.iloc[:train_size]
test = daily_sales.iloc[train_size:]
print("Training data:", train.shape)
print("Testing data:", test.shape)
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(train["TotalSales"], order=(5, 1, 0))

model_fit = model.fit()

forecast = model_fit.forecast(steps=len(test))

print(forecast.head())

from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(test["TotalSales"], forecast)

mse = mean_squared_error(test["TotalSales"], forecast)

rmse = mse ** 0.5

print("MAE:", mae)
print("RMSE:", rmse)

plt.figure(figsize=(12, 6))
plt.plot(test.index, test["TotalSales"], label="Actual Sales")
plt.plot(test.index, forecast, label="Forecasted Sales")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.title("Actual vs Forecasted Sales")
plt.legend()
plt.show()

print("\n--- Model Performance ---")
print("MAE:", mae)
print("RMSE:", rmse)
model = ARIMA(train["TotalSales"], order=(5, 1, 0))

model_fit = model.fit()

forecast = model_fit.forecast(steps=len(test))

print("\nForecasted Sales:")
print(forecast.head())
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(test["TotalSales"], forecast)

mse = mean_squared_error(test["TotalSales"], forecast)

rmse = mse ** 0.5

print("\n--- Model Performance ---")
print("MAE:", mae)
print("RMSE:", rmse)
plt.figure(figsize=(12, 6))

plt.plot(test.index, test["TotalSales"], label="Actual Sales")

plt.plot(test.index, forecast, label="Forecasted Sales")

plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.title("Actual vs Forecasted Sales")

plt.legend()

plt.show()
average_sales = test["TotalSales"].mean()

print("Average Actual Daily Sales:", average_sales)
print("MAE:", mae)
print("RMSE:", rmse)
from sklearn.metrics import mean_absolute_error, mean_squared_error
error_percentage = (mae / average_sales) * 100

print("MAE Percentage:", error_percentage)
print("Average Actual Daily Sales:", average_sales)
print("MAE:", mae)
print("RMSE:", rmse)
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(test.index, test["TotalSales"], label="Actual Sales")
plt.plot(test.index, forecast, label="ARIMA Forecast")
plt.legend()
plt.title("Actual vs ARIMA Forecast")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.grid(True)
plt.show()
accuracy = 100 - error_percentage

print("Forecast Error:", round(error_percentage, 2), "%")
print("Forecast Accuracy:", round(accuracy, 2), "%")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
results = pd.DataFrame({
    "Actual": test["TotalSales"],
    "Forecast": forecast
})

results["Error"] = results["Actual"] - results["Forecast"]

print(results.head())
results["Absolute_Error"] = abs(results["Error"])

results["Percentage_Error"] = (
    results["Absolute_Error"] / results["Actual"]
) * 100

print(results.head())
mean_percentage_error = results["Percentage_Error"].mean()

print("Mean Percentage Error:", round(mean_percentage_error, 2), "%")
print("Mean Absolute Error:", round(mae, 2))
print("Root Mean Squared Error:", round(rmse, 2))
print("Number of Test Days:", len(test))
results_valid = results[results["Actual"] != 0].copy()

results_valid["Percentage_Error"] = (
    results_valid["Absolute_Error"] / results_valid["Actual"]
) * 100

mean_percentage_error = results_valid["Percentage_Error"].mean()
print("Mean Percentage Error:", round(mean_percentage_error, 2), "%")
zero_sales_days = (results["Actual"] == 0).sum()

print("Zero Sales Days:", zero_sales_days)
print("Valid Days for Percentage Error:", len(results_valid))
print("Total Test Days:", len(test))
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
results.to_csv("arima_forecast_results.csv", index=True)

print("Forecast results saved successfully.")

print("Rows saved:", len(results))
print("Columns:", list(results.columns))
print(results.head(3))
print("\nARIMA Model Summary:")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("Mean Percentage Error:", round(mean_percentage_error, 2), "%")
from statsmodels.tsa.holtwinters import ExponentialSmoothing

model_es = ExponentialSmoothing(
    train["TotalSales"],
    trend="add",
    seasonal=None
)

model_es_fit = model_es.fit()
forecast_es = model_es_fit.forecast(len(test))

print("Exponential Smoothing forecast created successfully.")
mae_es = mean_absolute_error(test["TotalSales"], forecast_es)

mse_es = mean_squared_error(test["TotalSales"], forecast_es)

rmse_es = mse_es ** 0.5

error_percentage_es = (mae_es / average_sales) * 100

print("Exponential Smoothing Results:")
print("MAE:", round(mae_es, 2))
print("MSE:", round(mse_es, 2))
print("RMSE:", round(rmse_es, 2))
print("MAE Percentage:", round(error_percentage_es, 2), "%")
comparison = pd.DataFrame({
    "Model": ["ARIMA", "Exponential Smoothing"],
    "MAE": [mae, mae_es],
    "RMSE": [rmse, rmse_es],
    "MAE Percentage": [error_percentage, error_percentage_es]
})

print("Model Comparison:")
print(comparison)

best_model = comparison.loc[comparison["MAE"].idxmin(), "Model"]

print("\nBest Model based on MAE:", best_model)
print("\nModel Performance Summary")
print("-" * 35)

for index, row in comparison.iterrows():
    print(row["Model"])
    print("MAE:", round(row["MAE"], 2))
    print("RMSE:", round(row["RMSE"], 2))
    print("MAE Percentage:", round(row["MAE Percentage"], 2), "%")
    print()

print("Selected Best Model:", best_model)
future_days = 30

if best_model == "ARIMA":
    final_model = ARIMA(daily_sales["TotalSales"], order=(5, 1, 0))
else:
    final_model = ExponentialSmoothing(
        daily_sales["TotalSales"], trend="add", seasonal=None
    )

final_model_fit = final_model.fit()
future_forecast = final_model_fit.forecast(steps=future_days)

last_date = daily_sales.index.max()
future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=future_days,
    freq="D"
)

future_results = pd.DataFrame({
    "Date": future_dates,
    "Forecasted_Sales": future_forecast
})

print("30-Day Future Sales Forecast:")
print(future_results)
future_results.to_csv("30_day_sales_forecast.csv", index=False)

highest_day = future_results.loc[
    future_results["Forecasted_Sales"].idxmax()
]

lowest_day = future_results.loc[
    future_results["Forecasted_Sales"].idxmin()
]

print("Forecast saved successfully.")
print("\nHighest Forecasted Sales:")
print(highest_day)
print("\nLowest Forecasted Sales:")
print(lowest_day)
plt.figure(figsize=(12, 6))

plt.plot(
    daily_sales.index,
    daily_sales["TotalSales"],
    label="Historical Sales"
)

plt.plot(
    future_results["Date"],
    future_results["Forecasted_Sales"],
    label="30-Day Forecast"
)

plt.title("E-Commerce Sales: Historical vs 30-Day Forecast")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

comparison.to_csv("model_comparison.csv", index=False)

print("Model comparison saved successfully.")
print("\nFinal Model Comparison:")
print(comparison)

print("\nSelected Best Model:")
print(best_model)
summary = {
    "Best Model": best_model,
    "ARIMA MAE": round(mae, 2),
    "ARIMA RMSE": round(rmse, 2),
    "Exponential Smoothing MAE": round(mae_es, 2),
    "Exponential Smoothing RMSE": round(rmse_es, 2),
    "Test Days": len(test),
    "Future Forecast Days": future_days,
    "Mean Percentage Error": round(mean_percentage_error, 2)
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv("project_summary.csv", index=False)
print("===== PROJECT COMPLETED =====")

print("Best Model:", best_model)
print("ARIMA MAE:", round(mae, 2))
print("ARIMA RMSE:", round(rmse, 2))
print("Exponential Smoothing MAE:", round(mae_es, 2))
print("Exponential Smoothing RMSE:", round(rmse_es, 2))
print("Test Days:", len(test))
print("Future Forecast Days:", future_days)
print("Mean Percentage Error:", round(mean_percentage_error, 2), "%")
import os

print("===== PROJECT FILES =====")
print("arima_forecast_results.csv:", os.path.exists("arima_forecast_results.csv"))
print("30_day_sales_forecast.csv:", os.path.exists("30_day_sales_forecast.csv"))
print("model_comparison.csv:", os.path.exists("model_comparison.csv"))
print("project_summary.csv:", os.path.exists("project_summary.csv"))

print("\nCurrent Folder:")
print(os.getcwd())