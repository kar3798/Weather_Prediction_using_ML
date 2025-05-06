import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# Output directory
results_dir = "evaluation_results"
os.makedirs(results_dir, exist_ok=True)

# Load data
predicted = pd.read_csv("predicted_weather_today.csv")
actual = pd.read_csv("weather_data_api.csv")

# Parse date
actual["date"] = pd.to_datetime(actual["date"])
pred_date = pd.to_datetime(predicted["Date_Time"].iloc[0])
actual_today = actual[actual["date"] == pred_date]

if actual_today.empty:
    print(f"No actual data for {pred_date.date()}")
    exit()

# Extract actual values using Open-Meteo field names
actual_row = actual_today.iloc[-1]
actual_values = {
    "temperature_2m_max": actual_row["temp_max"],
    "temperature_2m_min": actual_row["temp_min"],
    "temperature_2m_mean": actual_row["temp"],
    "precipitation_sum": actual_row["precipitation"],
    "wind_speed_10m_max": actual_row["wind_speed"],
    "wind_gusts_10m_max": actual_row["wind_gust"],
    "weathercode": actual_row["weathercode"]
}

# Match predicted columns
predicted_keys = [col for col in actual_values if col in predicted.columns and pd.notna(actual_values[col])]
predicted_values = predicted.iloc[0][predicted_keys]

# Ground truth and predictions
y_true = [actual_values[key] for key in predicted_keys]
y_pred = predicted_values.values

# Evaluation metrics
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean([abs((t - p) / t) * 100 for t, p in zip(y_true, y_pred) if abs(t) > 0.001])
    return {"MAE": mae, "RMSE": rmse, "R²": r2, "MAPE": mape}

metrics = evaluate(y_true, y_pred)

# Display results
print(f"\nPredicted vs Actual ({pred_date.date()}):\n")
for key in predicted_keys:
    pred = predicted_values[key]
    act = actual_values[key]
    print(f"{key:21}: Predicted = {pred:6.2f}, Actual = {act:6.2f}")

print("\nEvaluation Metrics:")
for k, v in metrics.items():
    print(f"{k:5}: {v:.4f}")

# Save bar chart
plt.figure(figsize=(8, 5))
bar_width = 0.35
index = np.arange(len(predicted_keys))
plt.bar(index, y_pred, bar_width, label="Predicted", alpha=0.7)
plt.bar(index + bar_width, y_true, bar_width, label="Actual", alpha=0.7)
plt.xticks(index + bar_width / 2, predicted_keys, rotation=45)
plt.ylabel("Value")
plt.title(f"Prediction vs Actual - {pred_date.date()}")
plt.legend()
plt.tight_layout()
plt.savefig(f"{results_dir}/comparison_{pred_date.date()}.png")
plt.close()
