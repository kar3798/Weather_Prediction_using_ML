import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import timedelta
import joblib

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
df = pd.read_csv("philadelphia_weather_openmeteo_daily.csv")
df["time"] = pd.to_datetime(df["time"])
df.sort_values("time", inplace=True)

# Load feature config
config = joblib.load("feature_config.pkl")
input_features = config["input_features"]
output_features = config["output_features"]
SEQUENCE_LENGTH = config["sequence_length"]

# Add cyclical time features
df["dayofyear"] = df["time"].dt.dayofyear
df["sin_day"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
df["cos_day"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)

# Clean data
df.dropna(subset=input_features, inplace=True)

# Load scalers
input_scaler = joblib.load("input_scaler.pkl")
output_scaler = joblib.load("output_scaler.pkl")

# Scale inputs
input_scaled = input_scaler.transform(df[input_features])

# Check for enough data
if len(input_scaled) < SEQUENCE_LENGTH:
    raise ValueError("Not enough data to form a prediction sequence")

# Prepare input sequence
sequence = input_scaled[-SEQUENCE_LENGTH:]
sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)

# Define model
class WeatherLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# Load model
model = WeatherLSTM(len(input_features), len(output_features), hidden_size=64).to(device)
model.load_state_dict(torch.load("weather_model.pth", map_location=device))
model.eval()

# Predict
with torch.no_grad():
    prediction_scaled = model(sequence_tensor).cpu().numpy()

# Inverse scale
prediction = output_scaler.inverse_transform(prediction_scaled)[0]

# Set prediction date
latest_date = df["time"].max().normalize()
target_date = pd.to_datetime("2025-05-05")
prediction_date = target_date if target_date > latest_date else latest_date + timedelta(days=1)

# Save to file
predicted_df = pd.DataFrame([prediction], columns=output_features)
predicted_df.insert(0, "Date_Time", prediction_date.strftime("%Y-%m-%d"))
predicted_df.to_csv("predicted_weather_today.csv", index=False)

# Display
print(f"\nWeather Prediction for {prediction_date.strftime('%Y-%m-%d')}:")
for feat, val in zip(output_features, prediction):
    print(f"{feat:22}: {val:6.2f}")
