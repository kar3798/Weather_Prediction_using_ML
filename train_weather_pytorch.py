import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# Configuration
SEQUENCE_LENGTH = 180
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50
EARLY_STOP_PATIENCE = 10
HIDDEN_SIZE = 64

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
file_path = "philadelphia_weather_openmeteo_daily.csv"
df = pd.read_csv(file_path)
df["time"] = pd.to_datetime(df["time"])
df.sort_values("time", inplace=True)

print(f"Data range: {df['time'].min()} to {df['time'].max()} — {len(df)} days")

# Add cyclical time features
df["dayofyear"] = df["time"].dt.dayofyear
df["sin_day"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
df["cos_day"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)

# Define features
input_features = [
    "temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max",
    "sin_day", "cos_day"
]

output_features = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "precipitation_sum", "wind_speed_10m_max", "wind_gusts_10m_max", "weathercode"
]

# Drop NA and scale
df.dropna(subset=input_features + output_features, inplace=True)
input_scaler = MinMaxScaler()
output_scaler = MinMaxScaler()
X_scaled = input_scaler.fit_transform(df[input_features])
y_scaled = output_scaler.fit_transform(df[output_features])

joblib.dump(input_scaler, "input_scaler.pkl")
joblib.dump(output_scaler, "output_scaler.pkl")
joblib.dump({
    "input_features": input_features,
    "output_features": output_features,
    "sequence_length": SEQUENCE_LENGTH
}, "feature_config.pkl")

# Sequence creation
def create_sequences(X, y, seq_len):
    xs, ys = [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQUENCE_LENGTH)
print(f"Training sequences: {X_seq.shape}, Targets: {y_seq.shape}")

X_tensor = torch.tensor(X_seq, dtype=torch.float32)
y_tensor = torch.tensor(y_seq, dtype=torch.float32)

# Train/val split
train_size = int(0.8 * len(X_tensor))
X_train, X_val = X_tensor[:train_size], X_tensor[train_size:]
y_train, y_val = y_tensor[:train_size], y_tensor[train_size:]

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

# Model
class WeatherLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = WeatherLSTM(X_seq.shape[2], y_seq.shape[1], HIDDEN_SIZE).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
best_val_loss = float('inf')
patience = 0
print("Starting training...")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            val_loss += criterion(model(X_batch), y_batch).item()

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}: Train {avg_train_loss:.4f}, Val {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "weather_model.pth")
        print("✓ Model improved — saved")
        patience = 0
    else:
        patience += 1
        if patience >= EARLY_STOP_PATIENCE:
            print("✖ Early stopping")
            break

print("Training complete.")
