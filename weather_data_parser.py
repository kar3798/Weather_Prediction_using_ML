import pandas as pd
from datetime import datetime

def get_latest_weather():
    file_path = "weather_data_api.csv"

    try:
        df = pd.read_csv(file_path)

        # Convert Date column to datetime format
        df["date"] = pd.to_datetime(df["date"])
        
        # Get the latest entry for today's date
        today = datetime.today().date()
        df_today = df[df["date"].dt.date == today]

        if df_today.empty:
            return None  # No data available for today

        latest_entry = df_today.iloc[-1]  # Get the most recent entry

        return {
            "Temperature": latest_entry["temp"],
            "Humidity": latest_entry["humidity"],
            "Pressure": latest_entry["pressure"],
            "Wind Speed": latest_entry["wind_speed"]
        }

    except Exception as e:
        print(f"Error reading weather data: {e}")
        return None

