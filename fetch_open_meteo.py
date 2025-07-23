import pandas as pd
import requests

# Configuration
latitude = 39.9526
longitude = -75.1652
start_date = "2015-05-01"
end_date = "2025-05-01"
timezone = "America/New_York"
out_csv = "philadelphia_weather_openmeteo_daily.csv"

# Define daily parameters
params = {
    "latitude": latitude,
    "longitude": longitude,
    "start_date": start_date,
    "end_date": end_date,
    "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,wind_speed_10m_max,wind_gusts_10m_max,weathercode",
    "timezone": timezone
}

# API Endpoint
url = "https://archive-api.open-meteo.com/v1/archive"

# Fetch data
print("Fetching daily weather data from Open-Meteo...")
response = requests.get(url, params=params)

if response.status_code != 200:
    print(" Failed to fetch data.")
    print(response.text)
else:
    data = response.json()
    print(" Data fetched successfully!")

    # Convert to DataFrame
    df = pd.DataFrame(data["daily"])
    df["time"] = pd.to_datetime(df["time"])

    # Save to CSV
    df.to_csv(out_csv, index=False)
    print(f" Daily weather data saved to {out_csv}")