import csv
import os
from datetime import datetime
from weather_data_fetcher_api import get_weather_data

CSV_FILE = "weather_data_api.csv"

def save_weather_to_csv(weather_data):
    if not weather_data or "daily" not in weather_data:
        print("No valid daily data to save.")
        return

    daily = weather_data["daily"]

    # Extract the first (i.e., today's) values
    date = daily["time"][0]
    temp_min = daily["temperature_2m_min"][0]
    temp_max = daily["temperature_2m_max"][0]
    temp_mean = daily["temperature_2m_mean"][0]
    precipitation = daily["precipitation_sum"][0]
    wind_speed = daily["wind_speed_10m_max"][0]
    wind_gust = daily["wind_gusts_10m_max"][0]
    weathercode = daily["weathercode"][0]

    # Write to CSV
    file_exists = os.path.exists(CSV_FILE)
    with open(CSV_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([
                "date", "temp_min", "temp_max", "temp",
                "precipitation", "wind_speed", "wind_gust", "weathercode"
            ])
        writer.writerow([
            date, temp_min, temp_max, temp_mean,
            precipitation, wind_speed, wind_gust, weathercode
        ])
    print(f"Weather data saved for {date} in {CSV_FILE}")


def main():
    weather_data = get_weather_data()
    if weather_data:
        save_weather_to_csv(weather_data)

if __name__ == "__main__":
    main()
