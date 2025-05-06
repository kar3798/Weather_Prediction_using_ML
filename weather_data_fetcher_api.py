import requests
import json
import os
from datetime import datetime
from dotenv import load_dotenv

USE_OPEN_METEO = True

# OpenWeatherMap
load_dotenv()

OWM_API_KEY = os.getenv("OPENWEATHER_API_KEY")
CITY = "Philadelphia"
OWM_URL = f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={OWM_API_KEY}&units=metric"

# Open-Meteo
LAT = 39.9526
LON = -75.1652
OM_URL = (
    f"https://api.open-meteo.com/v1/forecast"
    f"?latitude={LAT}&longitude={LON}"
    f"&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
    f"precipitation_sum,wind_speed_10m_max,wind_gusts_10m_max,weathercode"
    f"&timezone=auto"
)

# File to store daily API call count
API_CALL_COUNT_FILE = "api_counter.json"


# === API CALL LIMITING ===
def load_api_counter():
    if os.path.exists(API_CALL_COUNT_FILE):
        with open(API_CALL_COUNT_FILE, "r") as f:
            return json.load(f)
    return {"date": str(datetime.today().date()), "count": 0}


def save_api_counter(count):
    data = {"date": str(datetime.today().date()), "count": count}
    with open(API_CALL_COUNT_FILE, "w") as f:
        json.dump(data, f)


def can_make_api_call():
    api_data = load_api_counter()
    today = str(datetime.today().date())

    if api_data["date"] == today:
        if api_data["count"] >= 100:
            print("API CALL LIMIT REACHED. TRY AGAIN TOMORROW")
            return False
    else:
        api_data = {"date": today, "count": 0}
        save_api_counter(api_data["count"])

    return True



def get_weather_data():
    if not can_make_api_call():
        return None

    try:
        if USE_OPEN_METEO:
            response = requests.get(OM_URL)
        else:
            response = requests.get(OWM_URL)

        if response.status_code == 200:
            api_data = load_api_counter()
            api_data["count"] += 1
            save_api_counter(api_data["count"])
            print(f"API CALL DONE. CALLS USED: {api_data['count']}")
            return response.json()
        else:
            print(f"FAILED TO FETCH WEATHER DATA. Status code: {response.status_code}")
            return None

    except requests.exceptions.RequestException as e:
        print("Error fetching weather:", e)
        return None
