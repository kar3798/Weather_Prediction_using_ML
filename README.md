# Weather Forecasting with LSTM

This project uses historical weather data and a machine learning model (LSTM) to forecast daily weather variables such as temperature, wind speed, and precipitation.

## Project Structure

```
weather-project/
│
├── train_weather_pytorch.py         # Train the LSTM model
├── predict_weather_pytorch.py       # Make daily predictions using the trained model
├── evaluate_prediction.py           # Compare predictions with actual data
├── weather_data_fetcher_api.py      # Fetch current/daily weather from Open-Meteo API
├── weather_fetcher.py               # Saves today's weather to CSV for evaluation
├── weather_ui.py                    # (Optional) UI for viewing forecasts
├── .gitignore                       # Specifies files to exclude from Git
├── .env                             # Stores API keys (ignored by Git)
├── requirements.txt                 # Python dependencies
└── README.md                        # Project overview
```

## Setup

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Create a `.env` file and add your API key:

```
OPENWEATHER_API_KEY=your_openweather_api_key
```

3. Run training:

```
python train_weather_pytorch.py
```

4. Make a prediction:

```
python predict_weather_pytorch.py
```

5. Evaluate prediction accuracy:

```
python evaluate_prediction.py
```

## Model Summary

- Model: LSTM (Long Short-Term Memory)
- Input: Past 180 days of weather data
- Output: Forecast for the next day’s:
  - Max/Min/Mean Temperature
  - Precipitation
  - Wind Speed & Gusts
  - Weather Code

## Evaluation Metrics

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score
- MAPE (Mean Absolute Percentage Error)

## Notes

- Daily predictions are saved in `predicted_weather_today.csv`
- Evaluation comparisons are saved in `evaluation_results/`
- Model weights are saved as `weather_model.pth` (ignored by Git)
