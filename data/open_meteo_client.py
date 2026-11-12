"""
Open-Meteo Historical & Real-Time Weather Data Client.
Fetches authentic meteorological time-series for specified coordinates (ERA5 / High-Resolution API).
Default Coordinates: 12.8675° N, 80.0713° E (Chennai / Chengalpattu region, India).
"""

import os
import json
import urllib.request
from typing import Optional
import pandas as pd


def fetch_open_meteo_data(
    latitude: float = 12.8675,
    longitude: float = 80.0713,
    start_date: str = "2022-01-01",
    end_date: str = "2023-12-31",
    output_path: Optional[str] = "data/weather_data.csv"
) -> pd.DataFrame:
    """
    Query Open-Meteo Historical Archive API for multi-variable hourly weather data.
    """
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={latitude}&longitude={longitude}&"
        f"start_date={start_date}&end_date={end_date}&"
        f"hourly=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m"
    )
    print(f"[Open-Meteo] Querying archive API for Lat: {latitude}, Lon: {longitude} ({start_date} to {end_date})...")

    req = urllib.request.Request(url, headers={"User-Agent": "WeatherLSTMProject/1.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    hourly = payload["hourly"]
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"]),
        "temperature": hourly["temperature_2m"],
        "humidity": hourly["relative_humidity_2m"],
        "pressure": hourly["surface_pressure"],
        "wind_speed": hourly["wind_speed_10m"]
    })

    # Forward and backward fill any occasional missing sensor timestamps
    df = df.ffill().bfill()
    df["temperature"] = df["temperature"].round(2)
    df["humidity"] = df["humidity"].round(2)
    df["pressure"] = df["pressure"].round(2)
    df["wind_speed"] = df["wind_speed"].round(2)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[Open-Meteo] Successfully saved {len(df):,} hourly records to: {output_path}")

    return df


if __name__ == "__main__":
    fetch_open_meteo_data(latitude=12.8675, longitude=80.0713)
