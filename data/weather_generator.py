"""
Synthetic Multi-Parameter Weather Dataset Generator.
Simulates physically realistic hourly meteorological time-series:
- Temperature (°C) with seasonal, diurnal, and synoptic wave components
- Humidity (%) with inverse thermal correlation and precipitation events
- Air Pressure (hPa) with barometric synoptic waves and tidal oscillations
- Wind Speed (m/s) with pressure-gradient dependency and Rayleigh gust distribution
"""

import os
import argparse
import numpy as np
import pandas as pd


def generate_synthetic_weather_data(
    start_date: str = "2022-01-01 00:00:00",
    n_days: int = 730,  # 2 years of hourly data (17,520 samples)
    seed: int = 42,
    output_path: str = None
) -> pd.DataFrame:
    """
    Generate realistic multi-year hourly weather data.
    """
    np.random.seed(seed)
    n_hours = n_days * 24
    date_range = pd.date_range(start=start_date, periods=n_hours, freq="h")

    # Time indices
    t_hours = np.arange(n_hours)
    day_of_year = (t_hours / 24.0) % 365.25
    hour_of_day = t_hours % 24

    # 1. Temperature (°C)
    # Annual cycle: peak in July (~day 196), lowest in January
    annual_temp = 15.0 + 12.0 * np.sin(2 * np.pi * (day_of_year - 105) / 365.25)
    
    # Diurnal cycle: daily peak around 14:00 (hour 14), lowest around 05:00
    diurnal_temp = 5.5 * np.sin(2 * np.pi * (hour_of_day - 8) / 24.0)
    
    # Synoptic front waves (weather systems passing every 4 to 8 days)
    synoptic_temp = (
        3.0 * np.sin(2 * np.pi * t_hours / (24.0 * 5.5)) +
        2.0 * np.sin(2 * np.pi * t_hours / (24.0 * 11.2))
    )
    
    # AR(1) colored noise for natural temporal continuity
    temp_noise = np.zeros(n_hours)
    phi = 0.94
    innovations = np.random.normal(0, 0.45, n_hours)
    for t in range(1, n_hours):
        temp_noise[t] = phi * temp_noise[t - 1] + innovations[t]
        
    temperature = annual_temp + diurnal_temp + synoptic_temp + temp_noise
    temperature = np.round(temperature, 2)

    # 2. Air Pressure (hPa)
    # Mean 1013.25 hPa, low during storms, high during calm clear periods
    # Low pressure systems correlate with sudden temperature shifts
    base_pressure = 1013.25
    pressure_waves = (
        12.0 * np.sin(2 * np.pi * t_hours / (24.0 * 6.3) + 1.2) +
        6.0 * np.sin(2 * np.pi * t_hours / (24.0 * 3.1) - 0.5)
    )
    # Atmospheric thermal tide (semi-diurnal ~12h oscillation)
    pressure_tide = 1.0 * np.sin(2 * np.pi * hour_of_day / 12.0)
    
    pressure_noise = np.zeros(n_hours)
    p_phi = 0.92
    p_innovations = np.random.normal(0, 0.35, n_hours)
    for t in range(1, n_hours):
        pressure_noise[t] = p_phi * pressure_noise[t - 1] + p_innovations[t]

    pressure = base_pressure + pressure_waves + pressure_tide + pressure_noise
    pressure = np.round(pressure, 2)

    # 3. Relative Humidity (%)
    # Inversely related to temperature (higher saturation deficit as air warms)
    # Plus high humidity when pressure drops (storm systems)
    temp_normalized = (temperature - np.mean(temperature)) / np.std(temperature)
    pres_deviation = (1013.25 - pressure) / 10.0  # lower pressure -> higher humidity
    
    base_humidity = 68.0
    diurnal_humidity = -14.0 * np.sin(2 * np.pi * (hour_of_day - 8) / 24.0)
    seasonal_humidity = 8.0 * np.cos(2 * np.pi * day_of_year / 365.25)
    
    hum_noise = np.zeros(n_hours)
    h_phi = 0.88
    h_innovations = np.random.normal(0, 1.2, n_hours)
    for t in range(1, n_hours):
        hum_noise[t] = h_phi * hum_noise[t - 1] + h_innovations[t]

    humidity = (
        base_humidity 
        + diurnal_humidity 
        + seasonal_humidity 
        + (pres_deviation * 6.0) 
        - (temp_normalized * 4.0) 
        + hum_noise
    )
    # Realistic bound [15%, 100%]
    humidity = np.clip(humidity, 15.0, 99.0)
    humidity = np.round(humidity, 2)

    # 4. Wind Speed (m/s)
    # Stronger during low pressure gradients / frontal passages
    pressure_grad = np.abs(np.gradient(pressure))
    base_wind = 2.5 + pressure_grad * 4.5
    # Afternoon sea/land thermal breeze
    thermal_wind = 1.2 * np.maximum(0, np.sin(2 * np.pi * (hour_of_day - 10) / 24.0))
    # Rayleigh-like positive perturbation
    gust_component = np.random.rayleigh(scale=1.8, size=n_hours)
    
    wind_speed = base_wind + thermal_wind + gust_component
    wind_speed = np.clip(wind_speed, 0.2, 28.0)
    wind_speed = np.round(wind_speed, 2)

    df = pd.DataFrame({
        "timestamp": date_range,
        "temperature": temperature,
        "humidity": humidity,
        "pressure": pressure,
        "wind_speed": wind_speed
    })

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[Dataset] Generated {len(df):,} hourly records saved to: {output_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic weather time series CSV.")
    parser.add_argument("--days", type=int, default=730, help="Number of days to simulate (default: 730 = 2 years)")
    parser.add_argument("--output", type=str, default="data/weather_data.csv", help="Target output CSV file path")
    args = parser.parse_args()

    generate_synthetic_weather_data(n_days=args.days, output_path=args.output)
