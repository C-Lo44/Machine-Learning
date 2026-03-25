import requests

url = "https://api.open-meteo.com/v1/forecast"

params = {
    "latitude": 42.3314,   # Detroit
    "longitude": -83.0458,
    "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
    "timezone": "auto",
    "forecast_days": 3
}

response = requests.get(url, params=params)
response.raise_for_status()

data = response.json()

print(data["hourly"]["time"][:5])
print(data["hourly"]["temperature_2m"][:5])
print(data["hourly"]["relative_humidity_2m"][:5])
print(data["hourly"]["wind_speed_10m"][:5])