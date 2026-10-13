"""
Почасовая погода из архива Open-Meteo (бесплатно, без ключа), среднее по нескольким точкам Германии.

Запуск:
    python -m src.fetch_weather

Архив отстаёт от текущей даты примерно на 5 дней — последние дни добираются позже.
"""
import logging

import pandas as pd
import requests

from src.config import RAW_DIR, START, WEATHER_POINTS, today_tz

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("weather")

URL = "https://archive-api.open-meteo.com/v1/archive"
VARS = ["temperature_2m", "wind_speed_100m", "shortwave_radiation", "cloud_cover"]


def fetch_point(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(VARS),
        "timezone": "UTC",
    }
    resp = requests.get(URL, params=params, timeout=60)
    resp.raise_for_status()
    hourly = resp.json()["hourly"]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.set_index("time")


def main() -> None:
    start = START.strftime("%Y-%m-%d")
    end = (today_tz() - pd.Timedelta(days=6)).strftime("%Y-%m-%d")
    frames = []
    for name, (lat, lon) in WEATHER_POINTS.items():
        log.info("Погода: %s", name)
        frames.append(fetch_point(lat, lon, start, end))
    weather = pd.concat(frames).groupby(level=0).mean()
    weather.columns = [f"w_{c}" for c in weather.columns]
    out = RAW_DIR / "weather.parquet"
    weather.to_parquet(out)
    log.info("Сохранено %s часов в %s", len(weather), out)


if __name__ == "__main__":
    main()
