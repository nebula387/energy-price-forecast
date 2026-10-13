"""
Загрузка данных ENTSO-E помесячно с кэшированием.

Запуск из корня проекта:
    python -m src.fetch_entsoe
Повторный запуск докачивает только новые месяцы (текущий месяц всегда перекачивается).
"""
import logging
import os
import time

import pandas as pd
from dotenv import load_dotenv
from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError

from src.config import RAW_DIR, START, TZ, ZONE, today_tz

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("entsoe")

# Что качаем. Всё, кроме цены, — то, что известно ДО аукциона (прогнозы), чтобы не было утечки.
QUERIES = {
    "day_ahead_price": lambda c, s, e: c.query_day_ahead_prices(ZONE, start=s, end=e),
    "load_forecast": lambda c, s, e: c.query_load_forecast(ZONE, start=s, end=e),
    "wind_solar_forecast": lambda c, s, e: c.query_wind_and_solar_forecast(ZONE, start=s, end=e),
    # фактическая нагрузка — только для анализа, НЕ как признак модели
    "load_actual": lambda c, s, e: c.query_load(ZONE, start=s, end=e),
}


def month_ranges(start: pd.Timestamp, end: pd.Timestamp):
    months = pd.date_range(start, end, freq="MS", tz=TZ)
    for m_start in months:
        m_end = min(m_start + pd.offsets.MonthBegin(1), end)
        yield m_start, m_end


def fetch_with_retry(fn, client, s, e, attempts: int = 3):
    for i in range(1, attempts + 1):
        try:
            return fn(client, s, e)
        except NoMatchingDataError:
            return None
        except Exception as exc:  # сеть, 503 и т.п.
            log.warning("Попытка %s/%s не удалась: %s", i, attempts, exc)
            time.sleep(5 * i)
    raise RuntimeError(f"Не удалось загрузить {s:%Y-%m} после {attempts} попыток")


def main() -> None:
    load_dotenv()
    api_key = os.getenv("ENTSOE_API_KEY")
    if not api_key:
        raise SystemExit("Нет ENTSOE_API_KEY в .env")
    client = EntsoePandasClient(api_key=api_key)

    # +2 дня: цены на завтра публикуются около 12:00–13:00 CET
    end = today_tz() + pd.Timedelta(days=2)
    current_month = today_tz().strftime("%Y-%m")

    for name, fn in QUERIES.items():
        out_dir = RAW_DIR / "entsoe" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        for s, e in month_ranges(START, end):
            path = out_dir / f"{s:%Y-%m}.parquet"
            if path.exists() and s.strftime("%Y-%m") != current_month:
                continue
            log.info("%s: %s", name, f"{s:%Y-%m}")
            data = fetch_with_retry(fn, client, s, e)
            if data is None or len(data) == 0:
                log.warning("%s: нет данных за %s", name, f"{s:%Y-%m}")
                continue
            df = data.to_frame(name) if isinstance(data, pd.Series) else data
            df.to_parquet(path)
            time.sleep(1)  # бережём лимиты API


if __name__ == "__main__":
    main()
