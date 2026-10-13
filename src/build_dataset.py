"""
Сборка единой почасовой таблицы (UTC) из сырых данных.

Запуск:
    python -m src.build_dataset

Правило против утечки: в признаках только то, что известно до аукциона на сутки вперёд
(12:00 CET накануне дня поставки). Дневные ряды (газ, CO2, хранилища) привязываются к дню
поставки по берлинскому времени с задержкой публикации из config.py.

Колонки ТОЛЬКО для анализа (в модель не идут, см. src/features.py):
- load_actual — фактическая нагрузка, известна лишь после поставки;
- w_* — фактическая погода из архива Open-Meteo: в модели это был бы «идеальный прогноз погоды».
"""
import logging

import pandas as pd

from src.config import CRISIS_START, MARKET_LAG_DAYS, PROCESSED_DIR, RAW_DIR, STORAGE_LAG_DAYS, TZ

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("build")


def load_entsoe(name: str) -> pd.DataFrame:
    files = sorted((RAW_DIR / "entsoe" / name).glob("*.parquet"))
    if not files:
        raise SystemExit(f"Нет файлов для {name}. Сначала запусти src.fetch_entsoe")
    df = pd.concat(pd.read_parquet(f) for f in files)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    # С 01.10.2025 рынок на сутки вперёд торгуется 15-минутными интервалами — приводим к часу
    return df.resample("h").mean()


def daily_asof(df: pd.DataFrame, index: pd.DatetimeIndex, lag_days: int) -> pd.DataFrame:
    """Каждому часу поставки сопоставляет последнее дневное значение с датой <= (день поставки - lag_days).

    Выходные и праздники закрываются последним торговым днём, но не дольше 7 дней.
    """
    daily = df.copy()
    daily.index = pd.to_datetime(daily.index)
    if daily.index.tz is not None:
        daily.index = daily.index.tz_localize(None)
    daily = daily.dropna(how="all").sort_index()
    daily.index = daily.index.normalize().as_unit("ns").rename("value_date")

    delivery_date = index.tz_convert(TZ).tz_localize(None).normalize().as_unit("ns")
    left = pd.DataFrame({"cutoff": delivery_date - pd.Timedelta(days=lag_days)})
    merged = pd.merge_asof(
        left, daily.reset_index(), left_on="cutoff", right_on="value_date",
        direction="backward", tolerance=pd.Timedelta(days=7),
    )
    return merged.drop(columns=["cutoff", "value_date"]).set_index(index)


def main() -> None:
    price = load_entsoe("day_ahead_price").rename(columns={"day_ahead_price": "price"})
    load_fc = load_entsoe("load_forecast")
    load_fc.columns = ["load_forecast"]
    ws = load_entsoe("wind_solar_forecast")
    ws.columns = [f"fc_{c.lower().replace(' ', '_')}" for c in ws.columns]
    load_act = load_entsoe("load_actual")
    load_act.columns = ["load_actual"]

    df = price.join([load_fc, ws, load_act], how="left")
    wind_solar = df.filter(like="fc_").sum(axis=1, min_count=1)
    df["residual_load_forecast"] = df["load_forecast"] - wind_solar

    weather_path = RAW_DIR / "weather.parquet"
    if weather_path.exists():
        df = df.join(pd.read_parquet(weather_path), how="left")
    else:
        log.warning("Нет weather.parquet — пропускаю погоду")

    for name, lag in [("market_prices", MARKET_LAG_DAYS), ("gas_storage", STORAGE_LAG_DAYS)]:
        path = RAW_DIR / f"{name}.parquet"
        if path.exists():
            df = df.join(daily_asof(pd.read_parquet(path), df.index, lag))
        else:
            log.warning("Нет %s.parquet — пропускаю", name)

    df["is_crisis"] = (df.index >= CRISIS_START.tz_convert("UTC")).astype(int)
    df = df.dropna(subset=["price"])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "dataset.parquet"
    df.to_parquet(out)
    log.info("Итог: %s строк, %s колонок, %s → %s", len(df), df.shape[1],
             f"{df.index.min():%Y-%m-%d}", f"{df.index.max():%Y-%m-%d %H:%M} UTC")
    log.info("Сохранено в %s", out)
    log.info("Доля пропусков по колонкам:\n%s", df.isna().mean().round(3).to_string())


if __name__ == "__main__":
    main()
