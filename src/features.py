"""
Признаки для модели цены на сутки вперёд.

Каждая строка — час поставки дня D. Всё, что попадает в признаки, известно к 12:00 CET дня D-1:
- календарь;
- прогнозы TSO (нагрузка, ветер, солнце) — публикуются до аукциона;
- цены прошлых дней поставки: D-1 известен полностью (аукцион за D-1 прошёл в D-2);
- газ, CO2, хранилища — уже сдвинуты в build_dataset.

Лаги цены считаются по (дню поставки, местному часу), а не сдвигом строк: так переход
на летнее/зимнее время не подтягивает в признак цену того же дня D.
"""
import holidays
import pandas as pd

from src.config import TZ

CALENDAR = ["hour", "dow", "month", "is_holiday"]
FUNDAMENTALS = [
    "load_forecast", "fc_solar", "fc_wind_onshore", "fc_wind_offshore",
    "residual_load_forecast", "res_share",
]
PRICE_LAGS = [
    "price_d1_same_hour", "price_d2_same_hour", "price_d7_same_hour",
    "price_d1_mean", "price_d1_min", "price_d1_max", "price_d1_std",
]
GAS = ["ttf_gas", "carbon_proxy", "gas_storage_full", "ttf_change_5d"]

FEATURE_SETS = {
    "base": CALENDAR + FUNDAMENTALS + PRICE_LAGS,
    "gas": CALENDAR + FUNDAMENTALS + PRICE_LAGS + GAS,
}
TARGET = "price"


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    local = out.index.tz_convert(TZ)
    out["delivery_date"] = local.tz_localize(None).normalize()
    out["hour"] = local.hour
    out["dow"] = local.dayofweek
    out["month"] = local.month
    de_holidays = holidays.Germany(years=range(local.year.min(), local.year.max() + 1))
    out["is_holiday"] = pd.Index(out["delivery_date"].dt.date).isin(de_holidays).astype(int)

    fc_res = out[["fc_solar", "fc_wind_onshore", "fc_wind_offshore"]].sum(axis=1, min_count=1)
    out["res_share"] = fc_res / out["load_forecast"]

    # Таблица день × местный час (в день перевода часов на зиму час 2 встречается дважды — усредняем)
    grid = out.pivot_table(index="delivery_date", columns="hour", values=TARGET, aggfunc="mean")
    for k in (1, 2, 7):
        lagged = grid.copy()
        lagged.index = lagged.index + pd.Timedelta(days=k)
        stacked = lagged.stack().rename(f"price_d{k}_same_hour")
        out = out.join(stacked, on=["delivery_date", "hour"])

    daily = out.groupby("delivery_date")[TARGET].agg(["mean", "min", "max", "std"])
    daily.index = daily.index + pd.Timedelta(days=1)
    daily.columns = [f"price_d1_{c}" for c in daily.columns]
    out = out.join(daily, on="delivery_date")

    if "ttf_gas" in out:
        # изменение TTF за 5 дней — по уже сдвинутому ряду, поэтому без утечки
        ttf_daily = out.groupby("delivery_date")["ttf_gas"].first()
        change = (ttf_daily - ttf_daily.shift(5)).rename("ttf_change_5d")
        out = out.join(change, on="delivery_date")
    return out
