"""
Тесты правила «никакой утечки будущего».

Запуск:
    python -m pytest -q
"""
import numpy as np
import pandas as pd

from src.build_dataset import daily_asof
from src.config import TZ
from src.features import PRICE_LAGS, make_features


def hourly_index(start: str, end: str) -> pd.DatetimeIndex:
    # период с обоими переходами на летнее и зимнее время 2025 года
    return pd.date_range(pd.Timestamp(start, tz=TZ), pd.Timestamp(end, tz=TZ), freq="h", inclusive="left").tz_convert("UTC")


def synthetic(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Цена = порядковый номер дня поставки: любая утечка того же дня сразу видна."""
    delivery = index.tz_convert(TZ).tz_localize(None).normalize()
    day_no = (delivery - delivery.min()).days.astype(float)
    n = len(index)
    return pd.DataFrame({
        "price": day_no,
        "load_forecast": np.full(n, 50_000.0),
        "fc_solar": np.full(n, 5_000.0),
        "fc_wind_onshore": np.full(n, 10_000.0),
        "fc_wind_offshore": np.full(n, 2_000.0),
    }, index=index)


def test_price_lags_never_use_delivery_day():
    df = make_features(synthetic(hourly_index("2025-03-01", "2025-11-30")))
    day_no = df["price"]
    for col in PRICE_LAGS:
        k = int(col.split("_")[1][1:])  # price_d7_same_hour -> 7
        valid = df[col].dropna()
        assert (valid <= day_no[valid.index] - k).all(), col
    # и лаги реально заполнены, в том числе в дни перевода часов
    for day in ("2025-03-30", "2025-10-26"):
        rows = df[df["delivery_date"] == pd.Timestamp(day)]
        assert rows["price_d1_same_hour"].notna().all(), day
        assert (rows["price_d1_mean"] == rows["price"] - 1).all(), day


def test_daily_series_respect_publication_lag():
    index = hourly_index("2025-06-01", "2025-06-15")
    dates = pd.date_range("2025-05-20", "2025-06-15", freq="D")
    daily = pd.DataFrame({"ttf": np.arange(len(dates), dtype=float)}, index=dates)
    out = daily_asof(daily, index, lag_days=2)

    delivery = index.tz_convert(TZ).tz_localize(None).normalize()
    value_date = dates[out["ttf"].astype(int).to_numpy()]
    assert (value_date <= delivery - pd.Timedelta(days=2)).all()
    assert (value_date == delivery - pd.Timedelta(days=2)).all()  # без пропусков берётся ровно D-2


def test_weekend_gap_is_forward_filled_with_older_value():
    index = hourly_index("2025-06-09", "2025-06-10")  # понедельник
    dates = pd.to_datetime(["2025-06-05", "2025-06-06"])  # чт, пт; выходных котировок нет
    daily = pd.DataFrame({"ttf": [1.0, 2.0]}, index=dates)
    out = daily_asof(daily, index, lag_days=2)  # нужен день <= субботы 07.06 → пятница
    assert (out["ttf"] == 2.0).all()
