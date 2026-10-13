"""Общие настройки проекта."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Германия + Люксембург — единая ценовая зона
ZONE = "DE_LU"
TZ = "Europe/Berlin"

# 2023–2025 — «нормальный» период для обучения, с 28.02.2026 — война с Ираном и газовый шок
START = pd.Timestamp("2023-01-01", tz=TZ)
CRISIS_START = pd.Timestamp("2026-02-28", tz=TZ)

# Публикационные задержки дневных рядов относительно ДНЯ ПОСТАВКИ D.
# Аукцион на D закрывается в 12:00 CET дня D-1, поэтому:
# - котировки TTF/CO2: закрытие за D-1 появляется вечером D-1 — уже после аукциона,
#   последнее известное закрытие — за D-2;
# - AGSI публикует газовый день X вечером X+1, к 12:00 D-1 известен только день D-3.
MARKET_LAG_DAYS = 2
STORAGE_LAG_DAYS = 3

# Точки для погоды: север (ветер), юг (солнце), центр (нагрузка)
WEATHER_POINTS = {
    "hamburg": (53.55, 9.99),
    "berlin": (52.52, 13.40),
    "frankfurt": (50.11, 8.68),
    "munich": (48.14, 11.58),
}


def today_tz() -> pd.Timestamp:
    return pd.Timestamp.now(tz=TZ).normalize()
