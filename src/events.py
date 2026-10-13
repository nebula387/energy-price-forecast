"""
Ключевые события войны с Ираном 2026 года, важные для европейского газа.
Используются только для подписей на графиках и разбивки периода — НЕ как признаки модели.
"""
import pandas as pd

EVENTS = [
    # (дата, короткая подпись, источник)
    ("2026-02-28", "War starts (US/Israel strikes on Iran)",
     "https://www.everycrsreport.com/reports/IN12696.html"),
    ("2026-03-02", "Qatar halts LNG output",
     "https://en.wikipedia.org/wiki/2026_Iran_war_fuel_crisis"),
    ("2026-03-18", "Strike on Ras Laffan LNG",
     "https://en.wikipedia.org/wiki/2026_Iran_war_fuel_crisis"),
    ("2026-04-08", "US–Iran ceasefire",
     "https://en.wikipedia.org/wiki/2026_Iran_war_fuel_crisis"),
    ("2026-07-22", "Re-escalation: TTF highest since 2023",
     "https://www.kpler.com/blog/global-lng-and-natural-gas-prices-surge-as-us-and-iran-resume-hot-war"),
]


def events_frame() -> pd.DataFrame:
    df = pd.DataFrame(EVENTS, columns=["date", "label", "source"])
    df["date"] = pd.to_datetime(df["date"])
    df.index = range(len(df))
    return df
