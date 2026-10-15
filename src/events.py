"""
Ключевые события войны с Ираном 2026 года, важные для европейского газа.
Используются только для подписей на графиках и разбивки периода — НЕ как признаки модели.
Источники — новостные публикации от даты события.
"""
import pandas as pd

EVENTS = [
    # (дата, короткая подпись, источник)
    ("2026-02-28", "War starts (US/Israel strikes on Iran)",
     "https://thedefensepost.com/2026/02/28/us-israel-strike-iran/"),
    ("2026-03-02", "Qatar halts LNG output",
     "https://www.tradewindsnews.com/gas/qatarenergy-halts-production-after-iranian-drones-hit-ras-laffan/2-1-1952194"),
    ("2026-03-18", "Missile strike on Ras Laffan (-17% Qatar LNG)",
     "https://www.aljazeera.com/news/2026/3/19/iran-attacks-cut-17-of-qatars-lng-capacity-for-up-to-5-years-qatarenergy"),
    ("2026-04-08", "US–Iran ceasefire",
     "https://www.newsonair.gov.in/us-iran-ceasefire-us-president-donald-trump-says-it-is-a-big-day-for-world-peace"),
    ("2026-07-08", "Ceasefire collapses, US strikes resume",
     "https://www.cnn.com/2026/07/08/world/live-news/iran-war-nato-summit-ukraine-trump"),
]


def events_frame() -> pd.DataFrame:
    df = pd.DataFrame(EVENTS, columns=["date", "label", "source"])
    df["date"] = pd.to_datetime(df["date"])
    df.index = range(len(df))
    return df
