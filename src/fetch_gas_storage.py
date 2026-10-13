"""
Заполненность газовых хранилищ Германии из AGSI+ (Gas Infrastructure Europe).
Нужен бесплатный ключ: регистрация на agsi.gie.eu.

Запуск:
    python -m src.fetch_gas_storage

Если структура ответа отличается, скрипт выведет ключи первой записи — подправь COLS.
"""
import logging
import os

import pandas as pd
import requests
from dotenv import load_dotenv

from src.config import RAW_DIR, START, today_tz

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("agsi")

URL = "https://agsi.gie.eu/api"
COLS = ["gasDayStart", "full", "gasInStorage", "injection", "withdrawal"]


def main() -> None:
    load_dotenv()
    key = os.getenv("AGSI_API_KEY")
    if not key:
        raise SystemExit("Нет AGSI_API_KEY в .env")

    params = {
        "country": "DE",
        "from": START.strftime("%Y-%m-%d"),
        "to": today_tz().strftime("%Y-%m-%d"),
        "size": 300,
        "page": 1,
    }
    rows = []
    while True:
        resp = requests.get(URL, params=params, headers={"x-key": key}, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data", [])
        if params["page"] == 1 and data:
            log.info("Поля записи: %s", list(data[0].keys()))
        rows.extend(data)
        if params["page"] >= int(payload.get("last_page", 1)):
            break
        params["page"] += 1

    df = pd.DataFrame(rows)
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"Нет полей {missing}; посмотри лог с полями и поправь COLS")
    df = df[COLS].copy()
    df["gasDayStart"] = pd.to_datetime(df["gasDayStart"])
    for c in COLS[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.set_index("gasDayStart").sort_index()
    df.columns = [f"gas_storage_{c}" for c in df.columns]
    out = RAW_DIR / "gas_storage.parquet"
    df.to_parquet(out)
    log.info("Сохранено %s дней в %s", len(df), out)


if __name__ == "__main__":
    main()
