"""
Проверка, годятся ли бесплатные котировки Yahoo Finance для цены газа и CO2.
Это НЕОФИЦИАЛЬНЫЙ источник: сначала проверяем покрытие и пропуски, потом решаем.

Запуск:
    python -m src.check_market_tickers
"""
import pandas as pd
import yfinance as yf

from src.config import RAW_DIR, START

CANDIDATES = {
    "ttf_gas": "TTF=F",  # голландский газ TTF, ближайший фьючерс, €/МВт·ч
    "carbon_proxy": "KRBN",  # ETF на глобальные квоты CO2 — только грубый прокси для EU ETS
}


def main() -> None:
    frames = {}
    for name, ticker in CANDIDATES.items():
        df = yf.download(ticker, start=START.strftime("%Y-%m-%d"), progress=False, auto_adjust=False)
        if df.empty:
            print(f"{name} ({ticker}): данных нет")
            continue
        close = df["Close"].squeeze()
        bdays = pd.bdate_range(close.index.min(), close.index.max())
        coverage = len(close.dropna()) / len(bdays)
        print(
            f"{name} ({ticker}): {close.index.min():%Y-%m-%d} → {close.index.max():%Y-%m-%d}, "
            f"{len(close)} строк, покрытие рабочих дней {coverage:.0%}, "
            f"последнее значение {float(close.dropna().iloc[-1]):.2f}"
        )
        frames[name] = close.rename(name)
    if frames:
        out = RAW_DIR / "market_prices.parquet"
        pd.concat(frames.values(), axis=1, sort=True).to_parquet(out)
        print(f"Сохранено в {out}")


if __name__ == "__main__":
    main()
