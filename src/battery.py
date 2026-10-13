"""
Ценность прогноза в деньгах: аккумулятор 1 МВт / 1 МВт·ч на рынке на сутки вперёд.

Запуск (после src.backtest):
    python -m src.battery

Стратегия: один цикл в день. По ПРОГНОЗУ выбираем час заряда i и более поздний час разряда j,
максимизирующие eff * p[j] - p[i]; торгуем, только если ожидаемая прибыль положительна.
Доход считаем по ФАКТИЧЕСКИМ ценам. Идеальный бенчмарк — та же стратегия по фактическим ценам.
"""
import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, REPORTS_DIR

EFFICIENCY = 0.88  # round-trip КПД литий-ионного аккумулятора


def best_cycle(prices: np.ndarray) -> tuple[int, int, float]:
    """Лучшая пара (заряд i < разряд j) за один проход."""
    best, best_i, best_j = 0.0, -1, -1
    min_i = 0
    for j in range(1, len(prices)):
        if prices[j - 1] < prices[min_i]:
            min_i = j - 1
        gain = EFFICIENCY * prices[j] - prices[min_i]
        if gain > best:
            best, best_i, best_j = gain, min_i, j
    return best_i, best_j, best


def simulate(pred: pd.DataFrame, model: str) -> pd.DataFrame:
    rows = []
    for (period, day), g in pred.groupby(["period", "delivery_date"], sort=True):
        g = g.dropna(subset=["price", model])
        if len(g) < 20:
            continue
        actual = g["price"].to_numpy()
        i, j, _ = best_cycle(g[model].to_numpy())
        realised = EFFICIENCY * actual[j] - actual[i] if i >= 0 else 0.0
        _, _, perfect = best_cycle(actual)
        rows.append({"period": period, "date": day, "model": model,
                     "revenue": realised, "perfect": perfect})
    return pd.DataFrame(rows)


def main() -> None:
    pred = pd.read_parquet(PROCESSED_DIR / "predictions.parquet")
    pred = pred.sort_index()
    models = [c for c in pred.columns if c not in ("delivery_date", "period", "price")]
    daily = pd.concat(simulate(pred, m) for m in models)

    summary = (daily.groupby(["period", "model"])
               .agg(days=("date", "nunique"), revenue_eur=("revenue", "sum"), perfect_eur=("perfect", "sum"))
               .reset_index())
    summary["capture_rate"] = summary["revenue_eur"] / summary["perfect_eur"]
    summary["revenue_per_day"] = summary["revenue_eur"] / summary["days"]
    summary = summary.round(3).sort_values(["period", "capture_rate"], ascending=[True, False])
    summary.to_csv(REPORTS_DIR / "battery.csv", index=False)
    daily.to_parquet(PROCESSED_DIR / "battery_daily.parquet")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
