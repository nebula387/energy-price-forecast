"""
Тест Диболда–Мариано: значима ли разница в точности двух моделей.

Запуск (после src.backtest):
    python -m src.dm_test

Вариант, принятый в литературе по прогнозу цен (Lago et al., 2021): потери считаются по дням —
средняя абсолютная ошибка за 24 часа дня поставки, — потому что все часы дня прогнозируются одновременно.
d_t = loss_A(t) - loss_B(t); H1 (односторонняя): модель A точнее модели B.
Дисперсия — Newey–West, т.к. дневные потери автокоррелированы.
"""
import numpy as np
import pandas as pd
from scipy import stats

from src.config import PROCESSED_DIR, REPORTS_DIR


def daily_loss(pred: pd.DataFrame, model: str) -> pd.Series:
    return (pred["price"] - pred[model]).abs().groupby(pred["delivery_date"]).mean()


def dm_pvalue(loss_a: pd.Series, loss_b: pd.Series) -> float:
    """p-value для H1: A точнее B. Малое значение — A значимо лучше."""
    d = (loss_a - loss_b).dropna().to_numpy()
    n = len(d)
    lags = int(np.floor(n ** (1 / 3)))
    dc = d - d.mean()
    var = dc @ dc / n
    for k in range(1, lags + 1):
        var += 2 * (1 - k / (lags + 1)) * (dc[k:] @ dc[:-k]) / n
    stat = d.mean() / np.sqrt(var / n)
    return float(stats.norm.cdf(stat))


def main() -> None:
    pred = pd.read_parquet(PROCESSED_DIR / "predictions.parquet").dropna(subset=["naive"])
    models = [c for c in pred.columns if c not in ("delivery_date", "period", "price")]
    rows = []
    for period, g in pred.groupby("period", sort=False):
        losses = {m: daily_loss(g, m) for m in models}
        for a in models:
            for b in models:
                if a != b:
                    rows.append({"period": period, "model_a": a, "model_b": b,
                                 "p_value_a_better": round(dm_pvalue(losses[a], losses[b]), 4)})
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS_DIR / "dm_test.csv", index=False)

    for period, g in out.groupby("period", sort=False):
        print(f"\n{period}: p-value «строка точнее столбца» (< 0.05 — значимо)")
        print(g.pivot(index="model_a", columns="model_b", values="p_value_a_better").round(3).to_string())


if __name__ == "__main__":
    main()
