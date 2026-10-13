"""
Бэктест прогноза цены на сутки вперёд: «нормальный» год против периода кризиса.

Запуск:
    python -m src.backtest

Два тестовых периода:
- normal: 01.03.2025 – 27.02.2026 — контроль, как модели работают в обычное время;
- crisis: с 28.02.2026 (начало войны с Ираном) до последнего дня с данными.

Варианты моделей:
- naive          — цена неделю назад, тот же час;
- ridge / lgbm   — линейная регрессия и LightGBM;
- base / gas     — без и с газовыми признаками (TTF, CO2-прокси, хранилища);
- static         — обучена один раз на данных до начала периода;
- rolling        — переобучается каждые 7 дней на всех днях поставки до прогнозного.

Прогноз на день D строится только по строкам с днём поставки < D, все они известны к аукциону.
Цены с сайтов (прогнозы бирж, аналитиков) нигде не используются — только факты и прогнозы TSO.
"""
import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CRISIS_START, PROCESSED_DIR, REPORTS_DIR
from src.features import CALENDAR, FEATURE_SETS, TARGET, make_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("backtest")

RETRAIN_EVERY_DAYS = 7
PERIODS = {
    "normal": (pd.Timestamp("2025-03-01"), CRISIS_START.tz_localize(None) - pd.Timedelta(days=1)),
    "crisis": (CRISIS_START.tz_localize(None), None),
}


def make_model(kind: str, features: list[str]):
    if kind == "lgbm":
        return lgb.LGBMRegressor(
            n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=50,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=42,
        )
    categorical = [c for c in CALENDAR if c != "is_holiday"]
    numeric = [c for c in features if c not in categorical]
    prep = ColumnTransformer([
        ("cal", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
    ])
    return make_pipeline(prep, Ridge(alpha=1.0))


def fit_predict(kind: str, features: list[str], train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    train = train.dropna(subset=[TARGET, "price_d7_same_hour"])
    model = make_model(kind, features)
    model.fit(train[features], train[TARGET])
    return model.predict(test[features])


def run_period(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp | None) -> pd.DataFrame:
    end = end or df["delivery_date"].max()
    test_mask = df["delivery_date"].between(start, end)
    test = df.loc[test_mask, ["delivery_date", TARGET, "price_d7_same_hour"]].copy()
    test = test.rename(columns={"price_d7_same_hour": "naive"})

    for kind in ("ridge", "lgbm"):
        for fset, features in FEATURE_SETS.items():
            # static: один раз на всём, что было до начала периода
            history = df[df["delivery_date"] < start]
            test[f"{kind}_{fset}_static"] = fit_predict(kind, features, history, df[test_mask])

            # rolling: блоки по 7 дней, перед каждым блоком — переобучение на расширяющемся окне
            preds = pd.Series(index=test.index, dtype=float)
            for block_start in pd.date_range(start, end, freq=f"{RETRAIN_EVERY_DAYS}D"):
                block_end = block_start + pd.Timedelta(days=RETRAIN_EVERY_DAYS - 1)
                block = df["delivery_date"].between(block_start, block_end) & test_mask
                history = df[df["delivery_date"] < block_start]
                preds[block[block].index] = fit_predict(kind, features, history, df[block])
            test[f"{kind}_{fset}_rolling"] = preds
            log.info("  %s_%s готов", kind, fset)
    return test


def metrics(pred: pd.DataFrame) -> pd.DataFrame:
    models = [c for c in pred.columns if c not in ("delivery_date", "period", TARGET)]
    rows = []
    for period, g in pred.groupby("period", sort=False):
        g = g.dropna(subset=[TARGET, "naive"])
        mae_naive = (g[TARGET] - g["naive"]).abs().mean()
        for m in models:
            err = g[TARGET] - g[m]
            rows.append({
                "period": period, "model": m,
                "MAE": err.abs().mean(), "RMSE": np.sqrt((err ** 2).mean()),
                "rMAE_vs_naive": err.abs().mean() / mae_naive,
            })
    return pd.DataFrame(rows).round(3)


def main() -> None:
    df = make_features(pd.read_parquet(PROCESSED_DIR / "dataset.parquet"))
    results = []
    for name, (start, end) in PERIODS.items():
        log.info("Период %s: %s → %s", name, f"{start:%Y-%m-%d}", end and f"{end:%Y-%m-%d}")
        res = run_period(df, start, end)
        res["period"] = name
        results.append(res)
    pred = pd.concat(results)
    pred.to_parquet(PROCESSED_DIR / "predictions.parquet")

    REPORTS_DIR.mkdir(exist_ok=True)
    table = metrics(pred)
    table.to_csv(REPORTS_DIR / "metrics.csv", index=False)
    log.info("Метрики (€/МВт·ч):\n%s", table.sort_values(["period", "MAE"]).to_string(index=False))


if __name__ == "__main__":
    main()
