"""Полный пайплайн House Prices: baseline LinearRegression vs Ridge с полной предобработкой.

train -> hold-out test сплит -> baseline (сырые числовые признаки) ->
final (инженерия + чистка мультиколлинеарности + log-target + Ridge с CV-подбором
alpha) -> метрики в долларах на одном и том же hold-out сплите -> сохранение
модели/метрик/графиков в models/house_prices/.

Запуск:
    python scripts/train_house_prices.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline

from src.evaluation import print_metrics, regression_metrics, save_json
from src.house_prices_preprocessing import (
    TARGET,
    build_preprocessor,
    engineer_and_clean,
    fill_missing,
    get_baseline_numeric_columns,
    get_final_feature_columns,
    load_raw,
    log_transform_skewed,
)
from utils.plotting import plot_coefficients, plot_predicted_vs_actual, plot_residuals, set_style

MODELS_DIR = Path("models/house_prices")
PROCESSED_DIR = Path("data/house_prices/processed")
ALPHA_GRID = [0.1, 0.3, 1, 3, 10, 30, 100, 300]


def run(test_size: float = 0.2, random_state: int = 42) -> None:
    set_style()
    train_raw, test_raw = load_raw()

    train_df, holdout_df = train_test_split(train_raw, test_size=test_size, random_state=random_state)

    # ================= BASELINE: LinearRegression без предобработки =================
    numeric_cols = get_baseline_numeric_columns(train_df)
    baseline_pipeline = Pipeline([
        ("preprocess", ColumnTransformer([("num", SimpleImputer(strategy="median"), numeric_cols)])),
        ("model", LinearRegression()),
    ])
    baseline_pipeline.fit(train_df, train_df[TARGET])
    baseline_pred = baseline_pipeline.predict(holdout_df)
    baseline_metrics = regression_metrics(holdout_df[TARGET], baseline_pred)
    print_metrics("BASELINE: LinearRegression на сырых числовых признаках", baseline_metrics)

    # ================= FINAL: инженерия признаков + Ridge с CV =================
    train_fe = engineer_and_clean(train_df)
    holdout_fe = engineer_and_clean(holdout_df)
    test_fe = engineer_and_clean(test_raw)
    train_fe, holdout_fe, test_fe = fill_missing(train_fe, holdout_fe, test_fe)
    train_fe, holdout_fe, test_fe = log_transform_skewed(train_fe, holdout_fe, test_fe)

    feature_cols = get_final_feature_columns(train_fe)
    preprocessor = build_preprocessor(feature_cols)
    pipeline = Pipeline([("preprocess", preprocessor), ("model", Ridge(random_state=random_state))])

    y_train_log = np.log1p(train_fe[TARGET])
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    grid = GridSearchCV(
        pipeline, {"model__alpha": ALPHA_GRID}, scoring="neg_root_mean_squared_error", cv=cv, n_jobs=-1
    )
    grid.fit(train_fe, y_train_log)
    print(f"\nЛучшая alpha по CV (RMSE(log)={-grid.best_score_:.4f}): {grid.best_params_['model__alpha']}")

    final_model = grid.best_estimator_
    final_pred = np.expm1(final_model.predict(holdout_fe))
    final_metrics = regression_metrics(holdout_fe[TARGET], final_pred)
    print_metrics("FINAL: инженерия признаков + Ridge (CV-подбор alpha)", final_metrics)

    print("\n=== BASELINE vs FINAL (hold-out test, $) ===")
    for key, final_value in final_metrics.items():
        base_value = baseline_metrics[key]
        print(f"{key:>5}: {base_value:,.2f} -> {final_value:,.2f}  (delta {final_value - base_value:+,.2f})")

    # ================= сохранение артефактов =================
    (MODELS_DIR / "plots").mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(final_model, MODELS_DIR / "model.joblib")
    save_json(
        {
            "baseline": baseline_metrics,
            "final": final_metrics,
            "best_alpha": grid.best_params_["model__alpha"],
            "cv_rmse_log": -grid.best_score_,
            "test_size": test_size,
            "random_state": random_state,
        },
        MODELS_DIR / "metrics.json",
    )

    plot_residuals(holdout_fe[TARGET], final_pred, title="Residuals: Ridge, $ scale")
    plt.savefig(MODELS_DIR / "plots" / "residuals.png", dpi=120, bbox_inches="tight")
    plt.close()

    plot_predicted_vs_actual(holdout_fe[TARGET], final_pred, title="Predicted vs Actual SalePrice")
    plt.savefig(MODELS_DIR / "plots" / "predicted_vs_actual.png", dpi=120, bbox_inches="tight")
    plt.close()

    feature_names = final_model.named_steps["preprocess"].get_feature_names_out()
    coefficients = final_model.named_steps["model"].coef_
    plot_coefficients(feature_names, coefficients, title="Коэффициенты Ridge (House Prices, log-target)")
    plt.savefig(MODELS_DIR / "plots" / "coefficients.png", dpi=120, bbox_inches="tight")
    plt.close()

    train_fe.to_csv(PROCESSED_DIR / "train_processed.csv", index=False)
    test_fe.to_csv(PROCESSED_DIR / "test_processed.csv", index=False)

    # Kaggle test.csv не содержит SalePrice — используем его только для submission-файла,
    # а не для метрик (метрики выше честно посчитаны на hold-out из train.csv).
    submission = pd.DataFrame({
        "Id": test_raw["Id"],
        "SalePrice": np.expm1(final_model.predict(test_fe)),
    })
    submission.to_csv(MODELS_DIR / "submission.csv", index=False)

    print(f"\nАртефакты сохранены в {MODELS_DIR}/ и {PROCESSED_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    run(test_size=args.test_size, random_state=args.random_state)
