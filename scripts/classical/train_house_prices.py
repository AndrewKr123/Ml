"""Полный пайплайн House Prices: baseline LinearRegression vs Ridge с полной предобработкой.

train -> hold-out test сплит -> baseline (сырые числовые признаки) ->
final (инженерия + чистка мультиколлинеарности + log-target + Ridge с CV-подбором
alpha) -> метрики в долларах на одном и том же hold-out сплите -> сохранение
модели/метрик/графиков в artifacts/house_prices/<model>/<run_id>/.

Запуск:
    python scripts/train_house_prices.py
"""

from __future__ import annotations

import argparse
from time import perf_counter
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import numpy as np
from sklearn.model_selection import KFold
from src.datasets.house_prices import (
    TARGET, build_preprocessor, get_final_feature_columns, engineer_and_clean,
    fill_missing, load_raw, log_transform_skewed,
)

from src.classical.house_prices import (
    train_baseline,
    train_random_forest_cv,
    train_knn_cv,
    train_ridge_cv,
    train_decision_tree_cv,
    train_lightgbm_cv,
)
from src.reporting.house_prices import save_report
from src.reporting.random_forest import save_random_forest_report
from src.reporting.knn import save_knn_report
from utils.plotting import set_style
from src.paths import DATA
from src.reporting.runs import create_run, finish_run, classical_metrics
from src.evaluation import save_json
from src.datasets.splits import split_holdout

PROCESSED_DIR = DATA / "house_prices" / "processed"


def run(test_size: float = 0.2, random_state: int = 42, random_forest_only: bool = False, knn_only: bool = False) -> None:
    if random_forest_only and knn_only:
        raise ValueError("Выберите только один режим обучения")
    started = perf_counter()
    config = {'test_size': test_size, 'random_state': random_state,
              'random_forest_only': random_forest_only, 'knn_only': knn_only,
              'preprocessing_protocol': 'legacy-v1',
              'prediction_scale': "dollars (model predicts log1p)"}
    models_dir = create_run('house_prices', 'random_forest' if random_forest_only else 'knn' if knn_only else 'classical_suite', config)
    set_style()
    train_raw, test_raw = load_raw()

    train_df, holdout_df = split_holdout(train_raw, TARGET, test_size=test_size, random_state=random_state, stratified=False)

    save_json({'train': train_df.index.tolist(), 'test': holdout_df.index.tolist()}, models_dir / 'split.json')

    if not random_forest_only and not knn_only:
        baseline = train_baseline(train_df, holdout_df)

    # Общие признаки и log-target для Ridge и дерева.
    train_fe = engineer_and_clean(train_df)
    holdout_fe = engineer_and_clean(holdout_df)
    test_fe = engineer_and_clean(test_raw)
    train_fe, holdout_fe, test_fe = fill_missing(train_fe, holdout_fe, test_fe)
    train_fe, holdout_fe, test_fe = log_transform_skewed(train_fe, holdout_fe, test_fe)

    preprocessor = build_preprocessor(get_final_feature_columns(train_fe))
    y_train_log = np.log1p(train_fe[TARGET])

    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    if not random_forest_only:
        knn = train_knn_cv(train_fe, holdout_fe, preprocessor, y_train_log, cv, random_state)
        save_knn_report(
            knn, models_dir, test_size=test_size, random_state=random_state,
            test_fe=test_fe, test_raw=test_raw,
        )
        if knn_only:
            finish_run(models_dir, classical_metrics({'knn': knn}), elapsed_seconds=perf_counter() - started)
            print(f"Artifacts: {models_dir}")
            return

    if random_forest_only:
        forest = train_random_forest_cv(train_fe, holdout_fe, preprocessor, y_train_log, cv, random_state)
        save_random_forest_report(
            forest, models_dir, test_size=test_size, random_state=random_state,
            test_fe=test_fe, test_raw=test_raw,
        )
        finish_run(models_dir, classical_metrics({'random_forest': forest}), elapsed_seconds=perf_counter() - started)
        print(f"Artifacts: {models_dir}")
        return

    ridge = train_ridge_cv(train_fe, holdout_fe, preprocessor, y_train_log, cv, random_state)

    # ================= Decision Tree =================
    tree = train_decision_tree_cv(
        train_fe, holdout_fe, preprocessor, y_train_log, cv, random_state
    )
    boost = train_lightgbm_cv(
        train_fe, holdout_fe, preprocessor, y_train_log, cv, random_state
    )

    print("\n=== BASELINE vs FINAL (hold-out test, $) ===")
    for key, final_value in ridge.metrics.items():
        base_value = baseline.metrics[key]
        print(f"{key:>5}: {base_value:,.2f} -> {final_value:,.2f}  (delta {final_value - base_value:+,.2f})")

    forest = train_random_forest_cv(train_fe, holdout_fe, preprocessor, y_train_log, cv, random_state)

    save_report(
        train_fe=train_fe,
        holdout_fe=holdout_fe,
        test_fe=test_fe,
        test_raw=test_raw,
        baseline=baseline,
        ridge=ridge,
        tree=tree,
        boost=boost,
        forest=forest,
        models_dir=models_dir,
        processed_dir=PROCESSED_DIR,
        test_size=test_size,
        random_state=random_state,
    )

    save_knn_report(
        knn, models_dir, test_size=test_size, random_state=random_state,
        test_fe=test_fe, test_raw=test_raw,
    )

    finish_run(models_dir, classical_metrics({'baseline': baseline, 'ridge': ridge, 'decision_tree': tree, 'lightgbm': boost, 'random_forest': forest, 'knn': knn}), elapsed_seconds=perf_counter() - started)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--random-forest-only", action="store_true", help="Обучить и сохранить только случайный лес")
    parser.add_argument("--knn-only", action="store_true", help="Обучить и сохранить только KNN")
    args = parser.parse_args()
    run(test_size=args.test_size, random_state=args.random_state, random_forest_only=args.random_forest_only, knn_only=args.knn_only)
