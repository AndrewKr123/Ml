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
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from src.house_prices_preprocessing import (
    TARGET, build_preprocessor, get_final_feature_columns, engineer_and_clean,
    fill_missing, load_raw, log_transform_skewed,
)

from src.experiments.house_prices import (
    train_baseline,
    train_ridge_cv,
    train_decision_tree_cv,
)
from src.reporting.house_prices import save_report
from utils.plotting import set_style

MODELS_DIR = Path("models/house_prices")
PROCESSED_DIR = Path("data/house_prices/processed")


def run(test_size: float = 0.2, random_state: int = 42) -> None:
    set_style()
    train_raw, test_raw = load_raw()

    train_df, holdout_df = train_test_split(train_raw, test_size=test_size, random_state=random_state)

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
    ridge = train_ridge_cv(train_fe, holdout_fe, preprocessor, y_train_log, cv, random_state)

    # ================= Decision Tree =================
    tree = train_decision_tree_cv(
        train_fe, holdout_fe, preprocessor, y_train_log, cv, random_state
    )

    print("\n=== BASELINE vs FINAL (hold-out test, $) ===")
    for key, final_value in ridge.metrics.items():
        base_value = baseline.metrics[key]
        print(f"{key:>5}: {base_value:,.2f} -> {final_value:,.2f}  (delta {final_value - base_value:+,.2f})")

    save_report(
        train_fe=train_fe,
        holdout_fe=holdout_fe,
        test_fe=test_fe,
        test_raw=test_raw,
        baseline=baseline,
        ridge=ridge,
        tree=tree,
        models_dir=MODELS_DIR,
        processed_dir=PROCESSED_DIR,
        test_size=test_size,
        random_state=random_state,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    run(test_size=args.test_size, random_state=args.random_state)
