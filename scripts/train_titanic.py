"""Полный пайплайн Titanic: baseline vs итоговая логистическая регрессия.

train -> hold-out test сплит -> baseline (без инженерии признаков) ->
final (инженерия признаков + CV-подбор C) -> метрики на одном и том же
hold-out сплите -> сохранение модели/метрик/графиков в models/titanic/.

Запуск:
    python scripts/train_titanic.py
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")

from sklearn.model_selection import StratifiedKFold, train_test_split
from src.titanic_preprocessing import engineer_features, impute_age_by_title, load_raw

from src.experiments.titanic import (
    train_baseline,
    train_logistic_cv,
    train_svm_linear_cv,
    train_svm_rbf_cv,
    train_decision_tree_cv,
)
from src.reporting.titanic import save_report
from utils.plotting import set_style

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.svm")

MODELS_DIR = Path("models/titanic")
PROCESSED_DIR = Path("data/titanic/processed")


def run(test_size: float = 0.2, random_state: int = 42) -> None:
    set_style()
    train_raw, test_raw = load_raw()

    train_df, holdout_df = train_test_split(
        train_raw, test_size=test_size, stratify=train_raw["Survived"], random_state=random_state
    )

    baseline = train_baseline(train_df, holdout_df, random_state)

    # Общие признаки для моделей с подбором параметров.
    train_fe = engineer_features(train_df)
    holdout_fe = engineer_features(holdout_df)
    test_fe = engineer_features(test_raw)
    train_fe, holdout_fe, test_fe = impute_age_by_title(train_fe, holdout_fe, test_fe)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    lr = train_logistic_cv(train_fe, holdout_fe, cv, random_state)

    print("\n=== BASELINE vs FINAL (hold-out test) ===")
    for key, final_value in lr.metrics.items():
        base_value = baseline.metrics[key]
        print(f"{key:>10}: {base_value:.4f} -> {final_value:.4f}  (delta {final_value - base_value:+.4f})")

    svm_linear = train_svm_linear_cv(train_fe, holdout_fe, cv, random_state)
    svm_rbf = train_svm_rbf_cv(train_fe, holdout_fe, cv, random_state)
    tree = train_decision_tree_cv(train_fe, holdout_fe, cv, random_state)

    print("\n=== СРАВНЕНИЕ ВСЕХ МОДЕЛЕЙ (hold-out test) ===")
    models_comp = {
        "Baseline (LR)": baseline.metrics,
        "Final (LR)": lr.metrics,
        "SVM (Linear)": svm_linear.metrics,
        "SVM (RBF)": svm_rbf.metrics,
        "Decision Tree": tree.metrics
    }
    for m_name, m_metrics in models_comp.items():
        print(
            f"{m_name:<15} | ROC-AUC: {m_metrics.get('roc_auc', 0):.4f} | Accuracy: {m_metrics.get('accuracy', 0):.4f}"
        )
    save_report(
        train_fe=train_fe,
        holdout_fe=holdout_fe,
        test_fe=test_fe,
        test_raw=test_raw,
        baseline=baseline,
        lr=lr,
        svm_linear=svm_linear,
        svm_rbf=svm_rbf,
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
