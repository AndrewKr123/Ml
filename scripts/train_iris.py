"""Полный пайплайн Iris: baseline vs итоговые модели.

train -> hold-out test сплит -> baseline LogisticRegression ->
final LogisticRegression с CV ->
SVM Linear / SVM RBF / DecisionTree -> метрики на hold-out ->
сохранение модели/метрик/графиков в models/iris/.

Запуск:
    python scripts/train_iris.py
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
from src.iris_preprocessing import load_raw, prepare_features

from src.experiments.iris import (
    train_baseline,
    train_logistic_cv,
    train_svm_linear_cv,
    train_svm_rbf_cv,
    train_decision_tree_cv,
)
from src.reporting.iris import save_report
from utils.plotting import set_style

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.svm")

MODELS_DIR = Path("models/iris")
PROCESSED_DIR = Path("data/iris/processed")


def run(test_size: float = 0.2, random_state: int = 42) -> None:
    set_style()

    df = load_raw()

    if "target" not in df.columns:
        raise ValueError("В датасете нет колонки target. Проверь load_raw().")

    train_df, holdout_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["target"],
        random_state=random_state,
    )

    X_train = prepare_features(train_df)
    X_holdout = prepare_features(holdout_df)

    y_train = train_df["target"]
    y_holdout = holdout_df["target"]

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    baseline = train_baseline(X_train, y_train, X_holdout, y_holdout, random_state)
    lr = train_logistic_cv(X_train, y_train, X_holdout, y_holdout, cv, random_state)

    print("\n=== BASELINE vs FINAL (hold-out test) ===")
    for key, final_value in lr.metrics.items():
        base_value = baseline.metrics.get(key)

        if base_value is None or final_value is None:
            continue

        print(
            f"{key:>15}: {base_value:.4f} -> {final_value:.4f}  "
            f"(delta {final_value - base_value:+.4f})"
        )

    svm_linear = train_svm_linear_cv(X_train, y_train, X_holdout, y_holdout, cv, random_state)
    svm_rbf = train_svm_rbf_cv(X_train, y_train, X_holdout, y_holdout, cv, random_state)
    tree = train_decision_tree_cv(X_train, y_train, X_holdout, y_holdout, cv, random_state)

    # ================= Сравнение всех моделей =================
    print("\n=== СРАВНЕНИЕ ВСЕХ МОДЕЛЕЙ (hold-out test) ===")

    models_comp = {
        "Baseline (LR)": baseline.metrics,
        "Final (LR)": lr.metrics,
        "SVM (Linear)": svm_linear.metrics,
        "SVM (RBF)": svm_rbf.metrics,
        "Decision Tree": tree.metrics,
    }

    for m_name, m_metrics in models_comp.items():
        roc_auc_value = m_metrics.get("roc_auc_ovr")
        roc_auc_text = f"{roc_auc_value:.4f}" if roc_auc_value is not None else "N/A"

        print(
            f"{m_name:<18} | "
            f"Accuracy: {m_metrics.get('accuracy', 0):.4f} | "
            f"F1 macro: {m_metrics.get('f1_macro', 0):.4f} | "
            f"ROC-AUC OVR: {roc_auc_text}"
        )

    save_report(
        train_df=train_df,
        holdout_df=holdout_df,
        X_train=X_train,
        X_holdout=X_holdout,
        y_train=y_train,
        y_holdout=y_holdout,
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

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    run(
        test_size=args.test_size,
        random_state=args.random_state,
    )
