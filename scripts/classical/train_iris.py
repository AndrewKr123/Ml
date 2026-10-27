"""Полный пайплайн Iris: baseline vs итоговые модели.

train -> hold-out test сплит -> baseline LogisticRegression ->
final LogisticRegression с CV ->
SVM Linear / SVM RBF / DecisionTree -> метрики на hold-out ->
сохранение модели/метрик/графиков в artifacts/iris/<model>/<run_id>/.

Запуск:
    python scripts/train_iris.py
"""

from __future__ import annotations

import argparse
from time import perf_counter
import sys
import warnings
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")

from sklearn.model_selection import StratifiedKFold
from src.datasets.iris import load_raw, prepare_features

from src.classical.iris import (
    train_baseline,
    train_random_forest_cv,
    train_knn_cv,
    train_logistic_cv,
    train_svm_linear_cv,
    train_svm_rbf_cv,
    train_decision_tree_cv,
    train_xgboost_cv,
)
from src.reporting.iris import save_report
from src.reporting.random_forest import save_random_forest_report
from src.reporting.knn import save_knn_report
from utils.plotting import set_style
from src.paths import DATA
from src.reporting.runs import create_run, finish_run, classical_metrics
from src.evaluation import save_json
from src.datasets.splits import split_holdout

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.svm")

PROCESSED_DIR = DATA / "iris" / "processed"


def run(test_size: float = 0.2, random_state: int = 42, random_forest_only: bool = False, knn_only: bool = False) -> None:
    if random_forest_only and knn_only:
        raise ValueError("Выберите только один режим обучения")
    started = perf_counter()
    config = {'test_size': test_size, 'random_state': random_state,
              'random_forest_only': random_forest_only, 'knn_only': knn_only,
              'preprocessing_protocol': 'legacy-v1',
              'prediction_scale': "class_id"}
    models_dir = create_run('iris', 'random_forest' if random_forest_only else 'knn' if knn_only else 'classical_suite', config)
    set_style()

    df = load_raw()

    if "target" not in df.columns:
        raise ValueError("В датасете нет колонки target. Проверь load_raw().")

    train_df, holdout_df = split_holdout(
        df, 'target', test_size=test_size, random_state=random_state,
    )

    save_json({'train': train_df.index.tolist(), 'test': holdout_df.index.tolist()}, models_dir / 'split.json')

    X_train = prepare_features(train_df)
    X_holdout = prepare_features(holdout_df)

    y_train = train_df["target"]
    y_holdout = holdout_df["target"]

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    if not random_forest_only:
        knn = train_knn_cv(X_train, y_train, X_holdout, y_holdout, cv, random_state)
        save_knn_report(
            knn, models_dir, test_size=test_size, random_state=random_state,
            y_holdout=y_holdout,
        )
        if knn_only:
            finish_run(models_dir, classical_metrics({'knn': knn}), elapsed_seconds=perf_counter() - started)
            print(f"Artifacts: {models_dir}")
            return

    if random_forest_only:
        forest = train_random_forest_cv(X_train, y_train, X_holdout, y_holdout, cv, random_state)
        save_random_forest_report(
            forest, models_dir, test_size=test_size, random_state=random_state,
            y_holdout=y_holdout,
        )
        finish_run(models_dir, classical_metrics({'random_forest': forest}), elapsed_seconds=perf_counter() - started)
        print(f"Artifacts: {models_dir}")
        return

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
    boost = train_xgboost_cv(X_train, y_train, X_holdout, y_holdout, cv, random_state)

    # ================= Сравнение всех моделей =================
    print("\n=== СРАВНЕНИЕ ВСЕХ МОДЕЛЕЙ (hold-out test) ===")

    models_comp = {
        "KNN": knn.metrics,
        "Baseline (LR)": baseline.metrics,
        "Final (LR)": lr.metrics,
        "SVM (Linear)": svm_linear.metrics,
        "SVM (RBF)": svm_rbf.metrics,
        "Decision Tree": tree.metrics,
        "XGBoost": boost.metrics,
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

    forest = train_random_forest_cv(X_train, y_train, X_holdout, y_holdout, cv, random_state)

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
        boost=boost,
        forest=forest,
        models_dir=models_dir,
        processed_dir=PROCESSED_DIR,
        test_size=test_size,
        random_state=random_state,
    )

    save_knn_report(
        knn, models_dir, test_size=test_size, random_state=random_state,
        y_holdout=y_holdout,
    )

    finish_run(models_dir, classical_metrics({'baseline': baseline, 'logistic': lr, 'svm_linear': svm_linear, 'svm_rbf': svm_rbf, 'decision_tree': tree, 'boost': boost, 'random_forest': forest, 'knn': knn}), elapsed_seconds=perf_counter() - started)


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

    parser.add_argument("--random-forest-only", action="store_true", help="Обучить и сохранить только случайный лес")
    parser.add_argument("--knn-only", action="store_true", help="Обучить и сохранить только KNN")
    args = parser.parse_args()

    run(
        test_size=args.test_size,
        random_state=args.random_state,
        random_forest_only=args.random_forest_only,
        knn_only=args.knn_only,
    )
