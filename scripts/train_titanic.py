"""Полный пайплайн Titanic: baseline vs итоговая логистическая регрессия.

train -> hold-out test сплит -> baseline (без инженерии признаков) ->
final (инженерия признаков + CV-подбор C) -> метрики на одном и том же
hold-out сплите -> сохранение модели/метрик/графиков в models/titanic/.

Запуск:
    python scripts/train_titanic.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from src.evaluation import classification_metrics, print_metrics, save_json
from src.titanic_preprocessing import (
    build_baseline_pipeline,
    build_final_pipeline,
    engineer_features,
    impute_age_by_title,
    load_raw,
)
from utils.plotting import plot_coefficients, plot_confusion_matrix, plot_roc_curve, set_style

MODELS_DIR = Path("models/titanic")
PROCESSED_DIR = Path("data/titanic/processed")
C_GRID = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]


def run(test_size: float = 0.2, random_state: int = 42) -> None:
    set_style()
    train_raw, test_raw = load_raw()

    train_df, holdout_df = train_test_split(
        train_raw, test_size=test_size, stratify=train_raw["Survived"], random_state=random_state
    )

    # ================= BASELINE: минимум предобработки =================
    baseline_model = build_baseline_pipeline(LogisticRegression(max_iter=1000, random_state=random_state))
    baseline_model.fit(train_df, train_df["Survived"])
    baseline_pred = baseline_model.predict(holdout_df)
    baseline_prob = baseline_model.predict_proba(holdout_df)[:, 1]
    baseline_metrics = classification_metrics(holdout_df["Survived"], baseline_pred, baseline_prob)
    print_metrics("BASELINE: LogisticRegression без feature engineering", baseline_metrics)

    # ================= FINAL: инженерия признаков + CV =================
    train_fe = engineer_features(train_df)
    holdout_fe = engineer_features(holdout_df)
    test_fe = engineer_features(test_raw)
    train_fe, holdout_fe, test_fe = impute_age_by_title(train_fe, holdout_fe, test_fe)

    pipeline = build_final_pipeline(LogisticRegression(max_iter=1000, random_state=random_state))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    grid = GridSearchCV(pipeline, {"model__C": C_GRID}, scoring="roc_auc", cv=cv, n_jobs=-1)
    grid.fit(train_fe, train_fe["Survived"])
    print(f"\nЛучший C по CV (roc_auc={grid.best_score_:.4f}): {grid.best_params_['model__C']}")

    final_model = grid.best_estimator_
    final_pred = final_model.predict(holdout_fe)
    final_prob = final_model.predict_proba(holdout_fe)[:, 1]
    final_metrics = classification_metrics(holdout_fe["Survived"], final_pred, final_prob)
    print_metrics("FINAL: инженерия признаков + CV-подбор C", final_metrics)

    print("\n=== BASELINE vs FINAL (hold-out test) ===")
    for key, final_value in final_metrics.items():
        base_value = baseline_metrics[key]
        print(f"{key:>10}: {base_value:.4f} -> {final_value:.4f}  (delta {final_value - base_value:+.4f})")

    # ================= сохранение артефактов =================
    (MODELS_DIR / "plots").mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(final_model, MODELS_DIR / "model.joblib")
    save_json(
        {
            "baseline": baseline_metrics,
            "final": final_metrics,
            "best_C": grid.best_params_["model__C"],
            "cv_roc_auc": grid.best_score_,
            "test_size": test_size,
            "random_state": random_state,
        },
        MODELS_DIR / "metrics.json",
    )

    plot_confusion_matrix(holdout_fe["Survived"], final_pred, labels=("Died", "Survived"))
    plt.savefig(MODELS_DIR / "plots" / "confusion_matrix.png", dpi=120, bbox_inches="tight")
    plt.close()

    plot_roc_curve(holdout_fe["Survived"], final_prob)
    plt.savefig(MODELS_DIR / "plots" / "roc_curve.png", dpi=120, bbox_inches="tight")
    plt.close()

    feature_names = final_model.named_steps["preprocess"].get_feature_names_out()
    coefficients = final_model.named_steps["model"].coef_[0]
    plot_coefficients(feature_names, coefficients, title="Коэффициенты логистической регрессии (Titanic)")
    plt.savefig(MODELS_DIR / "plots" / "coefficients.png", dpi=120, bbox_inches="tight")
    plt.close()

    train_fe.to_csv(PROCESSED_DIR / "train_processed.csv", index=False)
    test_fe.to_csv(PROCESSED_DIR / "test_processed.csv", index=False)

    # Kaggle test.csv не содержит Survived — используем его только для submission-файла,
    # а не для метрик (метрики выше честно посчитаны на hold-out из train.csv).
    submission = pd.DataFrame({
        "PassengerId": test_raw["PassengerId"],
        "Survived": final_model.predict(test_fe),
    })
    submission.to_csv(MODELS_DIR / "submission.csv", index=False)

    print(f"\nАртефакты сохранены в {MODELS_DIR}/ и {PROCESSED_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    run(test_size=args.test_size, random_state=args.random_state)
