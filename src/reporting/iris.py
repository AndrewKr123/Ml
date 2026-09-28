"""Артефакты эксперимента iris."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation import save_json
from src.experiments.result import TrainingResult
from utils.plotting import plot_coefficients, plot_multiclass_confusion_matrix, plot_multiclass_roc

TARGET_NAMES = ["setosa", "versicolor", "virginica"]


def save_report(
    *,
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    X_train: pd.DataFrame,
    X_holdout: pd.DataFrame,
    y_train: pd.Series,
    y_holdout: pd.Series,
    baseline: TrainingResult,
    lr: TrainingResult,
    svm_linear: TrainingResult,
    svm_rbf: TrainingResult,
    tree: TrainingResult,
    models_dir: Path,
    processed_dir: Path,
    test_size: float,
    random_state: int,
) -> None:
    """Сохраняет артефакты эксперимента без повторного обучения."""

    (models_dir / "plots").mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(lr.model, models_dir / "model.joblib")
    joblib.dump(svm_rbf.model, models_dir / "svm_rbf.joblib")
    joblib.dump(tree.model, models_dir / "decision_tree.joblib")

    save_json(
        {
            "baseline": baseline.metrics,
            "final_lr": lr.metrics,
            "svm_linear": svm_linear.metrics,
            "svm_rbf": svm_rbf.metrics,
            "decision_tree": tree.metrics,
            "cv_accuracy_lr": float(lr.search.best_score_),
            "cv_accuracy_svm_linear": float(svm_linear.search.best_score_),
            "cv_accuracy_svm_rbf": float(svm_rbf.search.best_score_),
            "cv_accuracy_decision_tree": float(tree.search.best_score_),
            "target_names": TARGET_NAMES,
            "test_size": test_size,
            "random_state": random_state,
        },
        models_dir / "metrics.json",
    )

    # Confusion matrix для финальной модели
    y_holdout_display = (
        holdout_df["species"].to_numpy()
        if "species" in holdout_df.columns
        else np.asarray(TARGET_NAMES)[y_holdout.to_numpy()]
    )

    final_pred_display = np.asarray(TARGET_NAMES)[lr.predictions]

    plot_multiclass_confusion_matrix(
        y_holdout_display,
        final_pred_display,
        labels=TARGET_NAMES,
        title="Confusion matrix: финальная модель Iris",
    )

    plt.savefig(
        models_dir / "plots" / "confusion_matrix.png",
        dpi=120,
        bbox_inches="tight",
    )
    plt.close()

    # Многоклассовая ROC-кривая для финальной модели
    plot_multiclass_roc(
        y_holdout.to_numpy(),
        lr.probabilities,
        target_names=TARGET_NAMES,
    )

    plt.savefig(
        models_dir / "plots" / "roc_curve_ovr.png",
        dpi=120,
        bbox_inches="tight",
    )
    plt.close()

    # Коэффициенты логистической регрессии после масштабирования признаков
    try:
        feature_names = lr.model.named_steps["preprocess"].get_feature_names_out()
        feature_names = [name.replace("num__", "") for name in feature_names]

        coef = lr.model.named_steps["model"].coef_

        if coef.ndim == 1:
            plot_coefficients(
                feature_names,
                coef,
                title="Коэффициенты LogisticRegression (Iris)",
            )

            plt.savefig(
                models_dir / "plots" / "coefficients.png",
                dpi=120,
                bbox_inches="tight",
            )
            plt.close()
        else:
            for i, class_name in enumerate(TARGET_NAMES):
                if i < coef.shape[0]:
                    plot_coefficients(
                        feature_names,
                        coef[i],
                        title=f"Коэффициенты LogisticRegression: {class_name}",
                    )

                    plt.savefig(
                        models_dir / "plots" / f"coefficients_{class_name}.png",
                        dpi=120,
                        bbox_inches="tight",
                    )
                    plt.close()

    except Exception as e:
        print(f"Не удалось построить график коэффициентов: {e}")

    # Сохраняем обработанные данные
    train_processed = X_train.copy()
    train_processed["target"] = y_train.to_numpy()

    if "species" in train_df.columns:
        train_processed["species"] = train_df["species"].to_numpy()

    holdout_processed = X_holdout.copy()
    holdout_processed["target"] = y_holdout.to_numpy()

    if "species" in holdout_df.columns:
        holdout_processed["species"] = holdout_df["species"].to_numpy()

    train_processed.to_csv(
        processed_dir / "train_processed.csv",
        index=False,
    )

    holdout_processed.to_csv(
        processed_dir / "holdout_processed.csv",
        index=False,
    )

    # Небольшой artifact с предсказаниями на hold-out
    prediction_df = pd.DataFrame(
        {
            "y_true": y_holdout.to_numpy(),
            "y_pred": lr.predictions,
            "species_true": np.asarray(TARGET_NAMES)[y_holdout.to_numpy()],
            "species_pred": np.asarray(TARGET_NAMES)[lr.predictions],
        }
    )

    prediction_df.to_csv(
        models_dir / "holdout_predictions.csv",
        index=False,
    )

    print(f"\nАртефакты сохранены в {models_dir}/ и {processed_dir}/")
