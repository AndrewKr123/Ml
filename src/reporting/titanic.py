"""Артефакты эксперимента titanic."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from src.evaluation import save_json
from src.experiments.result import TrainingResult
from utils.plotting import plot_coefficients, plot_confusion_matrix, plot_roc_curve


def save_report(
    *,
    train_fe: pd.DataFrame,
    holdout_fe: pd.DataFrame,
    test_fe: pd.DataFrame,
    test_raw: pd.DataFrame,
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
    joblib.dump(tree.model, models_dir / "decision_tree_model.joblib")
    save_json(
        {
            "baseline": baseline.metrics,
            "final_lr": lr.metrics,
            "svm_linear": svm_linear.metrics,
            "svm_rbf": svm_rbf.metrics,
            "dt_metrics": tree.metrics,
            "best_C_lr": lr.search.best_params_["model__C"],
            "best_params_svm_linear": svm_linear.search.best_params_,
            "best_params_svm_rbf": svm_rbf.search.best_params_,
            "best_params_dt": tree.search.best_params_,
            "cv_roc_auc_lr": lr.search.best_score_,
            "cv_roc_auc_svm_rbf": svm_rbf.search.best_score_,
            "cv_roc_auc_dt": tree.search.best_score_,
            "test_size": test_size,
            "random_state": random_state,
        },
        models_dir / "metrics.json",
    )

    plot_confusion_matrix(holdout_fe["Survived"], lr.predictions, labels=("Died", "Survived"))
    plt.savefig(models_dir / "plots" / "confusion_matrix.png", dpi=120, bbox_inches="tight")
    plt.close()

    plot_roc_curve(holdout_fe["Survived"], lr.probabilities)
    plt.savefig(models_dir / "plots" / "roc_curve.png", dpi=120, bbox_inches="tight")
    plt.close()

    feature_names = lr.model.named_steps["preprocess"].get_feature_names_out()
    coefficients = lr.model.named_steps["model"].coef_[0]
    plot_coefficients(feature_names, coefficients, title="Коэффициенты логистической регрессии (Titanic)")
    plt.savefig(models_dir / "plots" / "coefficients.png", dpi=120, bbox_inches="tight")
    plt.close()

    train_fe.to_csv(processed_dir / "train_processed.csv", index=False)
    test_fe.to_csv(processed_dir / "test_processed.csv", index=False)

    # Kaggle test.csv не содержит Survived — используем его только для submission-файла,
    # а не для метрик (метрики выше честно посчитаны на hold-out из train.csv).
    submission = pd.DataFrame({
        "PassengerId": test_raw["PassengerId"],
        "Survived": lr.model.predict(test_fe),
    })
    submission.to_csv(models_dir / "submission.csv", index=False)

    print(f"\nАртефакты сохранены в {models_dir}/ и {processed_dir}/")
