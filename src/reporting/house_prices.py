"""Артефакты эксперимента house_prices."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation import save_json
from src.experiments.result import TrainingResult
from src.house_prices_preprocessing import TARGET
from utils.plotting import plot_coefficients, plot_predicted_vs_actual, plot_residuals


def save_report(
    *,
    train_fe: pd.DataFrame,
    holdout_fe: pd.DataFrame,
    test_fe: pd.DataFrame,
    test_raw: pd.DataFrame,
    baseline: TrainingResult,
    ridge: TrainingResult,
    tree: TrainingResult,
    models_dir: Path,
    processed_dir: Path,
    test_size: float,
    random_state: int,
) -> None:
    """Сохраняет артефакты эксперимента без повторного обучения."""

    (models_dir / "plots").mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(ridge.model, models_dir / "model.joblib")
    save_json(
        {
            "baseline": baseline.metrics,
            "final": ridge.metrics,
            "decision tree": tree.metrics,
            "best_alpha": ridge.search.best_params_["model__alpha"],
            "cv_rmse_log": -ridge.search.best_score_,
            "best params for tree": tree.search.best_params_,
            "test_size": test_size,
            "random_state": random_state,
        },
        models_dir / "metrics.json",
    )

    plot_residuals(holdout_fe[TARGET], ridge.predictions, title="Residuals: Ridge, $ scale")
    plt.savefig(models_dir / "plots" / "residuals.png", dpi=120, bbox_inches="tight")
    plt.close()

    plot_predicted_vs_actual(holdout_fe[TARGET], ridge.predictions, title="Predicted vs Actual SalePrice")
    plt.savefig(models_dir / "plots" / "predicted_vs_actual.png", dpi=120, bbox_inches="tight")
    plt.close()

    feature_names = ridge.model.named_steps["preprocess"].get_feature_names_out()
    coefficients = ridge.model.named_steps["model"].coef_
    plot_coefficients(feature_names, coefficients, title="Коэффициенты Ridge (House Prices, log-target)")
    plt.savefig(models_dir / "plots" / "coefficients.png", dpi=120, bbox_inches="tight")
    plt.close()

    train_fe.to_csv(processed_dir / "train_processed.csv", index=False)
    test_fe.to_csv(processed_dir / "test_processed.csv", index=False)

    # Kaggle test.csv не содержит SalePrice — используем его только для submission-файла,
    # а не для метрик (метрики выше честно посчитаны на hold-out из train.csv).
    submission = pd.DataFrame({
        "Id": test_raw["Id"],
        "SalePrice": np.expm1(ridge.model.predict(test_fe)),
    })
    submission.to_csv(models_dir / "submission.csv", index=False)

    print(f"\nАртефакты сохранены в {models_dir}/ и {processed_dir}/")
