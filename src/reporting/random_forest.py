"""Артефакты случайного леса; сохранение не запускает повторное обучение."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.evaluation import save_json
from src.experiments.result import TrainingResult


def save_random_forest_report(
    result: TrainingResult,
    models_dir: Path,
    *,
    test_size: float,
    random_state: int,
    y_holdout: pd.Series | None = None,
    test_fe: pd.DataFrame | None = None,
    test_raw: pd.DataFrame | None = None,
) -> None:
    """Сохраняет train/test, CV и параметры, дополняет общий metrics.json."""
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.model, models_dir / "random_forest.joblib")
    payload = {
        "train": result.train_metrics,
        "test": result.metrics,
        "best_params": result.search.best_params_,
        "cv_scoring": result.search.scoring,
        "cv_best_score": float(result.search.best_score_),
        "test_size": test_size,
        "random_state": random_state,
    }
    save_json(payload, models_dir / "random_forest_metrics.json")
    metrics_path = models_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    metrics["random_forest"] = payload
    save_json(metrics, metrics_path)
    if y_holdout is not None:
        pd.DataFrame({"y_true": y_holdout.to_numpy(), "y_pred": result.predictions}).to_csv(
            models_dir / "random_forest_holdout_predictions.csv", index=False
        )
    elif test_fe is not None and test_raw is not None:
        pred = result.model.predict(test_fe)
        if result.probabilities is None:
            submission = pd.DataFrame({"Id": test_raw["Id"], "SalePrice": np.expm1(pred)})
        else:
            submission = pd.DataFrame({"PassengerId": test_raw["PassengerId"], "Survived": pred})
        submission.to_csv(models_dir / "random_forest_submission.csv", index=False)
