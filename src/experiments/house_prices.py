"""Явные пайплайны и сетки параметров для House Prices."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src.evaluation import print_metrics, regression_metrics
from src.house_prices_preprocessing import TARGET, get_baseline_numeric_columns
from src.experiments.result import TrainingResult

ALPHA_GRID = [0.1, 0.3, 1, 3, 10, 30, 100, 300]


def train_baseline(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
) -> TrainingResult:
    """Обучает baseline на train и оценивает на holdout."""
    numeric_cols = get_baseline_numeric_columns(train_df)
    pipeline = Pipeline([
        ("preprocess", ColumnTransformer([("num", SimpleImputer(strategy="median"), numeric_cols)])),
        ("model", LinearRegression()),
    ])
    pipeline.fit(train_df, train_df[TARGET])
    pred = pipeline.predict(holdout_df)
    metrics = regression_metrics(holdout_df[TARGET], pred)
    print_metrics("BASELINE: LinearRegression на сырых числовых признаках", metrics)
    return TrainingResult(model=pipeline, metrics=metrics, predictions=pred)


def train_ridge_cv(
    train_fe: pd.DataFrame,
    holdout_fe: pd.DataFrame,
    preprocessor: ColumnTransformer,
    y_train_log: pd.Series,
    cv: KFold,
    random_state: int,
    *,
    n_jobs: int = -1,
) -> TrainingResult:
    """CV на log-target; метрики и predictions в долларах.

    Принимает готовые признаки и y_train_log; копирует общий препроцессор.
    """
    pipeline = Pipeline([
        ("preprocess", clone(preprocessor)),
        ("model", Ridge(random_state=random_state)),
    ])
    grid = GridSearchCV(
        pipeline, {"model__alpha": ALPHA_GRID}, scoring="neg_root_mean_squared_error", cv=cv, n_jobs=n_jobs
    )
    grid.fit(train_fe, y_train_log)
    print(f"\nЛучшая alpha по CV (RMSE(log)={-grid.best_score_:.4f}): {grid.best_params_['model__alpha']}")
    model = grid.best_estimator_
    pred = np.expm1(model.predict(holdout_fe))
    metrics = regression_metrics(holdout_fe[TARGET], pred)
    print_metrics("FINAL: инженерия признаков + Ridge (CV-подбор alpha)", metrics)
    return TrainingResult(model=model, metrics=metrics, predictions=pred, search=grid)


def train_decision_tree_cv(
    train_fe: pd.DataFrame,
    holdout_fe: pd.DataFrame,
    preprocessor: ColumnTransformer,
    y_train_log: pd.Series,
    cv: KFold,
    random_state: int,
    *,
    n_jobs: int = -1,
) -> TrainingResult:
    """CV на log-target; метрики и predictions в долларах.

    Принимает готовые признаки и y_train_log; копирует общий препроцессор.
    """
    params = {
        "model__max_depth": [2, 3, 4, 5, 6, 7, 8, None],
        "model__min_samples_split": [2, 5, 10, 20],
        "model__min_samples_leaf": [1, 2, 4, 8],
    }
    pipeline = Pipeline([
        ("preprocess", clone(preprocessor)),
        ("model", DecisionTreeRegressor(random_state=random_state)),
    ])
    grid = GridSearchCV(pipeline, params, scoring="neg_root_mean_squared_error", cv=cv, n_jobs=n_jobs)
    grid.fit(train_fe, y_train_log)
    print(f"\nЛучшие параметры для дерева решений:{grid.best_params_}")
    model = grid.best_estimator_
    pred = np.expm1(model.predict(holdout_fe))
    metrics = regression_metrics(holdout_fe[TARGET], pred)
    print_metrics("Метрики дерева решений", metrics)
    return TrainingResult(model=model, metrics=metrics, predictions=pred, search=grid)
