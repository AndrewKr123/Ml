"""Явные пайплайны и сетки параметров для Titanic."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.evaluation import classification_metrics, print_metrics
from src.titanic_preprocessing import build_baseline_pipeline, build_final_pipeline
from src.experiments.result import TrainingResult

C_GRID = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]


def train_baseline(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    random_state: int,
) -> TrainingResult:
    """Обучает baseline на train и оценивает на holdout."""
    model = build_baseline_pipeline(LogisticRegression(max_iter=1000, random_state=random_state))
    model.fit(train_df, train_df["Survived"])
    pred = model.predict(holdout_df)
    prob = model.predict_proba(holdout_df)[:, 1]
    metrics = classification_metrics(holdout_df["Survived"], pred, prob)
    print_metrics("BASELINE: LogisticRegression без feature engineering", metrics)
    return TrainingResult(model=model, metrics=metrics, predictions=pred, probabilities=prob)


def train_logistic_cv(
    train_fe: pd.DataFrame,
    holdout_fe: pd.DataFrame,
    cv: StratifiedKFold,
    random_state: int,
    *,
    n_jobs: int = -1,
) -> TrainingResult:
    """Подбирает параметры только на train; оценивает лучшую модель на holdout."""
    pipeline = build_final_pipeline(LogisticRegression(max_iter=1000, random_state=random_state))
    grid = GridSearchCV(pipeline, {"model__C": C_GRID}, scoring="roc_auc", cv=cv, n_jobs=n_jobs)
    grid.fit(train_fe, train_fe["Survived"])
    print(f"\nЛучший C по CV (roc_auc={grid.best_score_:.4f}): {grid.best_params_['model__C']}")
    model = grid.best_estimator_
    pred = model.predict(holdout_fe)
    prob = model.predict_proba(holdout_fe)[:, 1]
    metrics = classification_metrics(holdout_fe["Survived"], pred, prob)
    print_metrics("FINAL: инженерия признаков + CV-подбор C", metrics)
    return TrainingResult(
        model=model, metrics=metrics, predictions=pred, probabilities=prob, search=grid
    )


def train_svm_linear_cv(
    train_fe: pd.DataFrame,
    holdout_fe: pd.DataFrame,
    cv: StratifiedKFold,
    random_state: int,
    *,
    n_jobs: int = -1,
) -> TrainingResult:
    """Подбирает параметры только на train; оценивает лучшую модель на holdout."""
    pipeline = build_final_pipeline(SVC(kernel="linear", probability=True, random_state=random_state))
    grid = GridSearchCV(pipeline, {"model__C": C_GRID}, scoring="roc_auc", cv=cv, n_jobs=n_jobs)
    grid.fit(train_fe, train_fe["Survived"])
    model = grid.best_estimator_
    pred = model.predict(holdout_fe)
    prob = model.predict_proba(holdout_fe)[:, 1]
    metrics = classification_metrics(holdout_fe["Survived"], pred, prob)
    print_metrics(f"SVM Linear (best C={grid.best_params_['model__C']})", metrics)
    return TrainingResult(
        model=model, metrics=metrics, predictions=pred, probabilities=prob, search=grid
    )


def train_svm_rbf_cv(
    train_fe: pd.DataFrame,
    holdout_fe: pd.DataFrame,
    cv: StratifiedKFold,
    random_state: int,
    *,
    n_jobs: int = -1,
) -> TrainingResult:
    """Подбирает параметры только на train; оценивает лучшую модель на holdout."""
    pipeline = build_final_pipeline(SVC(kernel="rbf", probability=True, random_state=random_state))
    params = {"model__C": C_GRID, "model__gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1.0]}
    grid = GridSearchCV(pipeline, params, scoring="roc_auc", cv=cv, n_jobs=n_jobs)
    grid.fit(train_fe, train_fe["Survived"])
    model = grid.best_estimator_
    pred = model.predict(holdout_fe)
    prob = model.predict_proba(holdout_fe)[:, 1]
    metrics = classification_metrics(holdout_fe["Survived"], pred, prob)
    print_metrics(f"SVM RBF (best params={grid.best_params_})", metrics)
    return TrainingResult(
        model=model, metrics=metrics, predictions=pred, probabilities=prob, search=grid
    )


def train_decision_tree_cv(
    train_fe: pd.DataFrame,
    holdout_fe: pd.DataFrame,
    cv: StratifiedKFold,
    random_state: int,
    *,
    n_jobs: int = -1,
) -> TrainingResult:
    """Подбирает параметры только на train; оценивает лучшую модель на holdout."""
    pipeline = build_final_pipeline(DecisionTreeClassifier(random_state=random_state))
    params = {
        "model__criterion": ["gini", "entropy"],
        "model__max_depth": [2, 3, 4, 5, 6, 7, 8, None],
        "model__min_samples_split": [2, 5, 10, 20],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__class_weight": [None, "balanced"],
    }
    grid = GridSearchCV(pipeline, params, scoring="roc_auc", cv=cv, n_jobs=n_jobs)
    grid.fit(train_fe, train_fe["Survived"])
    model = grid.best_estimator_
    pred = model.predict(holdout_fe)
    prob = model.predict_proba(holdout_fe)[:, 1]
    metrics = classification_metrics(holdout_fe["Survived"], pred, prob)
    print_metrics(f"Decision Tree (best params={grid.best_params_})", metrics)
    return TrainingResult(
        model=model, metrics=metrics, predictions=pred, probabilities=prob, search=grid
    )
