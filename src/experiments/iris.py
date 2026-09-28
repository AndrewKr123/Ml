"""Явные пайплайны и сетки параметров для Iris."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.evaluation import classification_metrics_multiclass, print_metrics
from src.iris_preprocessing import build_iris_pipeline
from src.experiments.result import TrainingResult

C_GRID = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]


def train_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    random_state: int,
) -> TrainingResult:
    """Обучает baseline на train и оценивает на holdout."""
    model = build_iris_pipeline(LogisticRegression(max_iter=1000, random_state=random_state))
    model.fit(X_train, y_train)
    pred = model.predict(X_holdout)
    prob = model.predict_proba(X_holdout)
    metrics = classification_metrics_multiclass(y_holdout, pred, prob)
    print_metrics("BASELINE: LogisticRegression", metrics)
    return TrainingResult(model=model, metrics=metrics, predictions=pred, probabilities=prob)


def train_logistic_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    cv: StratifiedKFold,
    random_state: int,
    *,
    n_jobs: int = -1,
) -> TrainingResult:
    """Подбирает параметры только на train; оценивает лучшую модель на holdout."""
    pipeline = build_iris_pipeline(
        LogisticRegression(max_iter=1000, random_state=random_state),
    )
    search = GridSearchCV(pipeline, {"model__C": C_GRID}, scoring="accuracy", cv=cv, n_jobs=n_jobs)
    search.fit(X_train, y_train)
    print(
        f"\nЛучшие параметры LogisticRegression по CV "
        f"(accuracy={search.best_score_:.4f}): {search.best_params_}"
    )
    model = search.best_estimator_
    pred = model.predict(X_holdout)
    prob = model.predict_proba(X_holdout)
    metrics = classification_metrics_multiclass(y_holdout, pred, prob)
    print_metrics("FINAL: LogisticRegression + CV", metrics)
    return TrainingResult(
        model=model, metrics=metrics, predictions=pred, probabilities=prob, search=search
    )


def train_svm_linear_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    cv: StratifiedKFold,
    random_state: int,
    *,
    n_jobs: int = -1,
) -> TrainingResult:
    """Подбирает параметры только на train; оценивает лучшую модель на holdout."""
    pipeline = build_iris_pipeline(
        SVC(kernel="linear", probability=True, random_state=random_state),
    )
    search = GridSearchCV(pipeline, {"model__C": C_GRID}, scoring="accuracy", cv=cv, n_jobs=n_jobs)
    search.fit(X_train, y_train)
    model = search.best_estimator_
    pred = model.predict(X_holdout)
    prob = model.predict_proba(X_holdout)
    metrics = classification_metrics_multiclass(y_holdout, pred, prob)
    print_metrics("SVM Linear", metrics)
    return TrainingResult(
        model=model, metrics=metrics, predictions=pred, probabilities=prob, search=search
    )


def train_svm_rbf_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    cv: StratifiedKFold,
    random_state: int,
    *,
    n_jobs: int = -1,
) -> TrainingResult:
    """Подбирает параметры только на train; оценивает лучшую модель на holdout."""
    pipeline = build_iris_pipeline(
        SVC(kernel="rbf", probability=True, random_state=random_state),
    )
    params = {
        "model__C": C_GRID,
        "model__gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1.0],
    }
    search = GridSearchCV(pipeline, params, scoring="accuracy", cv=cv, n_jobs=n_jobs)
    search.fit(X_train, y_train)
    model = search.best_estimator_
    pred = model.predict(X_holdout)
    prob = model.predict_proba(X_holdout)
    metrics = classification_metrics_multiclass(y_holdout, pred, prob)
    print_metrics("SVM RBF", metrics)
    return TrainingResult(
        model=model, metrics=metrics, predictions=pred, probabilities=prob, search=search
    )


def train_decision_tree_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    cv: StratifiedKFold,
    random_state: int,
    *,
    n_jobs: int = -1,
) -> TrainingResult:
    """Подбирает параметры только на train; оценивает лучшую модель на holdout."""
    params = {
        "model__criterion": ["gini", "entropy"],
        "model__max_depth": [2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 25, 30, None],
        "model__min_samples_split": [2, 5, 10, 20],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__class_weight": [None, "balanced"],
    }
    pipeline = build_iris_pipeline(DecisionTreeClassifier(random_state=random_state))
    search = GridSearchCV(pipeline, params, scoring="accuracy", cv=cv, n_jobs=n_jobs)
    search.fit(X_train, y_train)
    model = search.best_estimator_
    pred = model.predict(X_holdout)
    prob = model.predict_proba(X_holdout)
    metrics = classification_metrics_multiclass(y_holdout, pred, prob)
    print_metrics("Decision Tree", metrics)
    return TrainingResult(
        model=model, metrics=metrics, predictions=pred, probabilities=prob, search=search
    )
