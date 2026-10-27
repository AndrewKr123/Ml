"""Явные пайплайны и сетки параметров для Iris."""

from __future__ import annotations

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.evaluation import classification_metrics_multiclass, print_metrics
from src.datasets.iris import build_iris_pipeline
from src.classical.result import TrainingResult

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


def train_xgboost_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    cv: StratifiedKFold,
    random_state: int,
    *,
    n_jobs: int = -1,
) -> TrainingResult:
    """Подбирает XGBoost для многоклассовой классификации Iris."""
    pipeline = build_iris_pipeline(
        XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=1,
        )
    )
    params = {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [2, 3, 4],
        "model__learning_rate": [0.03, 0.1],
        "model__subsample": [0.8, 1.0],
    }
    search = GridSearchCV(pipeline, params, scoring="accuracy", cv=cv, n_jobs=n_jobs)
    search.fit(X_train, y_train)
    model = search.best_estimator_
    pred = model.predict(X_holdout)
    prob = model.predict_proba(X_holdout)
    metrics = classification_metrics_multiclass(y_holdout, pred, prob)
    print_metrics(f"XGBoost (best params={search.best_params_})", metrics)
    return TrainingResult(
        model=model, metrics=metrics, predictions=pred, probabilities=prob, search=search
    )


def train_random_forest_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    cv: StratifiedKFold,
    random_state: int,
    *,
    n_jobs: int = -1,
) -> TrainingResult:
    """Подбирает случайный лес на train; возвращает метрики train и holdout."""
    pipeline = build_iris_pipeline(
        RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=1)
    )
    params = {
        "model__max_depth": [None, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", 0.8],
    }
    search = GridSearchCV(pipeline, params, scoring="accuracy", cv=cv, n_jobs=n_jobs)
    search.fit(X_train, y_train)
    model = search.best_estimator_
    pred = model.predict(X_holdout)
    prob = model.predict_proba(X_holdout)
    metrics = classification_metrics_multiclass(y_holdout, pred, prob)
    print_metrics(f"Random Forest (best params={search.best_params_})", metrics)
    train_metrics = classification_metrics_multiclass(y_train, model.predict(X_train), model.predict_proba(X_train))
    print_metrics("Random Forest: train", train_metrics)
    return TrainingResult(
        model=model, metrics=metrics, predictions=pred, probabilities=prob, search=search, train_metrics=train_metrics
    )


def train_knn_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    cv: StratifiedKFold,
    random_state: int,
    *,
    n_jobs: int = 1,
) -> TrainingResult:
    """Подбирает KNN на train; возвращает метрики train и holdout."""
    pipeline = build_iris_pipeline(
        KNeighborsClassifier(n_jobs=1)
    )
    params = {
        "model__n_neighbors": [3, 5, 7, 11, 15, 21],
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2],
    }

    search = GridSearchCV(pipeline, params, scoring="accuracy", cv=cv, n_jobs=n_jobs)
    search.fit(X_train, y_train)
    model = search.best_estimator_
    pred = model.predict(X_holdout)
    prob = model.predict_proba(X_holdout)
    metrics = classification_metrics_multiclass(y_holdout, pred, prob)
    print_metrics(f"KNN (best params={search.best_params_})", metrics)
    train_metrics = classification_metrics_multiclass(y_train, model.predict(X_train), model.predict_proba(X_train))
    print_metrics("KNN: train", train_metrics)
    return TrainingResult(
        model=model, metrics=metrics, predictions=pred, probabilities=prob, search=search, train_metrics=train_metrics
    )
