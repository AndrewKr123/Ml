"""Явные пайплайны и сетки параметров для Titanic."""

from __future__ import annotations

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.evaluation import classification_metrics, print_metrics
from src.datasets.titanic import build_baseline_pipeline, build_final_pipeline
from src.classical.result import TrainingResult

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


def train_catboost_cv(
    train_fe: pd.DataFrame,
    holdout_fe: pd.DataFrame,
    cv: StratifiedKFold,
    random_state: int,
    *,
    n_jobs: int = -1,
) -> TrainingResult:
    """Подбирает CatBoost для бинарной классификации Titanic."""
    pipeline = build_final_pipeline(
        CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            verbose=False,
            allow_writing_files=False,
            random_seed=random_state,
            thread_count=1,
        )
    )
    params = {
        "model__iterations": [150, 300],
        "model__depth": [3, 5, 7],
        "model__learning_rate": [0.03, 0.1],
        "model__l2_leaf_reg": [3, 7],
    }
    search = GridSearchCV(pipeline, params, scoring="roc_auc", cv=cv, n_jobs=n_jobs)
    search.fit(train_fe, train_fe["Survived"])
    model = search.best_estimator_
    pred = model.predict(holdout_fe).astype(int).ravel()
    prob = model.predict_proba(holdout_fe)[:, 1]
    metrics = classification_metrics(holdout_fe["Survived"], pred, prob)
    print_metrics(f"CatBoost (best params={search.best_params_})", metrics)
    return TrainingResult(
        model=model, metrics=metrics, predictions=pred, probabilities=prob, search=search
    )


def train_random_forest_cv(
    train_fe: pd.DataFrame,
    holdout_fe: pd.DataFrame,
    cv: StratifiedKFold,
    random_state: int,
    *,
    n_jobs: int = -1,
) -> TrainingResult:
    """Подбирает случайный лес на train; возвращает метрики train и holdout."""
    pipeline = build_final_pipeline(
        RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=1)
    )
    params = {
        "model__max_depth": [None, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", 0.8],
    }
    search = GridSearchCV(pipeline, params, scoring="roc_auc", cv=cv, n_jobs=n_jobs)
    search.fit(train_fe, train_fe["Survived"])
    model = search.best_estimator_
    pred = model.predict(holdout_fe).astype(int).ravel()
    prob = model.predict_proba(holdout_fe)[:, 1]
    metrics = classification_metrics(holdout_fe["Survived"], pred, prob)
    print_metrics(f"Random Forest (best params={search.best_params_})", metrics)
    train_metrics = classification_metrics(train_fe["Survived"], model.predict(train_fe), model.predict_proba(train_fe)[:, 1])
    print_metrics("Random Forest: train", train_metrics)
    return TrainingResult(
        model=model, metrics=metrics, predictions=pred, probabilities=prob, search=search, train_metrics=train_metrics
    )


def train_knn_cv(
    train_fe: pd.DataFrame,
    holdout_fe: pd.DataFrame,
    cv: StratifiedKFold,
    random_state: int,
    *,
    n_jobs: int = 1,
) -> TrainingResult:
    """Подбирает KNN на train; возвращает метрики train и holdout."""
    pipeline = build_final_pipeline(
        KNeighborsClassifier(n_jobs=1)
    )
    params = {
        "model__n_neighbors": [3, 5, 7, 11, 15, 21],
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2],
    }

    search = GridSearchCV(pipeline, params, scoring="roc_auc", cv=cv, n_jobs=n_jobs)
    search.fit(train_fe, train_fe["Survived"])
    model = search.best_estimator_
    pred = model.predict(holdout_fe).astype(int).ravel()
    prob = model.predict_proba(holdout_fe)[:, 1]
    metrics = classification_metrics(holdout_fe["Survived"], pred, prob)
    print_metrics(f"KNN (best params={search.best_params_})", metrics)
    train_metrics = classification_metrics(train_fe["Survived"], model.predict(train_fe), model.predict_proba(train_fe)[:, 1])
    print_metrics("KNN: train", train_metrics)
    return TrainingResult(
        model=model, metrics=metrics, predictions=pred, probabilities=prob, search=search, train_metrics=train_metrics
    )
