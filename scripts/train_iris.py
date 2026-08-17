"""Полный пайплайн Iris: baseline vs итоговые модели.

train -> hold-out test сплит -> baseline LogisticRegression ->
final LogisticRegression с CV и опциональным поиском PCA ->
SVM Linear / SVM RBF / DecisionTree -> метрики на hold-out ->
сохранение модели/метрик/графиков в models/iris/.

Запуск:
    python scripts/train_iris.py

Отключить поиск PCA:
    python scripts/train_iris.py --no-pca-search
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.evaluation import print_metrics, save_json
from src.iris_preprocessing import (
    build_iris_pipeline,
    load_raw,
    prepare_features,
)
from utils.plotting import plot_coefficients, set_style

# Если уже добавлен plot_multiclass_confusion_matrix в utils/plotting.py, будет использована она. Иначе скрипт использует запасную реализацию.
try:
    from utils.plotting import plot_multiclass_confusion_matrix
except ImportError:
    import seaborn as sns

    def plot_multiclass_confusion_matrix(
        y_true,
        y_pred,
        labels=None,
        normalize: bool = False,
        title: str = "Матрица ошибок",
        ax=None,
    ):
        unique_labels = np.unique(np.hstack([np.asarray(y_true), np.asarray(y_pred)]))

        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

        if normalize:
            row_sums = cm.sum(axis=1)[:, None]
            row_sums = np.where(row_sums == 0, 1, row_sums)
            cm_plot = cm.astype("float") / row_sums
            fmt = ".2f"
        else:
            cm_plot = cm
            fmt = "d"

        if labels is None:
            labels = unique_labels

        fig, ax = (None, ax) if ax is not None else plt.subplots(
            figsize=(0.65 * len(labels) + 2.5, 0.65 * len(labels) + 2.2)
        )

        sns.heatmap(
            cm_plot,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=list(labels),
            yticklabels=list(labels),
            linewidths=0.5,
            linecolor="white",
        )

        ax.set_xlabel("Предсказано")
        ax.set_ylabel("Реально")
        ax.set_title(title)

        return ax


# Убирает возможные FutureWarning от sklearn/SVC
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="sklearn.svm",
)

MODELS_DIR = Path("models/iris")
PROCESSED_DIR = Path("data/iris/processed")

C_GRID = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]

TARGET_NAMES = [
    "setosa",
    "versicolor",
    "virginica",
]


def _json_safe_params(params: dict) -> dict:
    """
    GridSearchCV может вернуть в best_params_ объект PCA.
    Такой объект нельзя напрямую сериализовать в JSON, поэтому делаем безопасное представление.
    """
    safe = {}

    for key, value in params.items():
        if isinstance(value, PCA):
            safe[key] = f"PCA(n_components={value.n_components})"
        elif isinstance(value, (np.integer,)):
            safe[key] = int(value)
        elif isinstance(value, (np.floating,)):
            safe[key] = float(value)
        elif value is None:
            safe[key] = None
        else:
            safe[key] = value

    return safe


def classification_metrics_multiclass(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_prob: np.ndarray | None = None,
) -> dict:
    """
    Метрики для многоклассовой классификации.

    y_prob должен иметь форму:
        (n_samples, n_classes)
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc_ovr"] = float(
                roc_auc_score(
                    y_true,
                    y_prob,
                    multi_class="ovr",
                    average="macro",
                )
            )
        except ValueError:
            metrics["roc_auc_ovr"] = None

    return metrics


def plot_multiclass_roc(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    target_names: list[str],
    ax=None,
):
    """
    Строит ROC-кривые one-vs-rest для многоклассовой классификации.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    classes = np.arange(y_prob.shape[1])
    y_bin = label_binarize(y_true, classes=classes)

    # На случай, если класс вдруг один/два
    if y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])

    n_classes = y_bin.shape[1]

    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    fig, ax = (None, ax) if ax is not None else plt.subplots(figsize=(6, 5))

    ax.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"micro-average (AUC = {roc_auc['micro']:.3f})",
        linewidth=2,
    )

    for i in range(n_classes):
        class_name = target_names[i] if i < len(target_names) else str(i)
        ax.plot(
            fpr[i],
            tpr[i],
            linestyle="--",
            label=f"{class_name} (AUC = {roc_auc[i]:.3f})",
        )

    ax.plot([0, 1], [0, 1], color="gray", linestyle=":")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC-кривые (one-vs-rest)")
    ax.legend(fontsize=8)

    return ax


def with_pca_options(param_grid: dict) -> list[dict]:
    """
    Делает список сеток для GridSearchCV:
    1. PCA выключен через 'passthrough'
    2. PCA включён с разными n_components
    """
    return [
        {
            "pca": ["passthrough"],
            **param_grid,
        },
        {
            "pca": [
                PCA(n_components=2),
                PCA(n_components=0.95),
                PCA(n_components=3),
                PCA(n_components=4),
                PCA(n_components=1.5)
            ],
            **param_grid,
        },
    ]


def make_pipeline_and_grid(
    model,
    param_grid: dict,
    pca_search: bool,
):
    """
    Собирает пайплайн и сетку параметров.

    Если pca_search=True:
        - в пайплайн добавляется шаг pca;
        - GridSearchCV будет сравнивать PCA и вариант без PCA.

    Если pca_search=False:
        - пайплайн без PCA;
        - сетка параметров остаётся обычной.
    """
    if pca_search:
        pipeline = build_iris_pipeline(
            model,
            use_pca=True,
            pca_components=None,
        )
        grid = with_pca_options(param_grid)
    else:
        pipeline = build_iris_pipeline(
            model,
            use_pca=False,
        )
        grid = param_grid

    return pipeline, grid


def run(
    test_size: float = 0.2,
    random_state: int = 42,
    pca_search: bool = True,
) -> None:
    set_style()

    df = load_raw()

    if "target" not in df.columns:
        raise ValueError("В датасете нет колонки target. Проверь load_raw().")

    train_df, holdout_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["target"],
        random_state=random_state,
    )

    X_train = prepare_features(train_df)
    X_holdout = prepare_features(holdout_df)

    y_train = train_df["target"]
    y_holdout = holdout_df["target"]

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    # ================= BASELINE =================
    baseline_model = build_iris_pipeline(
        LogisticRegression(max_iter=1000, random_state=random_state),
        use_pca=False,
    )

    baseline_model.fit(X_train, y_train)

    baseline_pred = baseline_model.predict(X_holdout)
    baseline_prob = baseline_model.predict_proba(X_holdout)

    baseline_metrics = classification_metrics_multiclass(
        y_holdout,
        baseline_pred,
        baseline_prob,
    )

    print_metrics(
        "BASELINE: LogisticRegression без PCA",
        baseline_metrics,
    )

    # ================= FINAL: LogisticRegression + CV =================
    lr_pipeline, lr_grid = make_pipeline_and_grid(
        LogisticRegression(max_iter=1000, random_state=random_state),
        {"model__C": C_GRID},
        pca_search=pca_search,
    )

    lr_search = GridSearchCV(
        lr_pipeline,
        lr_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
    )

    lr_search.fit(X_train, y_train)

    print(
        f"\nЛучшие параметры LogisticRegression по CV "
        f"(accuracy={lr_search.best_score_:.4f}): "
        f"{_json_safe_params(lr_search.best_params_)}"
    )

    final_model = lr_search.best_estimator_

    final_pred = final_model.predict(X_holdout)
    final_prob = final_model.predict_proba(X_holdout)

    final_metrics = classification_metrics_multiclass(
        y_holdout,
        final_pred,
        final_prob,
    )

    print_metrics(
        "FINAL: LogisticRegression + CV"
        + (" + PCA search" if pca_search else ""),
        final_metrics,
    )

    print("\n=== BASELINE vs FINAL (hold-out test) ===")
    for key, final_value in final_metrics.items():
        base_value = baseline_metrics.get(key)

        if base_value is None or final_value is None:
            continue

        print(
            f"{key:>15}: {base_value:.4f} -> {final_value:.4f}  "
            f"(delta {final_value - base_value:+.4f})"
        )

    # ================= LogisticRegression без Ridge + PCA search =================
    lr_no_ridge_pipeline, lr_no_ridge_grid = make_pipeline_and_grid(
        LogisticRegression(
            penalty=None,
            max_iter=1000,
            random_state=random_state,
        ),
        {},
        pca_search=pca_search,
    )

    lr_no_ridge_search = GridSearchCV(
        lr_no_ridge_pipeline,
        lr_no_ridge_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
    )

    lr_no_ridge_search.fit(X_train, y_train)

    print(
        f"\nЛучшие параметры LogisticRegression без Ridge"
        f" (CV accuracy={lr_no_ridge_search.best_score_:.4f}): "
        f"{_json_safe_params(lr_no_ridge_search.best_params_)}"
    )

    lr_no_ridge_model = lr_no_ridge_search.best_estimator_

    lr_no_ridge_pred = lr_no_ridge_model.predict(X_holdout)
    lr_no_ridge_prob = lr_no_ridge_model.predict_proba(X_holdout)

    lr_no_ridge_metrics = classification_metrics_multiclass(
        y_holdout,
        lr_no_ridge_pred,
        lr_no_ridge_prob,
    )

    print_metrics(
        "LogisticRegression без Ridge"
        + (" + PCA search" if pca_search else ""),
        lr_no_ridge_metrics,
    )
    # ================= SVM Linear =================
    svm_lin_pipeline, svm_lin_grid = make_pipeline_and_grid(
        SVC(kernel="linear", probability=True, random_state=random_state),
        {"model__C": C_GRID},
        pca_search=pca_search,
    )

    svm_lin_search = GridSearchCV(
        svm_lin_pipeline,
        svm_lin_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
    )

    svm_lin_search.fit(X_train, y_train)

    svm_lin_model = svm_lin_search.best_estimator_
    svm_lin_pred = svm_lin_model.predict(X_holdout)
    svm_lin_prob = svm_lin_model.predict_proba(X_holdout)

    svm_lin_metrics = classification_metrics_multiclass(
        y_holdout,
        svm_lin_pred,
        svm_lin_prob,
    )

    print_metrics(
        f"SVM Linear (best params={_json_safe_params(svm_lin_search.best_params_)})",
        svm_lin_metrics,
    )

    # ================= SVM RBF =================
    svm_rbf_pipeline, svm_rbf_grid = make_pipeline_and_grid(
        SVC(kernel="rbf", probability=True, random_state=random_state),
        {
            "model__C": C_GRID,
            "model__gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1.0],
        },
        pca_search=pca_search,
    )

    svm_rbf_search = GridSearchCV(
        svm_rbf_pipeline,
        svm_rbf_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
    )

    svm_rbf_search.fit(X_train, y_train)

    svm_rbf_model = svm_rbf_search.best_estimator_
    svm_rbf_pred = svm_rbf_model.predict(X_holdout)
    svm_rbf_prob = svm_rbf_model.predict_proba(X_holdout)

    svm_rbf_metrics = classification_metrics_multiclass(
        y_holdout,
        svm_rbf_pred,
        svm_rbf_prob,
    )

    print_metrics(
        f"SVM RBF (best params={_json_safe_params(svm_rbf_search.best_params_)})",
        svm_rbf_metrics,
    )

    # ================= Decision Tree =================
    dt_param_grid = {
        "model__criterion": ["gini", "entropy"],
        "model__max_depth": [2, 3, 4, 5, 6, 7, 8, None],
        "model__min_samples_split": [2, 5, 10, 20],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__class_weight": [None, "balanced"],
    }

    dt_pipeline, dt_grid = make_pipeline_and_grid(
        DecisionTreeClassifier(random_state=random_state),
        dt_param_grid,
        pca_search=pca_search,
    )

    dt_search = GridSearchCV(
        dt_pipeline,
        dt_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
    )

    dt_search.fit(X_train, y_train)

    dt_model = dt_search.best_estimator_
    dt_pred = dt_model.predict(X_holdout)
    dt_prob = dt_model.predict_proba(X_holdout)

    dt_metrics = classification_metrics_multiclass(
        y_holdout,
        dt_pred,
        dt_prob,
    )

    print_metrics(
        f"Decision Tree (best params={_json_safe_params(dt_search.best_params_)})",
        dt_metrics,
    )

    # ================= Сравнение всех моделей =================
    print("\n=== СРАВНЕНИЕ ВСЕХ МОДЕЛЕЙ (hold-out test) ===")

    models_comp = {
        "Baseline (LR)": baseline_metrics,
        "Final (LR)": final_metrics,
        "SVM (Linear)": svm_lin_metrics,
        "SVM (RBF)": svm_rbf_metrics,
        "Decision Tree": dt_metrics,
    }

    for m_name, m_metrics in models_comp.items():
        roc_auc_value = m_metrics.get("roc_auc_ovr")
        roc_auc_text = f"{roc_auc_value:.4f}" if roc_auc_value is not None else "N/A"

        print(
            f"{m_name:<18} | "
            f"Accuracy: {m_metrics.get('accuracy', 0):.4f} | "
            f"F1 macro: {m_metrics.get('f1_macro', 0):.4f} | "
            f"ROC-AUC OVR: {roc_auc_text}"
        )

    # ================= сохранение артефактов =================
    (MODELS_DIR / "plots").mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(final_model, MODELS_DIR / "model.joblib")
    joblib.dump(svm_rbf_model, MODELS_DIR / "svm_rbf.joblib")
    joblib.dump(dt_model, MODELS_DIR / "decision_tree.joblib")

    save_json(
        {
            "baseline": baseline_metrics,
            "final_lr": final_metrics,
            "svm_linear": svm_lin_metrics,
            "svm_rbf": svm_rbf_metrics,
            "decision_tree": dt_metrics,
            "best_params_lr": _json_safe_params(lr_search.best_params_),
            "best_params_svm_linear": _json_safe_params(svm_lin_search.best_params_),
            "best_params_svm_rbf": _json_safe_params(svm_rbf_search.best_params_),
            "best_params_decision_tree": _json_safe_params(dt_search.best_params_),
            "cv_accuracy_lr": float(lr_search.best_score_),
            "cv_accuracy_svm_linear": float(svm_lin_search.best_score_),
            "cv_accuracy_svm_rbf": float(svm_rbf_search.best_score_),
            "cv_accuracy_decision_tree": float(dt_search.best_score_),
            "pca_search": pca_search,
            "target_names": TARGET_NAMES,
            "test_size": test_size,
            "random_state": random_state,
        },
        MODELS_DIR / "metrics.json",
    )

    # Confusion matrix для финальной модели
    y_holdout_display = (
        holdout_df["species"].to_numpy()
        if "species" in holdout_df.columns
        else np.asarray(TARGET_NAMES)[y_holdout.to_numpy()]
    )

    final_pred_display = np.asarray(TARGET_NAMES)[final_pred]

    plot_multiclass_confusion_matrix(
        y_holdout_display,
        final_pred_display,
        labels=TARGET_NAMES,
        title="Confusion matrix: финальная модель Iris",
    )

    plt.savefig(
        MODELS_DIR / "plots" / "confusion_matrix.png",
        dpi=120,
        bbox_inches="tight",
    )
    plt.close()

    # Многоклассовая ROC-кривая для финальной модели
    plot_multiclass_roc(
        y_holdout.to_numpy(),
        final_prob,
        target_names=TARGET_NAMES,
    )

    plt.savefig(
        MODELS_DIR / "plots" / "roc_curve_ovr.png",
        dpi=120,
        bbox_inches="tight",
    )
    plt.close()

    # Коэффициенты логистической регрессии можно нормально рисовать,
    # только если PCA выключен. Если PCA включён, коэффициенты живут
    # в пространстве главных компонент, а не исходных признаков.
    pca_step = final_model.named_steps.get("pca", "passthrough")

    if isinstance(pca_step, str) and pca_step == "passthrough":
        try:
            feature_names = final_model.named_steps["preprocess"].get_feature_names_out()
            feature_names = [name.replace("num__", "") for name in feature_names]

            coef = final_model.named_steps["model"].coef_

            if coef.ndim == 1:
                plot_coefficients(
                    feature_names,
                    coef,
                    title="Коэффициенты LogisticRegression (Iris)",
                )

                plt.savefig(
                    MODELS_DIR / "plots" / "coefficients.png",
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
                            MODELS_DIR / "plots" / f"coefficients_{class_name}.png",
                            dpi=120,
                            bbox_inches="tight",
                        )
                        plt.close()

        except Exception as e:
            print(f"Не удалось построить график коэффициентов: {e}")
    else:
        print(
            "Финальная LogisticRegression использует PCA, "
            "поэтому коэффициенты находятся в пространстве главных компонент."
        )

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
        PROCESSED_DIR / "train_processed.csv",
        index=False,
    )

    holdout_processed.to_csv(
        PROCESSED_DIR / "holdout_processed.csv",
        index=False,
    )

    # Небольшой artifact с предсказаниями на hold-out
    prediction_df = pd.DataFrame(
        {
            "y_true": y_holdout.to_numpy(),
            "y_pred": final_pred,
            "species_true": np.asarray(TARGET_NAMES)[y_holdout.to_numpy()],
            "species_pred": np.asarray(TARGET_NAMES)[final_pred],
        }
    )

    prediction_df.to_csv(
        MODELS_DIR / "holdout_predictions.csv",
        index=False,
    )

    print(f"\nАртефакты сохранены в {MODELS_DIR}/ и {PROCESSED_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--no-pca-search",
        dest="pca_search",
        action="store_false",
        help="Не искать вариант с PCA. Используется обычный пайплайн без PCA.",
    )

    parser.set_defaults(pca_search=True)

    args = parser.parse_args()

    run(
        test_size=args.test_size,
        random_state=args.random_state,
        pca_search=args.pca_search,
    )