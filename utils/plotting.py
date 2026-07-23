"""Переиспользуемые функции визуализации для EDA-ноутбуков и диагностики моделей.

Единый стиль для обеих задач (Titanic, House Prices): фиксированная
категориальная палитра, одна последовательная шкала для величин, одна
дивергирующая шкала (с нейтральной серединой) для корреляций/остатков.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Фиксированный порядок категориальных цветов (не переставлять — порядок
# подобран так, чтобы соседние цвета были различимы при дальтонизме).
CATEGORICAL_PALETTE = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#e34948",  # red
    "#4a3aa7",  # violet
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
]

SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "sequential_blue", ["#cde2fb", "#2a78d6", "#0d366b"]
)
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "diverging_blue_red", ["#184f95", "#f0efec", "#b03030"]
)

GRID_COLOR = "#d8d7d0"
TEXT_SECONDARY = "#52514e"


def set_style() -> None:
    """Единый стиль для всех графиков проекта. Вызывается один раз в начале ноутбука/скрипта."""
    sns.set_theme(style="white", palette=CATEGORICAL_PALETTE)
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": TEXT_SECONDARY,
        "axes.grid": True,
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
    })


def plot_missing_values(df: pd.DataFrame, top_n: int = 20, ax=None):
    """Горизонтальный бар-чарт доли пропусков по колонкам (только те, где есть пропуски)."""
    missing = df.isna().mean().sort_values(ascending=False)
    missing = missing[missing > 0].head(top_n)
    if missing.empty:
        print("Пропусков нет.")
        return None

    fig, ax = (None, ax) if ax is not None else plt.subplots(figsize=(7, max(3, 0.35 * len(missing))))
    ax.barh(missing.index[::-1], missing.values[::-1] * 100, color=CATEGORICAL_PALETTE[0])
    ax.set_xlabel("Доля пропусков, %")
    ax.set_title("Пропущенные значения по признакам")
    return ax


def plot_target_distribution(y: pd.Series, log: bool = False, title: str | None = None, ax=None):
    """Гистограмма распределения таргета (с опциональным log1p) + отметка среднего/медианы."""
    values = np.log1p(y) if log else y
    fig, ax = (None, ax) if ax is not None else plt.subplots(figsize=(7, 4.5))
    sns.histplot(values, kde=True, ax=ax, color=CATEGORICAL_PALETTE[0])
    ax.axvline(values.mean(), color=CATEGORICAL_PALETTE[3], linestyle="--", label=f"mean={values.mean():,.2f}")
    ax.axvline(values.median(), color=CATEGORICAL_PALETTE[1], linestyle=":", label=f"median={values.median():,.2f}")
    ax.set_title(title or ("log1p(target)" if log else "Распределение таргета"))
    ax.legend()
    return ax


def plot_correlation_heatmap(df: pd.DataFrame, target: str | None = None, top_n: int = 15, ax=None):
    """Тепловая карта корреляций. Если задан target — только top_n признаков, сильнее всего с ним связанных."""
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()

    if target is not None and target in corr.columns:
        top_features = corr[target].abs().sort_values(ascending=False).head(top_n + 1).index
        corr = corr.loc[top_features, top_features]

    fig, ax = (None, ax) if ax is not None else plt.subplots(figsize=(0.55 * len(corr) + 2, 0.55 * len(corr) + 1))
    sns.heatmap(
        corr, ax=ax, cmap=DIVERGING_CMAP, center=0, vmin=-1, vmax=1,
        annot=len(corr) <= 16, fmt=".2f", square=True,
        linewidths=0.5, linecolor="white", cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Матрица корреляций")
    return ax


def plot_categorical_vs_target(df: pd.DataFrame, cat_col: str, target_col: str, agg: str = "mean", ax=None):
    """Столбчатый график target по категориям (agg='mean' — доля выживших / средняя цена и т.п.)."""
    summary = df.groupby(cat_col)[target_col].agg(agg).sort_values(ascending=False)
    counts = df[cat_col].value_counts()

    fig, ax = (None, ax) if ax is not None else plt.subplots(figsize=(max(6, 0.5 * len(summary)), 4.5))
    ax.bar(summary.index.astype(str), summary.values, color=CATEGORICAL_PALETTE[0])
    ax.set_ylabel(f"{agg}({target_col})")
    ax.set_title(f"{target_col} по категориям {cat_col}")
    ax.tick_params(axis="x", rotation=45)
    for i, cat in enumerate(summary.index):
        ax.text(i, summary.values[i], f"n={counts[cat]}", ha="center", va="bottom",
                fontsize=8, color=TEXT_SECONDARY)
    return ax


def plot_numeric_distributions(df: pd.DataFrame, cols: list[str], ncols: int = 3):
    """Сетка гистограмм для набора числовых признаков (быстрый обзор распределений)."""
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for i, col in enumerate(cols):
        sns.histplot(df[col].dropna(), ax=axes[i], color=CATEGORICAL_PALETTE[0], kde=True)
        axes[i].set_title(col, fontsize=10)
        axes[i].set_xlabel("")
    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_boxplot_by_category(df: pd.DataFrame, cat_col: str, num_col: str, order: list[str] | None = None, ax=None):
    fig, ax = (None, ax) if ax is not None else plt.subplots(figsize=(max(6, 0.5 * df[cat_col].nunique()), 4.5))
    sns.boxplot(data=df, x=cat_col, y=num_col, order=order, ax=ax, color=CATEGORICAL_PALETTE[0])
    ax.tick_params(axis="x", rotation=45)
    ax.set_title(f"{num_col} по {cat_col}")
    return ax


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Residual plot", ax=None):
    """y_pred на оси X, остатки (y_true - y_pred) на оси Y — диагностика линейности/гетероскедастичности."""
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = (None, ax) if ax is not None else plt.subplots(figsize=(6.5, 5))
    ax.scatter(y_pred, residuals, alpha=0.4, color=CATEGORICAL_PALETTE[0], edgecolor="none", s=18)
    ax.axhline(0, color=CATEGORICAL_PALETTE[3], linestyle="--")
    ax.set_xlabel("Предсказание")
    ax.set_ylabel("Остаток (y - ŷ)")
    ax.set_title(title)
    return ax


def plot_predicted_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Predicted vs Actual", ax=None):
    fig, ax = (None, ax) if ax is not None else plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, color=CATEGORICAL_PALETTE[0], edgecolor="none", s=18)
    lims = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
    ax.plot(lims, lims, color=CATEGORICAL_PALETTE[3], linestyle="--", label="идеальное предсказание")
    ax.set_xlabel("Реальное значение")
    ax.set_ylabel("Предсказание")
    ax.set_title(title)
    ax.legend()
    return ax


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: tuple[str, str] = ("0", "1"), ax=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = (None, ax) if ax is not None else plt.subplots(figsize=(4.5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=SEQUENTIAL_CMAP, cbar=False, ax=ax,
        xticklabels=labels, yticklabels=labels, linewidths=0.5, linecolor="white",
    )
    ax.set_xlabel("Предсказано")
    ax.set_ylabel("Реально")
    ax.set_title("Confusion matrix")
    return ax


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, ax=None):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = (None, ax) if ax is not None else plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, color=CATEGORICAL_PALETTE[0], label=f"ROC-AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color=TEXT_SECONDARY, linestyle="--", label="случайное угадывание")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC-кривая")
    ax.legend()
    return ax


def plot_coefficients(feature_names: list[str], coefficients: np.ndarray, top_n: int = 15,
                       title: str = "Веса модели", ax=None):
    """Топ-N признаков по модулю коэффициента (интерпретация линейной/логистической регрессии)."""
    order = np.argsort(np.abs(coefficients))[::-1][:top_n]
    names = np.array(feature_names)[order]
    values = np.array(coefficients)[order]
    colors = [CATEGORICAL_PALETTE[0] if v >= 0 else CATEGORICAL_PALETTE[3] for v in values]

    fig, ax = (None, ax) if ax is not None else plt.subplots(figsize=(7, max(3, 0.35 * len(names))))
    ax.barh(names[::-1], values[::-1], color=colors[::-1])
    ax.axvline(0, color=TEXT_SECONDARY, linewidth=0.8)
    ax.set_title(title)
    return ax
