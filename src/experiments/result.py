"""Именованный результат обучения и оценки на holdout."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


@dataclass
class TrainingResult:
    """Предсказания в исходной шкале target; search отсутствует у baseline.

    probabilities: P(y=1) для Titanic, матрица P(y=k) для Iris,
    None для регрессии. model — обученный Pipeline.
    """

    model: Pipeline
    metrics: dict[str, float | None]
    predictions: np.ndarray
    probabilities: np.ndarray | None = None
    search: GridSearchCV | None = None
