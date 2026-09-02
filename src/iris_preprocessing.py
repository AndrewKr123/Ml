"""Загрузка и предобработка датасета Iris."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

RAW_DIR = Path("data/iris/raw")

IRIS_NUMERIC = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

SKLEARN_COLUMN_MAP = {
    "sepal length (cm)": "sepal_length",
    "sepal width (cm)": "sepal_width",
    "petal length (cm)": "petal_length",
    "petal width (cm)": "petal_width",
    "target": "target",
}

KAGGLE_COLUMN_MAP = {
    "Id": "id",
    "id": "id",
    "SepalLengthCm": "sepal_length",
    "SepalWidthCm": "sepal_width",
    "PetalLengthCm": "petal_length",
    "PetalWidthCm": "petal_width",
    "Species": "species",
    "species": "species",
}

SPECIES_TO_TARGET = {
    "setosa": 0,
    "versicolor": 1,
    "virginica": 2,
}

NON_FEATURE_COLUMNS = ["id", "species", "target"]


def _standardize_iris(df: pd.DataFrame) -> pd.DataFrame:
    """
    Приводит датасет Iris к единому формату:

    - sepal_length
    - sepal_width
    - petal_length
    - petal_width
    - target
    - species
    """
    df = df.copy()

    rename_map = {}

    for col in df.columns:
        col_name = str(col).strip()

        if col_name in SKLEARN_COLUMN_MAP:
            rename_map[col] = SKLEARN_COLUMN_MAP[col_name]
        elif col_name in KAGGLE_COLUMN_MAP:
            rename_map[col] = KAGGLE_COLUMN_MAP[col_name]

    df = df.rename(columns=rename_map)

    # Приводим названия колонок к нижнему регистру
    df.columns = [str(col).lower() for col in df.columns]

    # Нормализуем строковый признак species
    if "species" in df.columns:
        df["species"] = (
            df["species"]
            .astype(str)
            .str.lower()
            .str.replace("iris-", "", regex=False)
            .str.strip()
        )

    # Если target нет, но есть species, создаём target
    if "target" not in df.columns and "species" in df.columns:
        df["target"] = df["species"].map(SPECIES_TO_TARGET)

    # Если есть target, но нет species, создаём species
    if "target" in df.columns and "species" not in df.columns:
        target_to_species = {v: k for k, v in SPECIES_TO_TARGET.items()}
        df["species"] = df["target"].map(target_to_species)

    return df


def load_raw(raw_dir: Path | str | None = RAW_DIR) -> pd.DataFrame:
    """
    Загружает Iris.

    Сначала пытается найти CSV в raw_dir.
    Если файла нет, загружает датасет из sklearn.
    """
    if raw_dir is not None:
        raw_dir = Path(raw_dir)

        candidates = [
            raw_dir / "Iris.csv",
            raw_dir / "iris.csv",
            raw_dir / "train.csv",
        ]

        for path in candidates:
            if path.exists():
                df = pd.read_csv(path)
                return _standardize_iris(df)

    # Если CSV не найден, загружаем из sklearn
    iris = load_iris(as_frame=True)
    return _standardize_iris(iris.frame)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Удаляет служебные колонки: id, species, target.
    Оставляет только признаки для модели.
    """
    drop_cols = [col for col in NON_FEATURE_COLUMNS if col in df.columns]
    return df.drop(columns=drop_cols)


def build_iris_pipeline(
    model,
) -> Pipeline:
    """
    Строит пайплайн предобработки для Iris.

    Параметры:
    model:
        sklearn-совместимый классификатор.
    """
    numeric_transformer = Pipeline([
        ("scale", RobustScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, IRIS_NUMERIC),
        ],
        remainder="drop",
    )

    steps = [("preprocess", preprocessor)]

    steps.append(("model", model))

    return Pipeline(steps)


def build_final_pipeline(
    model,
) -> Pipeline:
    """Финальный пайплайн для Iris."""
    return build_iris_pipeline(
        model=model,
    )