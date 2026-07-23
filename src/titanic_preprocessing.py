"""Загрузка, очистка и инженерия признаков для задачи Titanic (логистическая регрессия)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAW_DIR = Path("data/titanic/raw")

# Частые титулы объединяем/переименовываем в 5 групп: Mr/Mrs/Miss/Master/Rare.
TITLE_MAP = {
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs",
    "Lady": "Rare",
    "Countess": "Rare",
    "Capt": "Rare",
    "Col": "Rare",
    "Don": "Rare",
    "Dr": "Rare",
    "Major": "Rare",
    "Rev": "Rare",
    "Sir": "Rare",
    "Jonkheer": "Rare",
    "Dona": "Rare",
}

BASELINE_NUMERIC = ["Pclass", "Age", "Fare", "SibSp", "Parch"]
BASELINE_CATEGORICAL = ["Sex"]

FINAL_NUMERIC = ["Pclass", "Age", "Fare", "FamilySize", "IsAlone"]
FINAL_CATEGORICAL = ["Sex", "Embarked", "Title", "Deck"]


def load_raw(raw_dir: Path | str = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_dir = Path(raw_dir)
    train = pd.read_csv(raw_dir / "train.csv")
    test = pd.read_csv(raw_dir / "test.csv")
    return train, test


def extract_title(df: pd.DataFrame) -> pd.Series:
    title = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    return title.replace(TITLE_MAP)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет Title, FamilySize, IsAlone, Deck поверх сырых колонок Titanic."""
    df = df.copy()
    df["Title"] = extract_title(df)
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["Deck"] = df["Cabin"].astype(str).str[0].where(df["Cabin"].notna(), "Unknown")
    df["Fare"] = np.log1p(df["Fare"])  # Fare сильно скошен вправо
    return df


def impute_age_by_title(train: pd.DataFrame, *frames: pd.DataFrame) -> list[pd.DataFrame]:
    """Заполняет пропуски в Age медианой возраста внутри той же группы Title (посчитанной по train)."""
    age_by_title = train.groupby("Title")["Age"].median()
    global_median = train["Age"].median()

    out = []
    for df in (train, *frames):
        df = df.copy()
        fill_values = df["Title"].map(age_by_title).fillna(global_median)
        df["Age"] = df["Age"].fillna(fill_values)
        out.append(df)
    return out


def build_baseline_pipeline(model) -> Pipeline:
    """Минимальная предобработка: только импутация пропусков и OHE пола, без инженерии признаков."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), BASELINE_NUMERIC),
            (
                "cat",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore", drop="if_binary")),
                ]),
                BASELINE_CATEGORICAL,
            ),
        ]
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def build_final_pipeline(model) -> Pipeline:
    """Полная предобработка: инженерные признаки + масштабирование + OHE."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]),
                FINAL_NUMERIC,
            ),
            (
                "cat",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore")),
                ]),
                FINAL_CATEGORICAL,
            ),
        ]
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])
