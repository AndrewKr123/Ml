"""Загрузка, очистка и инженерия признаков для задачи House Prices (линейная регрессия)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler, TargetEncoder

RAW_DIR = Path("data/house_prices/raw")
TARGET = "SalePrice"

# Признаки-дубли и признаки с >50% пропусков — убираем, чтобы не плодить
# идеальную мультиколлинеарность (see notebooks/02_house_prices_eda.ipynb).
DROP_COLS = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "MasVnrType",  # >50% пропусков
    "TotRmsAbvGrd",   # дублирует GrLivArea
    "GarageArea",     # дублирует GarageCars
    "BsmtFinSF2",     # почти всегда 0, шум
    "TotalBsmtSF",    # = BsmtFinSF1 + BsmtFinSF2 + BsmtUnfSF
    "GrLivArea",      # = 1stFlrSF + 2ndFlrSF + LowQualFinSF
    "LowQualFinSF",   # почти всегда 0, шум
    "GarageYrBlt",    # дублирует YearBuilt
]

# После агрегации в TotalBath/TotalPorchSF исходные компоненты тоже удаляем,
# иначе сумма и слагаемые снова дают идеальную мультиколлинеарность.
AGGREGATED_COMPONENT_COLS = [
    "FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath",
    "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch",
]

# Категориальные колонки, где NaN означает «этого элемента у дома нет».
CATEGORICAL_NONE_COLS = [
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond",
]

# Числовые колонки, где NaN означает «0 (этого элемента у дома нет)».
NUMERIC_ZERO_COLS = ["MasVnrArea", "BsmtFinSF1", "BsmtUnfSF", "GarageCars"]

# Порядковые шкалы качества/состояния: от худшего к лучшему.
ORDINAL_MAPS: dict[str, list[str]] = {
    "ExterQual": ["Po", "Fa", "TA", "Gd", "Ex"],
    "ExterCond": ["Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtQual": ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtCond": ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
    "HeatingQC": ["Po", "Fa", "TA", "Gd", "Ex"],
    "KitchenQual": ["Po", "Fa", "TA", "Gd", "Ex"],
    "FireplaceQu": ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
    "GarageQual": ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
    "GarageCond": ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtExposure": ["NA", "No", "Mn", "Av", "Gd"],
    "BsmtFinType1": ["NA", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmtFinType2": ["NA", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "GarageFinish": ["NA", "Unf", "RFn", "Fin"],
    "Functional": ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
}

# Номинальные категории без естественного порядка -> One-Hot.
ONEHOT_COLS = [
    "MSZoning", "Street", "LotShape", "LandContour", "Utilities", "LotConfig",
    "LandSlope", "Condition1", "Condition2", "BldgType", "HouseStyle",
    "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "Foundation",
    "Heating", "CentralAir", "Electrical", "GarageType", "PavedDrive",
    "SaleType", "SaleCondition",
]

# Высококардинальный признак -> target encoding (честный, с cross-fitting внутри sklearn.TargetEncoder).
TARGET_ENCODE_COLS = ["Neighborhood"]

BASELINE_DROP_ALWAYS = ["Id", TARGET]


def load_raw(raw_dir: Path | str = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_dir = Path(raw_dir)
    train = pd.read_csv(raw_dir / "train.csv")
    test = pd.read_csv(raw_dir / "test.csv")
    return train, test


def get_baseline_numeric_columns(train: pd.DataFrame) -> list[str]:
    """Все числовые колонки как есть — то, что взял бы новичок без раздумий."""
    numeric = train.select_dtypes(include="number").columns.tolist()
    return [c for c in numeric if c not in BASELINE_DROP_ALWAYS]


def engineer_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Создаёт новые признаки, затем удаляет дубли/шум и агрегированные компоненты."""
    df = df.copy()

    df["QualArea"] = df["OverallQual"] * df["GrLivArea"]
    df["TotalBath"] = (
        df["FullBath"] + 0.5 * df["HalfBath"]
        + df["BsmtFullBath"].fillna(0) + 0.5 * df["BsmtHalfBath"].fillna(0)
    )
    df["TotalPorchSF"] = (
        df["OpenPorchSF"] + df["EnclosedPorch"] + df["3SsnPorch"] + df["ScreenPorch"]
    )
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
    df["IsRemodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)

    cols_to_drop = DROP_COLS + AGGREGATED_COMPONENT_COLS
    df = df.drop(columns=cols_to_drop, errors="ignore")
    return df


def fill_missing(train: pd.DataFrame, *frames: pd.DataFrame) -> list[pd.DataFrame]:
    """Заполняет пропуски; все статистики (медианы/моды) считаются только по train."""
    lot_frontage_by_neighborhood = train.groupby("Neighborhood")["LotFrontage"].median()
    lot_frontage_global = train["LotFrontage"].median()

    remaining_num_cols = train.select_dtypes(include="number").columns
    remaining_cat_cols = train.select_dtypes(exclude="number").columns
    numeric_medians = train[remaining_num_cols].median()
    categorical_modes = train[remaining_cat_cols].mode().iloc[0]

    out = []
    for df in (train, *frames):
        df = df.copy()

        df["LotFrontage"] = df["LotFrontage"].fillna(
            df["Neighborhood"].map(lot_frontage_by_neighborhood)
        ).fillna(lot_frontage_global)

        for col in CATEGORICAL_NONE_COLS:
            if col in df.columns:
                df[col] = df[col].fillna("NA")
        for col in NUMERIC_ZERO_COLS:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # Подчищаем редкие "сиротские" пропуски (обычно только в test.csv).
        for col in df.columns.intersection(remaining_num_cols):
            df[col] = df[col].fillna(numeric_medians[col])
        for col in df.columns.intersection(remaining_cat_cols):
            df[col] = df[col].fillna(categorical_modes[col])

        out.append(df)
    return out


def log_transform_skewed(train: pd.DataFrame, *frames: pd.DataFrame, threshold: float = 0.75) -> list[pd.DataFrame]:
    """np.log1p для числовых признаков со скошенностью выше threshold (список признаков фиксируется по train)."""
    numeric_cols = [c for c in get_baseline_numeric_columns(train) if c in train.columns]
    skewed_cols = [
        c for c in numeric_cols
        if train[c].min() >= 0 and abs(skew(train[c].dropna())) > threshold
    ]

    out = []
    for df in (train, *frames):
        df = df.copy()
        for col in skewed_cols:
            df[col] = np.log1p(df[col].clip(lower=0))
        out.append(df)
    return out


def get_final_feature_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    ordinal_cols = [c for c in ORDINAL_MAPS if c in df.columns]
    onehot_cols = [c for c in ONEHOT_COLS if c in df.columns]
    target_cols = [c for c in TARGET_ENCODE_COLS if c in df.columns]
    exclude = set(ordinal_cols) | set(onehot_cols) | set(target_cols) | {"Id", TARGET}
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c not in exclude]
    return {
        "numeric": numeric_cols,
        "ordinal": ordinal_cols,
        "onehot": onehot_cols,
        "target": target_cols,
    }


def build_preprocessor(feature_cols: dict[str, list[str]]) -> ColumnTransformer:
    ordinal_categories = [ORDINAL_MAPS[c] for c in feature_cols["ordinal"]]
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", RobustScaler()),
                ]),
                feature_cols["numeric"],
            ),
            (
                "ord",
                Pipeline([
                    ("impute", SimpleImputer(strategy="constant", fill_value="NA")),
                    (
                        "encode",
                        OrdinalEncoder(
                            categories=ordinal_categories,
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
                        ),
                    ),
                ]),
                feature_cols["ordinal"],
            ),
            (
                "ohe",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("encode", OneHotEncoder(handle_unknown="ignore")),
                ]),
                feature_cols["onehot"],
            ),
            (
                "target",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("encode", TargetEncoder(cv=KFold(n_splits=5, shuffle=True, random_state=42))),
                ]),
                feature_cols["target"],
            ),
        ]
    )
