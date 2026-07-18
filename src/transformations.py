import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.preprocessing import RobustScaler


def encode_features(
    df: pd.DataFrame,
    ohe_cols: List[str] = None,
    target_median_cols: List[str] = None,
    target_mean_cols: List[str] = None,
    target_col: str = 'SalePrice'
) -> pd.DataFrame:
    """
    Кодирует категориальные признаки тремя способами:
    
    1. One-Hot Encoding (OHE) — для признаков с малой кардинальностью
    2. Target Encoding по медиане — для признаков с высокой кардинальностью
    3. Target Encoding по среднему — для порядковых признаков
    
    Parameters:
    -----------
    df : pd.DataFrame
        Исходный датасет
    ohe_cols : list
        Колонки для One-Hot Encoding
    target_median_cols : list
        Колонки для Target Encoding по медиане (высокая кардинальность)
    target_mean_cols : list
        Колонки для Target Encoding по среднему (порядковые)
    target_col : str
        Название целевой переменной (нужно для Target Encoding)
        
    Returns:
    --------
    pd.DataFrame
        Закодированный датасет
    """
    df = df.copy()
    
    # Значения по умолчанию
    if ohe_cols is None:
        ohe_cols = []
    if target_median_cols is None:
        target_median_cols = []
    if target_mean_cols is None:
        target_mean_cols = []
    
    # 1. ONE-HOT ENCODING (низкая кардинальность)
    if ohe_cols:
        # drop_first=True убирает мультиколлинеарность (dummy trap)
        df = pd.get_dummies(
            df, 
            columns=ohe_cols, 
            prefix=ohe_cols, 
            drop_first=True,
            dtype=int  # Чтобы были 0/1, а не True/False
        )
    
    # 2. TARGET ENCODING ПО МЕДИАНЕ (высокая кардинальность)
    for col in target_median_cols:
        if col not in df.columns:
            continue
        
        # Считаем медиану таргета по каждой категории
        median_map = df.groupby(col)[target_col].median()
        
        # Глобальная медиана для неизвестных категорий
        global_median = df[target_col].median()
        
        # Применяем маппинг, неизвестные категории заполняем глобальной медианой
        df[col] = df[col].map(median_map).fillna(global_median)
    
    # 3. TARGET ENCODING ПО СРЕДНЕМУ (порядковые)
    for col in target_mean_cols:
        if col not in df.columns:
            continue
        
        # Считаем среднее таргета по каждой категории
        mean_map = df.groupby(col)[target_col].mean()
        
        # Глобальное среднее для неизвестных категорий
        global_mean = df[target_col].mean()
        
        # Применяем маппинг
        df[col] = df[col].map(mean_map).fillna(global_mean)
    
    return df

class RobustScalerWrapper:
    """
    Обертка над RobustScaler для удобного применения к DataFrame.
    Сохраняет имена колонок и индексы.
    """
    
    def __init__(self):
        self.scaler = RobustScaler()
        self.fitted = False
        self.columns = None
    
    def fit_transform(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """
        Fit на данных и transform.
        На train
        """
        df = df.copy()
        self.columns = cols
        df[cols] = self.scaler.fit_transform(df[cols])
        self.fitted = True
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform без fit.
        Использовать на test
        """
        if not self.fitted:
            raise RuntimeError("Сначала нужно вызвать fit_transform на train данных!")
        
        df = df.copy()
        # Берем только те колонки, на которых обучались
        cols_to_scale = [col for col in self.columns if col in df.columns]
        df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])
        return df


def scale_features(
    df: pd.DataFrame,
    cols: List[str],
    scaler_wrapper: RobustScalerWrapper = None,
    fit: bool = False
) -> Tuple[pd.DataFrame, RobustScalerWrapper]:
    """
    Масштабирует числовые признаки через RobustScaler.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Исходный датасет
    cols : list
        Колонки для масштабирования
    scaler_wrapper : RobustScalerWrapper
        Обертка скалера (для применения к test)
    fit : bool
        Если True — делает fit_transform (для train)
        Если False — делает только transform (для test)
        
    Returns:
    --------
    tuple
        (масштабированный DataFrame, scaler_wrapper)
    """
    if scaler_wrapper is None:
        scaler_wrapper = RobustScalerWrapper()
    
    if fit:
        df = scaler_wrapper.fit_transform(df, cols)
    else:
        df = scaler_wrapper.transform(df)
    
    return df, scaler_wrapper

