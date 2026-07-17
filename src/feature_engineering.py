
import pandas as pd
import numpy as np
from scipy.stats import norm, skew
from scipy.stats import boxcox
from scipy import stats

def group_rare_neighborhoods(train, test, threshold):
    """
    Объединяет районы с числом домов < threshold в категорию 'Other'.
    """
    # Подсчитываем частоту районов в train
    freq = train['Neighborhood'].value_counts()
    rare = freq[freq < threshold].index
    # Заменяем редкие районы на 'Other' в train
    train['Neighborhood'] = train['Neighborhood'].replace(rare, 'Other')
    # В test заменяем те же редкие районы, а также районы, отсутствующие в train (если есть)
    test['Neighborhood'] = test['Neighborhood'].apply(lambda x: 'Other' if x in rare else x)
    return train, test

from itertools import combinations
from sklearn.feature_selection import mutual_info_regression
import pandas as pd
import numpy as np



def transform_skewed_features(data, test, threshold=0.75, lambda_val=0.15, auto_lambda=False):
    """
    Преобразование асимметричных признаков с помощью np.log1p.
    Простая и надёжная версия.
    """
    import numpy as np
    from scipy.stats import skew
    
    data = data.copy()
    test = test.copy()
    
    num_cols = data.select_dtypes(include=[np.number]).columns
    
    for col in num_cols:
        if data[col].nunique() <= 1:
            continue
            
        skew_val = skew(data[col].dropna())
        
        if abs(skew_val) > threshold:
            min_val = data[col].min()
            if min_val <= 0:
                shift = abs(min_val) + 1
                data[col] = np.log1p(data[col] + shift)
                test[col] = np.log1p(test[col] + shift)
            else:
                data[col] = np.log1p(data[col])
                test[col] = np.log1p(test[col])
            
            print(f"Transformed {col}: skew {skew_val:.2f} -> {skew(data[col]):.2f}")
    
    return data, test

def remove_outliers_iqr(data, threshold=1.5, max_outlier_pct=5.0):
    """
    Удаляет выбросы по правилу IQR только для колонок, 
    где процент выбросов не превышает max_outlier_pct.
    """
    data_clean = data.copy()
    num_cols = data_clean.select_dtypes(include=[np.number]).columns
    
    for col in num_cols:
        Q1 = data_clean[col].quantile(0.25)
        Q3 = data_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR == 0:
            continue
            
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        
        outliers_mask = (data_clean[col] < lower) | (data_clean[col] > upper)
        outlier_pct = outliers_mask.sum() / len(data_clean) * 100
        
        if outlier_pct <= max_outlier_pct:
            data_clean = data_clean[~outliers_mask]
    
    removed = len(data) - len(data_clean)
    print(f"Удалено строк: {removed} ({removed/len(data)*100:.2f}%)")
    return data_clean


def fill_missing_values(data,test):

    cat_cols = data.select_dtypes(include=['object']).columns
    num_cols = data.select_dtypes(exclude=['object']).columns

    for col in cat_cols:
        data[col] = data[col].fillna('None')
        test[col] = test[col].fillna('None')

    for col in num_cols:
        data[col] = data[col].fillna(0)
        test[col] = test[col].fillna(0)
    return data,test

def analyze_outliers_iqr(data):
    
    results = []
    num_cols = data.select_dtypes(include=[np.number]).columns
    
    for col in num_cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Если IQR = 0, пропускаем (все значения одинаковые)
        if IQR == 0:
            continue
            
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        

        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
        
        if len(outliers) > 0:
            results.append({
                'Column': col,
                'Outliers_Count': len(outliers),
                'Outliers_Pct': round(len(outliers) / len(data) * 100, 2),
                'Lower_Bound': round(lower_bound, 2),
                'Upper_Bound': round(upper_bound, 2),
                'Min_Outlier': round(outliers[col].min(), 2),
                'Max_Outlier': round(outliers[col].max(), 2)
            })
    
    df_outliers = pd.DataFrame(results).sort_values('Outliers_Count', ascending=False)
    return df_outliers


def get_saleprice_bounds(df, price_col='SalePrice', threshold=1.5, lower_percentile=0.01, upper_percentile=0.99):
    
    data = df[price_col].dropna()
    
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - threshold * IQR
    upper = Q3 + threshold * IQR

    anomalies = data[(data < lower) | (data > upper)]
    
    result = {
        'lower_bound': lower,
        'upper_bound': upper,
        'n_anomalies': len(anomalies),
        'percent_anomalies': len(anomalies) / len(data) * 100,
    }
    
    print(f"Нижняя граница: {lower:,.0f}")
    print(f"Верхняя граница: {upper:,.0f}")
    print(f"Аномалий: {result['n_anomalies']} ({result['percent_anomalies']:.2f}%)")
    
    return result

def clip_df(df, columns=None, lower_q=0.01, upper_q=0.99):
    df = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include='number').columns
    
    for col in columns:
        low = df[col].quantile(lower_q)
        high = df[col].quantile(upper_q)
        df[col] = df[col].clip(low, high)
    
    return df

def feature_saleprice(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Создает новые признаки для датасета House Prices.
    
    Созданные признаки:
    - QualArea: Взаимодействие качества и площади (OverallQual * GrLivArea)
    - TotalBath: Общее количество ванных комнат (с учетом половинных)
    - TotalPorchSF: Общая площадь крыльца/веранды
    - HouseAge: Возраст дома
    - RemodAge: Возраст после последнего ремонта
    - IsRemodeled: Был ли дом отремонтирован (бинарный признак)
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Тренировочные данные
    X_test : pd.DataFrame
        Тестовые данные
        
    Returns:
    --------
    tuple[pd.DataFrame, pd.DataFrame]
        Кортеж из (X_train с новыми признаками, X_test с новыми признаками)
    """
    # Создаем копии, чтобы не изменять оригинальные данные
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    # 1. Взаимодействие качества и площади
    X_train['QualArea'] = X_train['OverallQual'] * X_train['GrLivArea']
    X_test['QualArea'] = X_test['OverallQual'] * X_test['GrLivArea']
    
    # 2. Общее количество ванных комнат
    for df in [X_train, X_test]:
        df['TotalBath'] = (df['FullBath'] + 0.5 * df['HalfBath'] + 
                           df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath'])
    
    # 3. Общая площадь крыльца/веранды
    for df in [X_train, X_test]:
        df['TotalPorchSF'] = (df['OpenPorchSF'] + df['EnclosedPorch'] + 
                              df['3SsnPorch'] + df['ScreenPorch'])
    
    # 4. Возраст дома и ремонта
    for df in [X_train, X_test]:
        df['HouseAge'] = df['YrSold'] - df['YearBuilt']
        df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
        df['IsRemodeled'] = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)
    
    return X_train, X_test

