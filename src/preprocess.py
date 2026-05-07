import pandas as pd
import numpy as np
from scipy.stats import norm, skew
from scipy.stats import boxcox
from scipy import stats

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


def remove_outliers_iqr(data, threshold=1.5, max_outlier_pct=5.0):
    #Удаляет выбросы по правилу IQR только для колонок, где процент выбросов превышает max_outlier_pct.
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
        
        # Удаляем выбросы, только если их процент не превышает лимит
        if outlier_pct <= max_outlier_pct:
            data_clean = data_clean[~outliers_mask]
    
    removed = len(data) - len(data_clean)
    print(f"Удалено строк: {removed} ({removed/len(data)*100:.2f}%)")
    return data_clean

def transform_skewed_features(data, test, threshold=0.75, lambda_val=0.15, auto_lambda=False):
    """
    Преобразование асимметричных признаков с помощью Box-Cox.
    Работает с копиями данных, избегая ошибок read-only.
    """
    # Создаем копии, чтобы избежать ошибок read-only
    data = data.copy()
    test = test.copy()
    
    numerical_features = data.select_dtypes(include=[np.number]).columns
    
    # Вычисляем асимметрию
    skewness = data[numerical_features].apply(lambda x: stats.skew(x.dropna()))
    skewed_features = skewness[abs(skewness) > threshold]
    positive_skew = skewed_features[skewed_features > threshold].index
    
    # Обработка положительной асимметрии
    for feat in positive_skew:
        if auto_lambda:
            # Подбираем lambda по данным
            vals = data[feat].values
            # Добавляем сдвиг, если есть нули или отрицательные значения
            if np.min(vals) <= 0:
                shift = -np.min(vals) + 1e-6
                vals = vals + shift
            else:
                shift = 0
            
            try:
                transformed, lam = stats.boxcox(vals)
                # Применяем к train
                data[feat] = transformed
                # Применяем к test с тем же lam и сдвигом
                test_vals = test[feat].values
                test[feat] = stats.boxcox(test_vals + shift, lam)
            except Exception as e:
                print(f"Box-Cox failed for {feat}: {e}. Using fixed lambda={lambda_val}")
                data[feat] = stats.boxcox(data[feat].values + 1e-6, lambda_val)
                test[feat] = stats.boxcox(test[feat].values + 1e-6, lambda_val)
        else:
            # Фиксированный lambda
            data[feat] = stats.boxcox(data[feat].values + 1e-6, lambda_val)
            test[feat] = stats.boxcox(test[feat].values + 1e-6, lambda_val)
    
    return data, test

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