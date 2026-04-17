import pandas as pd
import numpy as np
from scipy.stats import norm, skew
from scipy.stats import boxcox
from scipy import stats

def fill_missing_values(train,test):

    cat_cols = train.select_dtypes(include=['object']).columns
    num_cols = train.select_dtypes(exclude=['object']).columns

    for col in cat_cols:
        train[col] = train[col].fillna('None')
        test[col] = test[col].fillna('None')

    for col in num_cols:
        train[col] = train[col].fillna(0)
        test[col] = test[col].fillna(0)
    return train,test

def analyze_outliers_iqr(train):
    
    results = []
    num_cols = train.select_dtypes(include=[np.number]).columns
    
    for col in num_cols:
        Q1 = train[col].quantile(0.25)
        Q3 = train[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Если IQR = 0, пропускаем (все значения одинаковые)
        if IQR == 0:
            continue
            
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        

        outliers = train[(train[col] < lower_bound) | (train[col] > upper_bound)]
        
        if len(outliers) > 0:
            results.append({
                'Column': col,
                'Outliers_Count': len(outliers),
                'Outliers_Pct': round(len(outliers) / len(train) * 100, 2),
                'Lower_Bound': round(lower_bound, 2),
                'Upper_Bound': round(upper_bound, 2),
                'Min_Outlier': round(outliers[col].min(), 2),
                'Max_Outlier': round(outliers[col].max(), 2)
            })
    
    df_outliers = pd.DataFrame(results).sort_values('Outliers_Count', ascending=False)
    return df_outliers




def remove_outliers_iqr(train):

    train_clean = train.copy()
    num_cols = train_clean.select_dtypes(include=[np.number]).columns
    
    for col in num_cols:
        Q1 = train_clean[col].quantile(0.25)
        Q3 = train_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lim_min = Q1 - 3 * IQR
        lim_max = Q3 + 3 * IQR
        
        train_clean = train_clean[
            (train_clean[col] >= lim_min) & 
            (train_clean[col] <= lim_max)
        ]
    
    removed = len(train) - len(train_clean)
    print(f"Удалено строк: {removed} ({removed/len(train)*100:.2f}%)")
    return train_clean

def transform_skewed_features(train, test, threshold=0.75, lambda_val=0.15, auto_lambda=False):
    
    numerical_features = train.select_dtypes(include=[np.number]).columns
    
    # Вычисляем асимметрию только на train
    skewness = train[numerical_features].apply(lambda x: skew(x.dropna()))
    skewed_features = skewness[abs(skewness) > threshold]
    positive_skew = skewed_features[skewed_features > threshold].index
    negative_skew = skewed_features[skewed_features < -threshold].index
    
    # Храним lambdas для положительных признаков (если auto_lambda)
    lambdas = {}
    
    # Обработка положительной асимметрии
    for feat in positive_skew:
        if auto_lambda:
            # Подбираем lambda по train, игнорируя нули/отрицательные (используем сдвиг)
            data = train[feat].values
            # Добавляем маленький сдвиг, если есть нули
            if np.min(data) <= 0:
                shift = -np.min(data) + 1e-6
                data = data + shift
            else:
                shift = 0
            # Box-Cox (требует положительных значений)
            try:
                transformed, lam = boxcox(data)
                lambdas[feat] = (lam, shift)
                train[feat] = transformed
                # Применяем к test: сначала сдвиг, потом Box-Cox с тем же lam
                test[feat] = boxcox(test[feat] + shift, lam)
            except:
                # Если не получилось, используем фиксированный lambda
                train[feat] = boxcox(train[feat], lambda_val)
                test[feat] = boxcox(test[feat], lambda_val)
        else:
            # Фиксированный lambda_val
            train[feat] = boxcox(train[feat], lambda_val)
            test[feat] = boxcox(test[feat], lambda_val)
    
    # Обработка отрицательной асимметрии (зеркальное логарифмирование)
    for feat in negative_skew:
        max_train = train[feat].max()
        # Сдвиг, чтобы минимальное значение стало положительным
        min_val = train[feat].min()
        shift = -min_val + 1e-6 if min_val <= 0 else 0
        train[feat] = np.log1p(max_train - train[feat] + shift + 1e-6)
        # Для теста используем тот же max_train (из train)
        test[feat] = np.log1p(max_train - test[feat] + shift + 1e-6)
        
    return train, test

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
