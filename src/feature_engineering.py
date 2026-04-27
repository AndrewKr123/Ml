import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from itertools import combinations
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

def mi_interaction_features(X, y, top_k=20, top_interactions=20):
    """
    Автоматически создаёт interaction features через Mutual Information.
    """

    X = X.copy()

    # 1. MI для одиночных признаков
    mi_single = mutual_info_regression(X, y)
    mi_single = pd.Series(mi_single, index=X.columns)

    # 2. выбираем top-K признаков
    top_features = mi_single.sort_values(ascending=False).head(top_k).index.tolist()

    interactions = []

    # 3. считаем MI для пар
    for f1, f2 in combinations(top_features, 2):

        pair = X[[f1, f2]].copy()
        mi_pair = mutual_info_regression(pair, y).sum()

        # приближение "выигрыша информации"
        gain = mi_pair - mi_single[f1] - mi_single[f2]

        interactions.append((f1, f2, gain))

    # 4. сортируем по полезности
    interactions.sort(key=lambda x: x[2], reverse=True)

    # 5. берём лучшие пары
    best_pairs = interactions[:top_interactions]

    # 6. создаём новые признаки
    for f1, f2, gain in best_pairs:
        new_name = f"{f1}_x_{f2}"
        X[new_name] = X[f1] * X[f2]

    return X, best_pairs