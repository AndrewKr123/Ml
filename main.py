import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error

# Импортируем ваши функции
from src.transformations import encode_features, RobustScalerWrapper, scale_features
from src.preprocess import to_drop

# 1. Загрузка и базовая очистка
df = pd.read_csv('data/saleprice/train.csv')
df = to_drop(df)

# ИСПРАВЛЕНИЕ 1: Явное присваивание, иначе fillna не сработает
numeric_cols = df.select_dtypes(include=[np.number]).columns.to_list()
# ИСПРАВЛЕНИЕ 2: Добавили 'string' чтобы убрать предупреждение Pandas
cat_cols = df.select_dtypes(include=['object', 'string', 'category']).columns.to_list()

df[numeric_cols] = df[numeric_cols].fillna(0)
df[cat_cols] = df[cat_cols].fillna('None')

# 2. Определяем группы признаков
ohe_cols = [
    'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
    'LotConfig', 'LandSlope',          
    'Condition1', 'Condition2',
    'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
    'Exterior1st', 'Exterior2nd', 
    'Foundation', 'Heating', 'CentralAir', 
    'Electrical', 'Functional', 'PavedDrive', 'SaleType', 
    'SaleCondition','GarageType'
] 

target_median_cols = ['Neighborhood', 'Exterior1st', 'Exterior2nd']

target_mean_cols = [
    'OverallQual', 'OverallCond', 'ExterQual', 'ExterCond',
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'GarageFinish'
]

# 3. Кодирование (сначала весь датасет, чтобы функция видела 'SalePrice')
df_encoded = encode_features(
    df,
    ohe_cols=ohe_cols,
    target_median_cols=target_median_cols,
    target_mean_cols=target_mean_cols,
    target_col='SalePrice'
)
df_encoded = encode_features(
    df,
    ohe_cols=ohe_cols,
    target_median_cols=target_median_cols,
    target_mean_cols=target_mean_cols,
    target_col='SalePrice'
)


# 4. Разделение на train и test
train_df, test_df = train_test_split(df_encoded, test_size=0.2, random_state=42)

# 5. ОТДЕЛЕНИЕ X и y (ОТВЕТ НА ВАШ ВОПРОС)
# X_train: берем всё, КРОМЕ 'Id' (бесполезен) и 'SalePrice' (это наш ответ)
X_train = train_df.drop(columns=['Id', 'SalePrice'])
# y_train: берем ТОЛЬКО целевую переменную
y_train = train_df['SalePrice']

# То же самое для теста
X_test = test_df.drop(columns=['Id', 'SalePrice'])
y_test = test_df['SalePrice']

# 6. Масштабирование (RobustScaler)
# Берем числовые колонки уже из X_train (там нет SalePrice и Id)
numeric_features_to_scale = X_train.select_dtypes(include=[np.number]).columns.tolist()

scaler = RobustScalerWrapper()
X_train_scaled, scaler = scale_features(X_train, numeric_features_to_scale, scaler, fit=True)
X_test_scaled, _ = scale_features(X_test, numeric_features_to_scale, scaler, fit=False)

print(f" X_train shape: {X_train_scaled.shape}")
print(f" X_test shape: {X_test_scaled.shape}")

# 7. Обучение модели
model = LinearRegression()
# Обучаем НА X_train и y_train
model.fit(X_train_scaled, y_train)

# 8. Предсказание и оценка
# Предсказываем на X_test_scaled
preds = model.predict(X_test_scaled)

# Сравниваем предсказания (preds) с реальными значениями из теста (y_test)
mse = mean_squared_error(y_test, preds)
rmse = root_mean_squared_error(y_test, preds) # root_mean_squared_error доступен в sklearn >= 1.4, np.sqrt надежнее
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print("МЕТРИКИ МОДЕЛИ")
print(f"MSE : {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE : {mae:.2f}")
print(f"R2  : {r2:.4f}")