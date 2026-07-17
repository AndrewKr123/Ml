import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error

train = pd.read_csv('data/saleprice/train.csv')
test = pd.read_csv('data/saleprice/test.csv')
y_house = train['SalePrice']
train=train.drop('SalePrice',axis=1)

# 1. Симулируем датасет (в реальности здесь будет pd.read_csv('train.csv'))
np.random.seed(42)
n_samples = 200
data = pd.DataFrame({
    'LotArea': np.random.exponential(scale=10000, size=n_samples), # числовой с выбросами
    'OverallQual': np.random.randint(1, 11, size=n_samples),       # числовой
    'Neighborhood': np.random.choice(['CollgCr', 'Veenker', 'Crawfor'], size=n_samples), # категория
    'SalePrice': np.random.normal(loc=180000, scale=50000, size=n_samples) # целевая переменная
})

# Выделяем признаки и таргет
X = data.drop(columns=['SalePrice'])
y = data['SalePrice']

# Важно для House Prices: логарифмируем таргет, чтобы сгладить выбросы
y_log = np.log1p(y)
y_house = np.log1p(y_house)
# Шаг 1: Разделение на Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

# 2. Группируем признаки по типам
num_features = ['LotArea', 'OverallQual']
cat_features = ['Neighborhood']

# 3. Создаем пайплайны пред-обработки (Preprocessing)
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # заполняем пропуски медианой
    ('scaler', RobustScaler())                    # масштабируем, устойчиво к выбросам
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # заполняем пропуски модой
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # кодируем
])

# Объединяем трансформации колонок
preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

# 4. Создаем сквозной Пайплайн (Предобработка -> Отбор признаков -> Модель)
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('feature_selection', SelectKBest(score_func=f_regression)), # отбор K лучших признаков
    ('model', Ridge()) # Линейная модель с L2 регуляризацией
])

# 5. Сетка гиперпараметров для настройки
param_grid = {
    'feature_selection__k': [2, 3, 'all'], # сколько признаков оставить после OneHot
    'model__alpha': [0.1, 1.0, 10.0, 100.0] # сила регуляризации Ridge
}

# 6. Настройка гиперпараметров с помощью Кросс-Валидации (GridSearchCV)
# cv=5 означает, что train разобьется на 5 частей (фолдов)
grid_search = GridSearchCV(
    estimator=full_pipeline, 
    param_grid=param_grid, 
    cv=5, 
    scoring='neg_mean_squared_error', # оптимизируем MSE
    n_jobs=-1
)

# Вот здесь происходит ВСЁ: предобработка, кросс-валидация, отбор признаков,
# выбор лучших параметров и финальное обучение на всем X_train!
grid_search.fit(X_train, y_train)

print(f"Лучшие параметры: {grid_search.best_params_}")

# 7. Оценка качества на отложенном тест-сете
# grid_search автоматически использует лучшую обученную модель при вызове predict
y_pred_log = grid_search.predict(X_test)

# Возвращаем из логарифмического масштаба в реальные доллары
y_test_real = np.expm1(y_test)
y_pred_real = np.expm1(y_pred_log)

rmse = root_mean_squared_error(y_test_real, y_pred_real)
print(f"RMSE на тестовой выборке: ${rmse:.2f}")

X_train, X_test, y_train, y_test = train_test_split(train, y_house, test_size=0.2, random_state=42)
grid_search.fit(X_train, y_train)
print(f"Лучшие параметры: {grid_search.best_params_}")
y_pred_log = grid_search.predict(X_test)
rmse_log = root_mean_squared_error(y_test, y_pred_log)
print(rmse_log)
y_test_real = np.expm1(y_test)
y_pred_real = np.expm1(y_pred_log)
rmse = root_mean_squared_error(y_test_real, y_pred_real)
print(f"RMSE на тестовой выборке: ${rmse:.2f}")