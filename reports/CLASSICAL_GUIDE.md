> Историческое руководство до реорганизации. Текущая структура и команды — в [README](../README.md).

# ML: классические модели и градиентный бустинг

Учебный репозиторий с двумя классическими Kaggle-задачами:

- **[Titanic](https://www.kaggle.com/c/titanic)** — предсказать выживаемость пассажира (бинарная классификация → **логистическая регрессия**).
- **[House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)** — предсказать цену дома (регрессия → **линейная регрессия / Ridge**).

Для каждой задачи есть: разведочный анализ данных (EDA) с объяснениями, честный пайплайн обучения с baseline-моделью для сравнения, и итоговая модель с воспроизводимыми метриками. Отдельно — теоретические ноутбуки по обоим методам и конспект вопросов к собеседованию.

Дополнительно в проекте есть **Iris** — многоклассовая классификация трёх видов ириса. Для классификации сравниваются логистическая регрессия, SVM, деревья решений и градиентный бустинг; для House Prices — линейные модели, дерево и бустинг. Используются три специализированные библиотеки: CatBoost для Titanic, LightGBM для House Prices и XGBoost для Iris.

## Структура репозитория

```
Ml/
├── data/
│   ├── titanic/{raw,processed}/          сырые и предобработанные данные Titanic
│   ├── iris/preprocess                    предобработанные данные Iris
│   └── house_prices/{raw,processed}/     сырые и предобработанные данные House Prices
├── notebooks/
│   ├── 01_titanic_eda.ipynb              EDA + обоснование feature engineering (Titanic)
│   ├── 02_house_prices_eda.ipynb         EDA + обоснование feature engineering (House Prices)
│   └── 03_iris_eda.ipynb                 EDA для многоклассовой классификации (Iris)
├── scripts/
│   ├── download_data.py                  скачивание raw-данных через Kaggle API
│   ├── train_iris.py                     полный пайплайн: baseline → CV → final → метрики
│   ├── train_titanic.py                  полный пайплайн: baseline → CV → final → метрики
│   └── train_house_prices.py             полный пайплайн: baseline → CV → метрики → отчёт
├── src/
│   ├── titanic_preprocessing.py          очистка/инженерия признаков + sklearn-пайплайны (Titanic)
│   ├── house_prices_preprocessing.py     очистка/инженерия признаков + sklearn-пайплайны (House Prices)
│   ├── iris_preprocessing.py             очистка/инженерия признаков + sklearn-пайплайн(Iris)
│   ├── evaluation.py                     метрики регрессии, бинарной и многоклассовой классификации
│   ├── experiments/                      обучение: iris.py, titanic.py, house_prices.py
│   │   └── result.py                     TrainingResult: модель, метрики, предсказания и CV
│   └── reporting/                        сохранение артефактов отдельно для каждого датасета
├── utils/
│   └── plotting.py                       все графические функции проекта (единый стиль)
├── models/
│   ├── titanic/{model.joblib,metrics.json,submission.csv,plots/}
│   └── house_prices/{model.joblib,metrics.json,submission.csv,plots/}
│   └──iris/{model.joblib,metrics.json,submission.csv,plots/}
├── tutorials/
│   ├── linearregression.ipynb            теория линейной регрессии с нуля
│   ├── logisticregression.ipynb          теория логистической регрессии с нуля
│   ├── svm.ipynb                         теория метода опорных векторов с нуля
│   ├── randomforest.ipynb                теория случайного леса
│   ├── BVD.ipynb                         разложение ошибки на смещение, разброс и шум
│   ├── decisiontrees.ipynb               теория и практика деревьев решений
│   ├── gradientboosting.ipynb            градиентный бустинг над деревьями: антиградиент, потери и псевдоостатки
│   ├── kmeans.ipynb                      кластеризация K-means: теория, реализация с нуля и практика
│   ├── knn.ipynb                         метод ближайших соседей: классификация, регрессия и bias–variance
│   └── pca.ipynb                         метод главных компонент и его связь с SVD
├── questions/
│   ├── questions_linearregression.md     вопросы по линейной регрессии
│   └── questions_logisticregression.md   вопросы по логистической регрессии
├── pyproject.toml
└── requirements.txt
```

Обучающий код читается в следующем порядке:

1. `scripts/train_<dataset>.py` — загрузка, общий train/holdout-сплит, подготовка данных, запуск моделей и сравнение.
2. `src/experiments/<dataset>.py` — явные `Pipeline`, сетки параметров и `GridSearchCV`. Функции `train_baseline`, `train_logistic_cv`, `train_ridge_cv`, `train_svm_*_cv`, `train_decision_tree_cv` возвращают `TrainingResult`.
3. `src/reporting/<dataset>.py` — сохранение моделей, метрик, таблиц и графиков; функции построения графиков находятся в `utils/plotting.py`.

У результата доступны `model`, `metrics`, `predictions`, `probabilities` и `search`.
У baseline `search=None`, у регрессии `probabilities=None`. Для House Prices
`predictions` уже выражены в долларах, хотя сама модель предсказывает `log1p(SalePrice)`.
Общий препроцессор и логарифм target готовятся до обучения Ridge и дерева;
каждая модель получает свою копию препроцессора и может обучаться независимо.
Параметр `n_jobs` функций с CV позволяет ограничить параллелизм (по умолчанию `-1`).

Команды запуска и имена артефактов сохранены. Название `final` в старых ключах
JSON и отчётах обозначает LogisticRegression/Ridge с CV, а не автоматический
выбор победителя среди всех моделей.

Дополнение к дереву выше — актуальные пути и дополнительные файлы:

- `notebooks/eda/03_iris_eda.ipynb` — EDA для Iris.
- `data/iris/processed/{train,holdout}_processed.csv` — обучающая и отложенная части Iris; фактическое имя каталога — `processed`.
- `models/iris/holdout_predictions.csv` — истинные и предсказанные классы Iris; вместо Kaggle submission здесь сохраняются прогнозы на hold-out.
- `models/iris/{svm_rbf,decision_tree,xgboost}.joblib` — дополнительные модели Iris.
- `models/titanic/{svm_rbf,decision_tree_model,catboost}.joblib` — дополнительные модели Titanic.
- `models/house_prices/lightgbm.joblib` — модель градиентного бустинга для House Prices.
- `tutorials/classical/decisiontrees.ipynb` — фактическое имя ноутбука по деревьям решений.
- `tutorials/classical/randomforest.ipynb` — случайный лес.
- `tutorials/classical/BVD.ipynb` — разложение MSE, регуляризация, деревья и ансамбли.

Файл `scripts/train_iris_compact.py` пока пуст; для обучения используется `scripts/train_iris.py`.

## Установка

Требуется **Python 3.10 или новее** (`requires-python` в `pyproject.toml`). Версии библиотек закреплены в `requirements.txt`; зависимости устанавливаются отдельной командой перед установкой проекта в editable-режиме.

```bash
git clone <repo-url>
cd Ml

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .                 # делает src/ и utils/ импортируемыми откуда угодно
```

`pip install -e .` — ключевой шаг: после него `from src.datasets.titanic import ...` и `from utils.plotting import ...` работают и в ноутбуках, и в скриптах, без ручных `sys.path.insert(...)`.

## Данные

Сырые данные уже лежат в `data/titanic/raw/` и `data/house_prices/raw/` — это подлинные CSV с Kaggle, ничего скачивать не обязательно, чтобы всё запустить.

Чтобы перекачать их заново (или в чистом окружении без этого репозитория):

1. Создать API-токен: [kaggle.com/settings](https://www.kaggle.com/settings) → *API* → *Create New Token* → положить `kaggle.json` в `~/.kaggle/kaggle.json` (`chmod 600`).
2. Принять правила соревнований: [titanic/rules](https://www.kaggle.com/c/titanic/rules), [house-prices.../rules](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/rules).
3. `python scripts/download_data.py`

Обработанные датасеты (`data/*/processed/*.csv`) генерируются автоматически при запуске `scripts/train_*.py` — их не нужно готовить отдельно.

**Данные Iris:** `src.datasets.iris.load_raw()` по очереди ищет `data/iris/raw/Iris.csv`, `iris.csv`, `train.csv`. Если CSV нет, используется встроенный `sklearn.datasets.load_iris` — Kaggle и скачивание не нужны. Загрузчик приводит названия колонок к единому формату и кодирует виды: `setosa → 0`, `versicolor → 1`, `virginica → 2`. Служебные `id`, `species`, `target` исключаются из признаков.

## Как пользоваться репозиторием

Порядок ниже — от «посмотреть и понять» к «запустить самому». Все команды
выполняются из корня репозитория с активированным `.venv`
(`source .venv/bin/activate`).

### 1. Ноутбуки с EDA (сначала сюда)

Открыть в Jupyter/VS Code — по одному на задачу:

```bash
jupyter lab notebooks/eda/01_titanic_eda.ipynb
jupyter lab notebooks/eda/02_house_prices_eda.ipynb
jupiter lab notebooks/eda/03_iris_eda.ipynb
```

Здесь разведочный анализ с объяснением, почему принято то или иное решение
по предобработке (что делать с пропусками, какие признаки инженерить, как
кодировать категории, где в данных мультиколлинеарность и как с ней
справляемся) и явный вывод после каждого раздела — как именно результат
исследования повлияет на препроцессинг. Все функции, которые здесь только
показываются и объясняются, реально используются ниже в `scripts/` —
никакого расхождения между "тем, что объяснили" и "тем, что выполнилось".

Для Iris корректная команда запуска (уточнение опечатки `jupiter` выше):

```bash
jupyter lab notebooks/eda/03_iris_eda.ipynb
```

### 2. Теория метода (если нужно освежить сам метод)

```bash
jupyter lab tutorials/classical/linearregression.ipynb
jupyter lab tutorials/classical/logisticregression.ipynb
```

Вывод функции потерь, градиентный спуск с нуля, регуляризация, диагностика
модели, метрики — на синтетических данных, изолированно от конкретной
задачи.

Дополнительные теоретические ноутбуки:

```bash
jupyter lab tutorials/classical/svm.ipynb
jupyter lab tutorials/classical/decisiontrees.ipynb
jupyter lab tutorials/classical/randomforest.ipynb
jupyter lab tutorials/classical/BVD.ipynb
jupyter lab tutorials/classical/kmeans.ipynb
jupyter lab tutorials/classical/knn.ipynb
jupyter lab tutorials/classical/pca.ipynb
```

### 3. Скачивание сырых данных (опционально — они уже есть в репозитории)

```bash
python scripts/download_data.py            # скачает только то, чего ещё нет
python scripts/download_data.py --force     # перекачает всё заново
```

Требует `~/.kaggle/kaggle.json` — подробности в разделе [«Данные»](#данные) выше.

### 4. Обучение моделей

```bash
python scripts/train_titanic.py
python scripts/train_house_prices.py

# необязательные флаги (одинаковые у обоих скриптов):
python scripts/train_titanic.py --test-size 0.25 --random-state 0
```

- `--test-size` (по умолчанию `0.2`) — доля `train.csv`, отложенная под hold-out;
- `--random-state` (по умолчанию `42`) — фиксирует и train/hold-out сплит, и обучение модели, для воспроизводимости.

Каждый скрипт:

- делит `train.csv` на train/hold-out (у Kaggle `test.csv` нет меток — он не участвует в подсчёте метрик, только в генерации `submission.csv`);
- обучает **baseline**-модель без предобработки/инженерии признаков;
- обучает **финальную** модель (инженерия признаков + корректный `ColumnTransformer` + кросс-валидация для подбора гиперпараметра) и печатает лучший гиперпараметр;
- печатает метрики baseline vs final на одном и том же hold-out сплите (см. таблицы ниже);
- сохраняет в `models/<task>/`: `model.joblib` (обученный пайплайн), `metrics.json` (те же числа, что в консоли), `submission.csv` (предсказания на официальном Kaggle `test.csv`, для загрузки на Kaggle), `plots/*.png` (диагностические графики);
- перезаписывает `data/<task>/processed/{train,test}_processed.csv` — предобработанные версии датасета после `engineer_and_clean`/`engineer_features`.

Прогонять оба скрипта не обязательно вместе — они полностью независимы.

Для Iris доступен отдельный независимый запуск с теми же параметрами сплита:

```bash
python scripts/train_iris.py
python scripts/train_iris.py --test-size 0.25 --random-state 0
```

Он сравнивает baseline LR, LR с подбором `C`, линейную и RBF SVM, дерево решений и XGBoost. В `models/iris/` сохраняются LR (`model.joblib`), RBF SVM (`svm_rbf.joblib`), дерево (`decision_tree.joblib`), XGBoost (`xgboost.joblib`), `metrics.json`, `holdout_predictions.csv` и графики. Линейная SVM отдельно не сериализуется. В `data/iris/processed/` записываются `train_processed.csv` и `holdout_processed.csv`: исходные числовые признаки после удаления служебных колонок, с возвращёнными метками; масштабирование выполняется внутри пайплайна.

Уточнение к сохранению моделей: `model.joblib` содержит выбранную LR для Titanic/Iris или Ridge для House Prices, а не автоматически лучшую из всех сравниваемых моделей. Скрипты сохраняют модель после обучения на обучающей части сплита; дополнительного обучения на всей выборке перед сохранением и формированием submission нет.

### Случайный лес для всех трёх датасетов

Случайный лес включён в обычный запуск каждого скрипта. Для обучения только леса:

```bash
python scripts/train_titanic.py --random-forest-only
python scripts/train_house_prices.py --random-forest-only
python scripts/train_iris.py --random-forest-only
```

Используются `RandomForestClassifier` для Titanic/Iris и `RandomForestRegressor` для House Prices: 200 деревьев, подбор глубины, минимального числа объектов в листе и доли признаков через `GridSearchCV` (18 комбинаций, 5 фолдов). Критерии подбора: ROC-AUC для Titanic, accuracy для Iris и отрицательный RMSE логарифма цены для House Prices. Итоговые метрики House Prices считаются в долларах после `expm1`.

В `models/<dataset>/` сохраняются `random_forest.joblib`, `random_forest_metrics.json` (метрики train и hold-out в полях `train` и `test`, параметры и CV-оценка) и `random_forest_submission.csv` для Kaggle-датасетов либо `random_forest_holdout_predictions.csv` для Iris. Общий `metrics.json` дополняется ключом `random_forest`; режим `--random-forest-only` сохраняет остальные записи и не обновляет обработанные CSV. Флаги `--test-size` и `--random-state` доступны и в этом режиме.

Проверки обучения и сохранения результатов:

```bash
python -m unittest discover -s tests -v
```

### Устойчивость леса и бустинга: повторная кросс-валидация

```bash
python scripts/check_cv_stability.py
# Отдельный датасет и число повторов:
python scripts/check_cv_stability.py --datasets house_prices --folds 5 --repeats 5 --seed 2026 --jobs 4
```

Скрипт берёт параметры из сохранённых моделей леса и бустинга и обучает новые копии на 5 фолдах × 5 повторах. Используется только исходная обучающая часть (80% данных, исходный сплит восстанавливается из `metrics.json`); hold-out не участвует. Для классификации разбиения стратифицированы. Обе модели получают одинаковые фолды, случайность моделей фиксирована. Статистики заполнения пропусков и выбор логарифмируемых признаков рассчитываются только по обучающей части каждого фолда; кодирование и масштабирование обучаются внутри пайплайна.

Результаты в `models/<dataset>/`:

- `repeated_cv_folds.csv` — train/validation-метрики каждой модели на всех 25 фолдах;
- `repeated_cv_repeats.csv` — средние метрики пяти фолдов каждого повтора;
- `repeated_cv_summary.json` — средние, стандартные отклонения, диапазоны, попарное сравнение моделей, параметры и индексы разбиений.

Это диагностика устойчивости фиксированных конфигураций, а не вложенная CV: гиперпараметры ранее подбирались на этих же обучающих объектах. Фолды зависимы, поэтому стандартное отклонение здесь описывает разброс и не является доверительным интервалом. Средние за повтор — арифметические средние метрик фолдов. Обученные ранее модели, `metrics.json` и submission-файлы не перезаписываются.

Результаты проверки: 5 × 5, `seed=2026`, исходный сплит `test_size=0.2`, `random_state=42`. В таблице — среднее ± стандартное отклонение по 25 validation-фолдам; в последней колонке — диапазон пяти средних за повтор.

| Датасет / метрика | Модель | Validation: среднее ± std | Средние за повтор: min–max |
|---|---|---:|---:|
| Iris / accuracy | Random Forest | 95,67% ± 3,06 п.п. | 95,00–96,67% |
| Iris / accuracy | XGBoost | 95,33% ± 4,55 п.п. | 94,17–95,83% |
| Titanic / ROC-AUC | Random Forest | 0,8807 ± 0,0279 | 0,8766–0,8862 |
| Titanic / ROC-AUC | CatBoost | 0,8856 ± 0,0222 | 0,8790–0,8919 |
| House Prices / RMSE(log1p) | Random Forest | 0,1345 ± 0,0138 | 0,1324–0,1392 |
| House Prices / RMSE(log1p) | LightGBM | 0,1238 ± 0,0140 | 0,1212–0,1301 |
| House Prices / RMSE, $ | Random Forest | 28 583 ± 3 683 | 27 856–30 010 |
| House Prices / RMSE, $ | LightGBM | 25 816 ± 4 296 | 24 846–28 625 |

На Iris разница мала: XGBoost выигрывает у леса на 6 фолдах, проигрывает на 7, на 12 — равенство. На Titanic CatBoost выигрывает по ROC-AUC на 18 из 25 фолдов; средняя accuracy — 83,37% против 82,78% у леса. На House Prices LightGBM выигрывает по RMSE логарифма цены на всех 25 фолдах; его средний долларовый RMSE примерно на 9,7% ниже. Это описательное сравнение на одинаковых фолдах, не тест статистической значимости.

Разрыв train/validation сохраняется: на Titanic средний train AUC у леса — 0,9436, у CatBoost — 0,9897; на House Prices средний train RMSE — 13 954 $ и 7 043 $ соответственно у леса и LightGBM. Более сильная подгонка бустинга под train не исключает лучшего качества на validation. Увеличение глубины само по себе этими результатами не обосновано; для следующего эксперимента полезнее совместно проверить регуляризацию и сложность моделей, сохраняя hold-out вне подбора.

## Titanic — логистическая регрессия

**Baseline** (`LogisticRegression`): только `Pclass, Sex, Age, Fare, SibSp, Parch`, без масштабирования и инженерии признаков — то, что сделал бы новичок в первый день.

**Final**: инженерия признаков (`Title` из имени, `FamilySize`/`IsAlone`, возраст, импутированный по группам `Title`, `Fare` после `log1p`, палуба `Deck` из `Cabin`), корректный `ColumnTransformer` (`StandardScaler` + `OneHotEncoder`), подбор `C` через `GridSearchCV` + `StratifiedKFold` (5 фолдов, метрика ROC-AUC).

**SVM (Linear)**: линейный метод опорных векторов на подготовленных признаках с подбором `C`.
**SVM (RBF)**: метод опорных векторов с радиально-базисным ядром для учета нелинейных зависимостей, подбор `C` и `gamma`.

Метрики на одном и том же hold-out сплите (20% от `train.csv`), воспроизводятся `python scripts/train_titanic.py`:

| Метрика | Baseline | Final | Δ (Final) | SVM (Linear) | Δ (SVM Lin) | SVM (RBF) | Δ (SVM RBF) |
| ---     | ---      | ---   | ---       | ---          | ---         | ---       | ---         |
| Accuracy| 0.8045   | 0.8268| +0.022    | 0.7821       | -0.022      | 0.8380    | +0.034      |
| Precision| 0.7656  | 0.7969 | +0.031   | 0.7500       | -0.016      | 0.8571    | +0.091      |
| Recall  | 0.7101   | 0.7391 | +0.029   | 0.6522       | -0.058      | 0.6957    | -0.014      |
| F1      | 0.7368   | 0.7669 | +0.030   | 0.6977       | -0.039      | 0.7680    | +0.031      |
| ROC-AUC | 0.8515   | 0.8648 | +0.013   | 0.8465       | -0.005      | 0.8557    | +0.004      |

Лучший `C` по кросс-валидации для logisticregression: `0.3` (CV ROC-AUC = 0.876). Подробности решений — `notebooks/eda/01_titanic_eda.ipynb`; код — `src/datasets/titanic.py`; диагностические графики (confusion matrix, ROC-кривая, коэффициенты модели) — `models/titanic/plots/`.
* **SVM (Linear)**: лучший параметр `C = 0.01`
* **SVM (RBF)**: лучшие параметры `C = 1`, `gamma = 0.1`

Логистическая регрессия с FE показала лучший результат по ROC-AUC (0.8648) и максимальный Recall (0.7391), обеспечивая наилучшее ранжирование вероятностей и баланс между полнотой и точностью.SVM с RBF-ядром показал наивысшие Accuracy (83.8%) и Precision (85.7%), совершая меньше всего ложноположительных ошибок. SVM справилась хуже Baseline, как и стоило ожидать, ведь данные линейно неразделимы

Итог:
 - Final (LR) лучше, если важна вероятностная оценка.
 - SVM (RBF) лучше, если нужна максимальная точность жесткой бинарной классификации (выжил / не выжил).

**Дополнение: Decision Tree** (`DecisionTreeClassifier`) обучается на подготовленных признаках с подбором параметров через ту же 5-фолдовую CV по ROC-AUC. Сохранённые результаты из `models/titanic/metrics.json` (`test_size=0.2`, `random_state=42`):

| Метрика | Decision Tree |
| --- | --- |
| Accuracy | 0.8212 |
| Precision | 0.7681 |
| Recall | 0.7681 |
| F1 | 0.7681 |
| ROC-AUC | 0.8509 |

Лучшие параметры: `criterion=entropy`, `max_depth=4`, `min_samples_leaf=4`, `min_samples_split=2`, `class_weight=balanced`; CV ROC-AUC = **0.8669**. Дерево сохраняется в `models/titanic/decision_tree_model.joblib`. С учётом дерева максимальный Recall среди сохранённых результатов — **0.7681**; LR сохраняет преимущество по hold-out ROC-AUC.

**CatBoost** выбран для Titanic как библиотека, ориентированная на табличные данные с категориальными признаками. 
| Метрика | CatBoost |
| --- | ---: |
| Accuracy | 0.8324 |
| Precision | 0.8095 |
| Recall | 0.7391 |
| F1 | 0.7727 |
| ROC-AUC | 0.8485 |

Лучшие параметры: `depth=5`, `iterations=300`, `learning_rate=0.1`, `l2_leaf_reg=7`; CV ROC-AUC = **0.8937**. Это лучший CV ROC-AUC среди проверенных моделей Titanic, хотя на конкретном hold-out LR выше по ROC-AUC. Модель сохраняется в `models/titanic/catboost.joblib`.

## House Prices — линейная регрессия

**Baseline** (`LinearRegression`): все сырые числовые колонки как есть, только медианная импутация — без кодирования категорий, без чистки мультиколлинеарности, без лог-таргета.

**Final** (`Ridge`): чистка признаков-дублей (`GrLivArea`, `TotalBsmtSF` и др. — идеальная мультиколлинеарность, см. EDA), инженерия (`TotalBath`, `TotalPorchSF`, `HouseAge`, `RemodAge`, `QualArea`), `log1p(SalePrice)` + логарифмирование скошенных числовых признаков, `ColumnTransformer` (`OrdinalEncoder` для порядковых шкал качества, `OneHotEncoder` для номинальных категорий, честный `TargetEncoder` с cross-fitting для `Neighborhood`) + `RobustScaler`, подбор `alpha` через `GridSearchCV` + `KFold` (5 фолдов, метрика RMSE на log-таргете).

Метрики на одном и том же hold-out сплите (20% от `train.csv`, в долларах, после обратного `expm1`-преобразования), воспроизводятся `python scripts/train_house_prices.py`:

| Метрика | Baseline | Final | Δ |
|---|---|---|---|
| MAE | $22 978.93 | $16 414.40 | −$6 564.54 |
| RMSE | $36 839.73 | $28 243.03 | −$8 596.70 |
| MAPE | 13.46% | 9.39% | −4.07 п.п. |
| R² | 0.823 | 0.896 | +0.073 |

Лучшая `alpha` по кросс-валидации: `30` (CV RMSE на log-таргете = 0.130). Подробности решений (включая разбор мультиколлинеарности через VIF) — `notebooks/eda/02_house_prices_eda.ipynb`; код — `src/datasets/house_prices.py`; диагностические графики (residuals, predicted vs actual, коэффициенты модели) — `models/house_prices/plots/`.

**Дополнение: Decision Tree** (`DecisionTreeRegressor`) использует подготовленные признаки и `log1p(SalePrice)`. Глубина и минимальные размеры узла/листа подбираются через `GridSearchCV` с 5-фолдовой CV по RMSE логарифма цены. После `expm1` сохранены следующие метрики (`models/house_prices/metrics.json`):

| Метрика | Decision Tree |
| --- | --- |
| MAE | $22 894.41 |
| RMSE | $36 688.19 |
| MAPE | 13.27% |
| R² | 0.825 |

Лучшие параметры: `max_depth=None`, `min_samples_leaf=8`, `min_samples_split=20`. Дерево немного улучшает baseline, но уступает Ridge по всем четырём метрикам. В JSON результаты находятся под ключом `decision tree`, параметры — `best params for tree`; отдельный `.joblib` для дерева этот скрипт не сохраняет.

**LightGBM** выбран для House Prices как быстрый гистограммный бустинг для широких табличных данных. Он обучается на том же `log1p(SalePrice)` и оценивается после обратного `expm1`.

| Метрика | LightGBM |
| --- | ---: |
| MAE | **$16 236.66** |
| RMSE | $28 848.83 |
| MAPE | 9.66% |
| R² | 0.8915 |

Лучшие параметры: `n_estimators=500`, `learning_rate=0.03`, `num_leaves=15`, `max_depth=-1`, `min_child_samples=10`; CV RMSE на log-таргете = **0.1228**. LightGBM улучшил Ridge по MAE и CV RMSE, а Ridge остался лучше по hold-out RMSE, MAPE и R². Модель сохраняется в `models/house_prices/lightgbm.joblib`.

## Iris — многоклассовая классификация сортов ириса

**Уточнение для текущего кода:** приведённое ниже описание поиска PCA относится к предыдущему варианту эксперимента. Сейчас `build_iris_pipeline()` содержит `ColumnTransformer` с `RobustScaler` и модель; шага PCA и его перебора в `scripts/train_iris.py` нет. Baseline также использует этот масштабировщик. Таблицы ниже соответствуют сохранённому `models/iris/metrics.json`; JSON не содержит лучших гиперпараметров Iris и не подтверждает прежние значения `pca`. Актуальный путь EDA — `notebooks/eda/03_iris_eda.ipynb`.

**Baseline** (`LogisticRegression`): стандартная логистическая регрессия с дефолтными параметрами (`C=1`, L2-регуляризация) на четырёх числовых признаках (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`) без масштабирования и снижения размерности — то, что сделал бы новичок в первый день.

**Final (LR + PCA search)**: логистическая регрессия с L2-регуляризацией внутри `Pipeline` (`RobustScaler` → опционально `PCA` → модель), подбор `C` и наличия PCA через `GridSearchCV` + `StratifiedKFold` (5 фолдов, метрика accuracy). По результатам поиска PCA не улучшил качество, лучшая конфигурация — без снижения размерности.

**SVM (Linear)**: линейный метод опорных векторов на масштабированных признаках с опциональным PCA, подбор `C`.
**SVM (RBF)**: метод опорных векторов с радиально-базисным ядром для учёта нелинейных границ, подбор `C` и `gamma`.
**Decision Tree**: дерево решений с подбором глубины, минимального размера листа/узла, критерия разбиения и весов классов.
**XGBoost**: многоклассовый градиентный бустинг (`multi:softprob`) с подбором числа и глубины деревьев, learning rate и subsample.

Метрики на одном и том же hold-out сплите (20% от датасета, 30 объектов), воспроизводятся командой `python scripts/train_iris.py`:

| Метрика | Baseline | Final LR | Δ (Final) | SVM Linear | Δ (SVM Lin) | SVM RBF | Δ (SVM RBF) | Decision Tree | Δ (DT) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Accuracy | 0.9333 | 0.9667 | +0.033 | **1.0000** | **+0.067** | 0.9667 | +0.033 | 0.9667 | +0.033 |
| F1 Macro | 0.9333 | 0.9666 | +0.033 | **1.0000** | **+0.067** | 0.9666 | +0.033 | 0.9666 | +0.033 |
| Precision Macro | 0.9333 | 0.9697 | +0.036 | **1.0000** | **+0.067** | 0.9697 | +0.036 | 0.9697 | +0.036 |
| Recall Macro | 0.9333 | 0.9667 | +0.033 | **1.0000** | **+0.067** | 0.9667 | +0.033 | 0.9667 | +0.033 |
| ROC-AUC OVR | 0.9883 | **1.0000** | +0.012 | **1.0000** | +0.012 | **1.0000** | +0.012 | 0.9717 | −0.017 |

Hold-out содержит всего 30 объектов; одна ошибка меняет accuracy на ±0.033. Для надёжного сравнения ориентируйтесь на CV-score.

### Результаты кросс-валидации (5-fold StratifiedKFold, accuracy)

| Модель | CV Accuracy | Holdout Accuracy |
| --- | --- | --- |
| Final LR | 0.9667 | 0.9667 |
| **SVM Linear** | **0.9750** | **1.0000** |
| **SVM RBF** | **0.9750** | 0.9667 |
| Decision Tree | 0.9583 | 0.9667 |
| XGBoost | 0.9667 | 0.9667 |

Лучшие гиперпараметры по кросс-валидации:
- **Final LR**: `C = 10`, `pca = passthrough` (PCA не выбран)
- **SVM Linear**: `C = 3`, `pca = passthrough`
- **SVM RBF**: `C = 10`, `gamma = 0.1`, `pca = passthrough`
- **Decision Tree**: `criterion = gini`, `max_depth = 3`, `min_samples_leaf = 1`, `min_samples_split = 2`, `class_weight = None`, `pca = passthrough`
- **XGBoost**: `n_estimators = 50`, `max_depth = 3`, `learning_rate = 0.03`, `subsample = 1.0`

XGBoost получил hold-out accuracy **0.9667**, F1 macro **0.9666** и ROC-AUC OVR **0.9800**. На маленьком hold-out он совпал по accuracy с LR, RBF SVM и деревом, но линейный SVM остался лучшим. Модель сохраняется в `models/iris/xgboost.joblib`.

Подробности EDA — `notebooks/02_iris_eda.ipynb`; код предобработки — `src/datasets/iris.py`; диагностические графики (confusion matrix, ROC-кривые OVR) — `models/iris/plots/`.

---

### Анализ результатов

**PCA не дал преимущества ни для одной модели.** Все лучшие конфигурации выбрали `passthrough`. Для Iris с 4 информативными признаками снижение размерности через PCA (неконтролируемый метод) скорее теряет направление, важное для разделения классов, чем убирает шум. Мультиколлинеарность между `petal_length` и `petal_width` (~0.96) не мешает предсказаниям: L2-регуляризация стабилизирует веса, а сами признаки сильно разделяют классы.

**SVM Linear показал лучший CV-score (0.975) и идеальный holdout (1.0).** Линейная разделяющая поверхность достаточна для Iris; после `RobustScaler` SVM эффективно строит гиперплоскости между тремя классами. Идеальная точность на holdout частично обусловлена малым размером теста (30 объектов), но высокий CV подтверждает устойчивость.

**Final LR и SVM RBF показали одинаковый holdout (0.9667)**, но SVM RBF имеет более высокий CV (0.975 vs 0.967). RBF-ядро не дало преимущества перед линейным, что подтверждает линейную разделимость данных.

**Decision Tree оказался слабее всех по CV (0.958).** Одиночное дерево строит ось-параллельные разбиения и хуже аппроксимирует гладкие границы между `versicolor` и `virginica`. Для улучшения нужны ансамбли (Random Forest, Gradient Boosting).

### Итог

- **SVM Linear** — лучший выбор по совокупности CV и holdout; простая, быстрая, интерпретируемая модель.
- **Final LR** — лучший выбор, если нужны калиброванные вероятности и ROC-AUC; уступает SVM по CV лишь на 0.008.
- **SVM RBF** — сопоставима с LR, но сложнее и не даёт выигрыша над линейным вариантом.
- **Decision Tree** — подходит как baseline или для интерпретации правил, но проигрывает линейным моделям.
- **PCA** — не нужен для предсказания на Iris; полезен только для визуализации (2D-проекция).



## Теория и справочные материалы

- `tutorials/classical/linearregression.ipynb`, `tutorials/classical/logisticregression.ipynb`, `tutorials/classical/svm.ipynb` — теория методов с нуля: постановка задачи, функция потерь, аналитическое и итеративное решение, регуляризация, диагностика, метрики.
- `questions.md` — личный конспект вопросов к собеседованию по линейной регрессии (справочный материал, не туториал).


Дополнительные материалы и уточнение путей:

- `tutorials/classical/decisiontrees.ipynb` — энтропия, критерии разбиения, собственная реализация дерева классификации, деревья регрессии и визуализация.
- `tutorials/classical/randomforest.ipynb` — bootstrap, случайный выбор признаков и усреднение прогнозов деревьев.
- `tutorials/classical/BVD.ipynb` — вывод разложения MSE на bias², variance и шум; OLS, Ridge, Lasso, влияние глубины деревьев и ансамбли.
- `questions/questions_linearregression.md` и `questions/questions_logisticregression.md` — фактические пути конспектов вопросов по двум видам регрессии.

### KNN для всех трёх задач

KNN включён в обычный запуск. Для отдельного обучения:

```bash
python scripts/train_titanic.py --knn-only
python scripts/train_house_prices.py --knn-only
python scripts/train_iris.py --knn-only
```

`KNeighborsClassifier` решает задачи Titanic и Iris, `KNeighborsRegressor`
предсказывает `log1p(SalePrice)` для House Prices. GridSearchCV перебирает
`n_neighbors=[3,5,7,11,15,21]`, `weights=[uniform,distance]` и `p=[1,2]`
(манхэттенское/евклидово расстояние): 24 комбинации, 5 фолдов.
Критерии выбора — ROC-AUC, accuracy и отрицательный RMSE логарифма цены
соответственно. Используются прежние сплиты (hold-out 20%, seed 42).
Масштабирование числовых признаков и кодирование категорий выполняются
внутри пайплайна; hold-out не участвует в подборе параметров.
Общая подготовка Titanic/House Prices сохраняет существующий порядок:
групповая импутация и выбор логарифмируемых признаков выполняются на всём
train до CV. Поэтому CV не полностью изолирует эти этапы между фолдами;
отложенная выборка остаётся независимой.

Артефакты: `models/<dataset>/knn.joblib`, `knn_metrics.json` (train, test,
лучшие параметры и CV), `knn_submission.csv` для Kaggle или
`knn_holdout_predictions.csv` для Iris. Общий `metrics.json` дополняется
ключом `knn`. Модель обучена на train-части; House Prices требует `expm1`
для перевода её прогнозов в доллары. Вход сохранённых моделей — признаки
после общей подготовки соответствующего скрипта.
`--knn-only` и `--random-forest-only` взаимоисключающие.
