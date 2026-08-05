# ML: линейная и логистическая регрессия

Учебный репозиторий с двумя классическими Kaggle-задачами:

- **[Titanic](https://www.kaggle.com/c/titanic)** — предсказать выживаемость пассажира (бинарная классификация → **логистическая регрессия**).
- **[House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)** — предсказать цену дома (регрессия → **линейная регрессия / Ridge**).

Для каждой задачи есть: разведочный анализ данных (EDA) с объяснениями, честный пайплайн обучения с baseline-моделью для сравнения, и итоговая модель с воспроизводимыми метриками. Отдельно — теоретические ноутбуки по обоим методам и конспект вопросов к собеседованию.

## Структура репозитория

```
Ml/
├── data/
│   ├── titanic/{raw,processed}/          сырые и предобработанные данные Titanic
│   └── house_prices/{raw,processed}/     сырые и предобработанные данные House Prices
├── notebooks/
│   ├── 01_titanic_eda.ipynb              EDA + обоснование feature engineering (Titanic)
│   └── 02_house_prices_eda.ipynb         EDA + обоснование feature engineering (House Prices)
├── scripts/
│   ├── download_data.py                  скачивание raw-данных через Kaggle API
│   ├── train_titanic.py                  полный пайплайн: baseline → CV → final → метрики
│   └── train_house_prices.py             полный пайплайн: baseline → CV → final → метрики
├── src/
│   ├── titanic_preprocessing.py          очистка/инженерия признаков + sklearn-пайплайны (Titanic)
│   ├── house_prices_preprocessing.py     очистка/инженерия признаков + sklearn-пайплайны (House Prices)
│   └── evaluation.py                     общие метрики/отчёты для обеих задач
├── utils/
│   └── plotting.py                       все графические функции проекта (единый стиль)
├── models/
│   ├── titanic/{model.joblib,metrics.json,submission.csv,plots/}
│   └── house_prices/{model.joblib,metrics.json,submission.csv,plots/}
├── tutorials/
│   ├── linearregression.ipynb            теория линейной регрессии с нуля
│   └── logisticregression.ipynb          теория логистической регрессии с нуля
├── questions.md                          конспект вопросов к собеседованию (линейная регрессия)
├── pyproject.toml
└── requirements.txt
```

## Установка

```bash
git clone <repo-url>
cd Ml

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .                 # делает src/ и utils/ импортируемыми откуда угодно
```

`pip install -e .` — ключевой шаг: после него `from src.titanic_preprocessing import ...` и `from utils.plotting import ...` работают и в ноутбуках, и в скриптах, без ручных `sys.path.insert(...)`.

## Данные

Сырые данные уже лежат в `data/titanic/raw/` и `data/house_prices/raw/` — это подлинные CSV с Kaggle, ничего скачивать не обязательно, чтобы всё запустить.

Чтобы перекачать их заново (или в чистом окружении без этого репозитория):

1. Создать API-токен: [kaggle.com/settings](https://www.kaggle.com/settings) → *API* → *Create New Token* → положить `kaggle.json` в `~/.kaggle/kaggle.json` (`chmod 600`).
2. Принять правила соревнований: [titanic/rules](https://www.kaggle.com/c/titanic/rules), [house-prices.../rules](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/rules).
3. `python scripts/download_data.py`

Обработанные датасеты (`data/*/processed/*.csv`) генерируются автоматически при запуске `scripts/train_*.py` — их не нужно готовить отдельно.

## Как пользоваться репозиторием

Порядок ниже — от «посмотреть и понять» к «запустить самому». Все команды
выполняются из корня репозитория с активированным `.venv`
(`source .venv/bin/activate`).

### 1. Ноутбуки с EDA (сначала сюда)

Открыть в Jupyter/VS Code — по одному на задачу:

```bash
jupyter lab notebooks/01_titanic_eda.ipynb
jupyter lab notebooks/02_house_prices_eda.ipynb
```

Здесь разведочный анализ с объяснением, почему принято то или иное решение
по предобработке (что делать с пропусками, какие признаки инженерить, как
кодировать категории, где в данных мультиколлинеарность и как с ней
справляемся) и явный вывод после каждого раздела — как именно результат
исследования повлияет на препроцессинг. Все функции, которые здесь только
показываются и объясняются, реально используются ниже в `scripts/` —
никакого расхождения между "тем, что объяснили" и "тем, что выполнилось".

### 2. Теория метода (если нужно освежить сам метод)

```bash
jupyter lab tutorials/linearregression.ipynb
jupyter lab tutorials/logisticregression.ipynb
```

Вывод функции потерь, градиентный спуск с нуля, регуляризация, диагностика
модели, метрики — на синтетических данных, изолированно от конкретной
задачи.

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

Лучший `C` по кросс-валидации для logisticregression: `0.3` (CV ROC-AUC = 0.876). Подробности решений — `notebooks/01_titanic_eda.ipynb`; код — `src/titanic_preprocessing.py`; диагностические графики (confusion matrix, ROC-кривая, коэффициенты модели) — `models/titanic/plots/`.
* **SVM (Linear)**: лучший параметр `C = 0.01`
* **SVM (RBF)**: лучшие параметры `C = 1`, `gamma = 0.1`

Логистическая регрессия с FE показала лучший результат по ROC-AUC (0.8648) и максимальный Recall (0.7391), обеспечивая наилучшее ранжирование вероятностей и баланс между полнотой и точностью.SVM с RBF-ядром показал наивысшие Accuracy (83.8%) и Precision (85.7%), совершая меньше всего ложноположительных ошибок. SVM справилась хуже Baseline, как и стоило ожидать, ведь данные линейно неразделимы

Итог:
Вердикт
 - Final (LR) лучше, если важна вероятностная оценка.
 - SVM (RBF) лучше, если нужна максимальная точность жесткой бинарной классификации (выжил / не выжил).

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

Лучшая `alpha` по кросс-валидации: `30` (CV RMSE на log-таргете = 0.130). Подробности решений (включая разбор мультиколлинеарности через VIF) — `notebooks/02_house_prices_eda.ipynb`; код — `src/house_prices_preprocessing.py`; диагностические графики (residuals, predicted vs actual, коэффициенты модели) — `models/house_prices/plots/`.

## Теория и справочные материалы

- `tutorials/linearregression.ipynb`, `tutorials/logisticregression.ipynb`, `tutorials/svm.ipynb` — теория методов с нуля: постановка задачи, функция потерь, аналитическое и итеративное решение, регуляризация, диагностика, метрики.
- `questions.md` — личный конспект вопросов к собеседованию по линейной регрессии (справочный материал, не туториал).
