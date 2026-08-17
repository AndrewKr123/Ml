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
│   ├── iris/preprocess                    предобработанные данные Iris
│   └── house_prices/{raw,processed}/     сырые и предобработанные данные House Prices
├── notebooks/
│   ├── 01_titanic_eda.ipynb              EDA + обоснование feature engineering (Titanic)
│   └── 02_house_prices_eda.ipynb         EDA + обоснование feature engineering (House Prices)
├── scripts/
│   ├── download_data.py                  скачивание raw-данных через Kaggle API
│   ├── train_iris.py                     полный пайплайн: baseline → CV → final → метрики
│   ├── train_titanic.py                  полный пайплайн: baseline → CV → final → метрики
│   └── train_house_prices.py             полный пайплайн: baseline → CV → final → метрики
├── src/
│   ├── titanic_preprocessing.py          очистка/инженерия признаков + sklearn-пайплайны (Titanic)
│   ├── house_prices_preprocessing.py     очистка/инженерия признаков + sklearn-пайплайны (House Prices)
│   ├── iris_preprocessing.py             очистка/инженерия признаков + sklearn-пайплайн(Iris)
│   └── evaluation.py                     общие метрики/отчёты для обеих задач
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
│   └── decisiontree.ipynb                теория по деревьям решений с нуля
├── questions/
│   ├── questions_linearregression.md     вопросы по линейной регрессии
│   └── questions_logisticregression.md   вопросы по логистической регрессии
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
jupiter lab notebooks/03_iris_eda.ipynb
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

## Iris — многоклассовая классификация сортов ириса

**Baseline** (`LogisticRegression`): стандартная логистическая регрессия с дефолтными параметрами (`C=1`, L2-регуляризация) на четырёх числовых признаках (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`) без масштабирования и снижения размерности — то, что сделал бы новичок в первый день.

**Final (LR + PCA search)**: логистическая регрессия с L2-регуляризацией внутри `Pipeline` (`RobustScaler` → опционально `PCA` → модель), подбор `C` и наличия PCA через `GridSearchCV` + `StratifiedKFold` (5 фолдов, метрика accuracy). По результатам поиска PCA не улучшил качество, лучшая конфигурация — без снижения размерности.

**SVM (Linear)**: линейный метод опорных векторов на масштабированных признаках с опциональным PCA, подбор `C`.
**SVM (RBF)**: метод опорных векторов с радиально-базисным ядром для учёта нелинейных границ, подбор `C` и `gamma`.
**Decision Tree**: дерево решений с подбором глубины, минимального размера листа/узла, критерия разбиения и весов классов.

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

Лучшие гиперпараметры по кросс-валидации:
- **Final LR**: `C = 10`, `pca = passthrough` (PCA не выбран)
- **SVM Linear**: `C = 3`, `pca = passthrough`
- **SVM RBF**: `C = 10`, `gamma = 0.1`, `pca = passthrough`
- **Decision Tree**: `criterion = gini`, `max_depth = 3`, `min_samples_leaf = 1`, `min_samples_split = 2`, `class_weight = None`, `pca = passthrough`

Подробности EDA — `notebooks/02_iris_eda.ipynb`; код предобработки — `src/iris_preprocessing.py`; диагностические графики (confusion matrix, ROC-кривые OVR) — `models/iris/plots/`.

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

- `tutorials/linearregression.ipynb`, `tutorials/logisticregression.ipynb`, `tutorials/svm.ipynb` — теория методов с нуля: постановка задачи, функция потерь, аналитическое и итеративное решение, регуляризация, диагностика, метрики.
- `questions.md` — личный конспект вопросов к собеседованию по линейной регрессии (справочный материал, не туториал).
