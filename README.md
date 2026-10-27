# Классическое и глубокое машинное обучение

Учебный проект классического ML со структурой для будущего глубокого обучения.
Titanic — бинарная классификация, Iris — многоклассовая классификация,
House Prices — регрессия. Для всех трёх задач сохранены классические модели,
включая KNN, SVM, деревья и ансамбли. Нейросетевых реализаций пока нет.

## Структура

```text
src/
├── datasets/          загрузка данных, признаки и разбиения
├── classical/         обучение sklearn-моделей и ансамблей
├── deep/              зарезервирован, только README
├── evaluation/        общие метрики классификации и регрессии
├── reporting/         сохранение результатов и метаданных запусков
└── paths.py           пути относительно корня проекта
scripts/
├── classical/         запуски классических моделей и проверка CV
└── deep/              зарезервирован, только README
configs/deep/          место для будущих конфигураций
notebooks/eda/         исследование трёх датасетов
notebooks/experiments/ место для интерактивных сравнений
tutorials/classical/ теория существующих методов
tutorials/deep/      теоретические конспекты по глубокому обучению
artifacts/             отдельный каталог каждого запуска
reports/               сводные метрики, историческое руководство и миграция
tests/                проверки обучения, разбиений и совместимости
```

`data/<dataset>/{raw,processed}` и `questions/` сохраняют прежнее назначение.
Старые модули `src/*_preprocessing.py`, `src/experiments/` и команды
`scripts/train_*.py` — совместимые обёртки. Новый код следует писать в новых пакетах.

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .

# Необязательно для классических моделей; PyTorch только для CPU:
python -m pip install -r requirements-deep-cpu.txt
```

Если окружение создано через `uv` и не содержит pip:

```bash
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python -e .
uv pip install --python .venv/bin/python -r requirements-deep-cpu.txt
```

PyTorch не импортируется при обычном запуске классических моделей.

## Классическое обучение

```bash
python scripts/classical/train_iris.py --knn-only
python scripts/classical/train_titanic.py --random-forest-only
python scripts/classical/train_house_prices.py --knn-only

# Все классические модели выбранной задачи (полный CV может быть долгим):
python scripts/classical/train_iris.py
```

У всех скриптов доступны `--test-size` (0.2) и `--random-state` (42).
Прежняя команда `python scripts/train_iris.py --knn-only` также работает.
Каждый запуск создаёт новый каталог `artifacts/<dataset>/<model>/<run_id>/`.
Для полного набора моделей имя каталога — `classical_suite`.
Состав файлов внутри сохранён: модели joblib, метрики, прогнозы и графики.
Полный запуск по-прежнему обновляет обработанные CSV в `data/`.

Исторические модели расположены в `artifacts/<dataset>/classical/legacy/`.
`models/<dataset>` — ссылки на эти каталоги: новые эксперименты их не перезаписывают.
Проверка устойчивости `python scripts/classical/check_cv_stability.py`
по-прежнему работает с историческими моделями и сохраняет результаты рядом с ними.

## Глубокое обучение

PyTorch для CPU установлен отдельно. Каталоги для будущего глубокого
обучения зарезервированы; кода нейросетей, конфигураций и результатов пока нет.
В `tutorials/deep/` добавлен [конспект по backpropagation](tutorials/deep/backpropagation.ipynb)
с формулами и ручным примером, без исполняемого кода.

## Общий формат запуска

Классические эксперименты сохраняют:

- `config.json` — параметры эксперимента;
- `split.json` — исходные индексы строк в разбиениях;
- `run.json` — версии библиотек, идентификатор запуска, статус;
- `run_metrics.json` — единая структура `models → имя → train/validation/test/cv`
  и время обучения с оценкой/сохранением результатов.

Идентификатор содержит время UTC и случайный суффикс. Запуски не перезаписываются.
Для старых классических функций, не вычисляющих train-метрики, поле `train`
равно `null`; отсутствующие validation/CV тоже обозначаются `null`.
Оригинальные форматы `metrics.json` сохранены для совместимости.

## Протокол оценки

Сводные исторические результаты находятся в
[reports/MODEL_METRICS_TRAIN_TEST.md](reports/MODEL_METRICS_TRAIN_TEST.md).
Корневой файл с прежним именем ссылается на этот отчёт.

Перенос каталогов не меняет историческую методику классических экспериментов:
часть импутации Titanic/House Prices и выбор логарифмируемых признаков
по-прежнему выполняются до внутренней CV. Это отмечено как `legacy-v1`
в конфигурациях новых классических запусков. Внешний hold-out независим.
Исправление внутренней изоляции предобработки потребует отдельного пересчёта
результатов; исторические числа нельзя смешивать с новым протоколом.

## Учебные материалы и проверки

```bash
jupyter lab notebooks/eda/03_iris_eda.ipynb
jupyter lab tutorials/classical/knn.ipynb
python -m unittest discover -s tests -v
```

Подробности прежних моделей и результаты сохранены в
[историческом руководстве](reports/CLASSICAL_GUIDE.md).
Карта переноса находится в [reports/PROJECT_MIGRATION.md](reports/PROJECT_MIGRATION.md).
Реализация нейросетевых задач отложена до отдельного запроса.
