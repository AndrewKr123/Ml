# Карта реорганизации

| Ранее | Основной путь теперь | Совместимость |
|---|---|---|
| `src/*_preprocessing.py` | `src/datasets/<dataset>.py` | Старые модули — aliases |
| `src/experiments/` | `src/classical/` | Старые модули — aliases |
| `src/evaluation.py` | `src/evaluation/__init__.py` | Импорт `src.evaluation` прежний |
| `scripts/train_*.py` | `scripts/classical/train_*.py` | Обёртки CLI и импортов |
| `scripts/check_cv_stability.py` | `scripts/classical/check_cv_stability.py` | Обёртка CLI и импорта |
| `notebooks/*.ipynb` | `notebooks/eda/*.ipynb` | Содержимое и outputs сохранены |
| `tutorials/*.ipynb` | `tutorials/classical/*.ipynb` | Содержимое сохранено |
| `models/<dataset>/` | `artifacts/<dataset>/classical/legacy/` | Символические ссылки |
| `MODEL_METRICS_TRAIN_TEST.md` | `reports/MODEL_METRICS_TRAIN_TEST.md` | Символическая ссылка |
| Длинный README с прежними результатами | `reports/CLASSICAL_GUIDE.md` | Сохранён как исторический документ |

Новые классические запуски создают уникальные каталоги.
Старые команды теперь также сохраняют новые результаты в `artifacts`,
поэтому архив `models` автоматически не обновляется.
Формат сохранённых sklearn-моделей не изменялся.

Добавлены общий модуль разбиений и формат метаданных запусков.
Каталоги глубокого обучения зарезервированы только с README.
Код нейросетей, конфигурация MLP и результаты его запуска удалены.

Реорганизация не пересчитывает исторические метрики и не меняет предобработку
классических моделей.
