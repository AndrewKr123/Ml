"""Скачивает сырые датасеты Titanic и House Prices с Kaggle через Kaggle API.

Подготовка (один раз):
    1. Зайти на https://www.kaggle.com/settings -> API -> Create New Token —
       скачается файл kaggle.json.
    2. Положить его в ~/.kaggle/kaggle.json и выставить права: chmod 600 ~/.kaggle/kaggle.json
    3. Принять правила соревнований (без этого API вернёт 403):
       https://www.kaggle.com/c/titanic/rules
       https://www.kaggle.com/c/house-prices-advanced-regression-techniques/rules

Запуск:
    python scripts/download_data.py            # скачает только то, чего ещё нет
    python scripts/download_data.py --force     # перекачает всё заново

Скрипт идемпотентный: raw-файлы уже лежат в data/*/raw в этом репозитории,
повторный запуск без --force ничего не скачивает и не портит.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

COMPETITIONS = {
    "titanic": {
        "slug": "titanic",
        "dest": Path("data/titanic/raw"),
        "expected_files": ["train.csv", "test.csv", "gender_submission.csv"],
    },
    "house_prices": {
        "slug": "house-prices-advanced-regression-techniques",
        "dest": Path("data/house_prices/raw"),
        "expected_files": ["train.csv", "test.csv", "data_description.txt", "sample_submission.csv"],
    },
}


def already_downloaded(config: dict) -> bool:
    return all((config["dest"] / name).exists() for name in config["expected_files"])


def download_competition(config: dict, force: bool) -> None:
    slug, dest = config["slug"], config["dest"]

    if already_downloaded(config) and not force:
        print(f"[{slug}] файлы уже на месте в {dest}/, пропускаю (--force для перекачивания).")
        return

    # Импорт внутри функции: kaggle требует валидный kaggle.json уже при импорте модуля,
    # так что download_data.py --help должен работать даже без настроенных credentials.
    from kaggle.api.kaggle_api_extended import KaggleApi

    dest.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()

    print(f"[{slug}] скачиваю в {dest}/ ...")
    api.competition_download_files(slug, path=str(dest))

    zip_path = dest / f"{slug}.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(dest)
        zip_path.unlink()

    print(f"[{slug}] готово.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="перекачать датасеты, даже если они уже есть")
    args = parser.parse_args()

    for config in COMPETITIONS.values():
        download_competition(config, force=args.force)


if __name__ == "__main__":
    main()
