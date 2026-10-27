"""Совместимая команда; реализация: scripts.classical.train_house_prices."""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    from runpy import run_module
    run_module("scripts.classical.train_house_prices", run_name="__main__")
else:
    from importlib import import_module
    sys.modules[__name__] = import_module("scripts.classical.train_house_prices")
