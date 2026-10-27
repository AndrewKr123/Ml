"""Совместимость со старым путём; реализация: src.datasets.titanic."""
import sys
from importlib import import_module

sys.modules[__name__] = import_module("src.datasets.titanic")
