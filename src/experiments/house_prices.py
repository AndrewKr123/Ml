"""Совместимость со старым путём; реализация: src.classical.house_prices."""
import sys
from importlib import import_module

sys.modules[__name__] = import_module("src.classical.house_prices")
