"""
Core package of CoSApp.
"""
from cosapp.core._version import __version__

from cosapp.core.config import CoSAppConfiguration
from cosapp.core.numerics.basics import MathematicalProblem

__all__ = [
    "MathematicalProblem",
]
