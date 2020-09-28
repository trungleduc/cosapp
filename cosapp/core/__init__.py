"""
Core package of CoSApp.
"""
from cosapp.core._version import __version__

from cosapp.core.numerics.basics import MathematicalProblem
from cosapp.core.numerics.root import root
from cosapp.core.config import CoSAppConfiguration

from cosapp.core.module import Module

__all__ = ["MathematicalProblem", "root"]
