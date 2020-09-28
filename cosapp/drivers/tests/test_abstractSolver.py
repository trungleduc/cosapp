from typing import Callable, Sequence, Union, Tuple, Optional

import numpy as np

from cosapp.systems import System
from cosapp.core.numerics.basics import SolverResults
from cosapp.drivers.abstractsolver import AbstractSolver
from cosapp.utils.options_dictionary import OptionsDictionary


class FailureSolver(AbstractSolver):
    def resolution_method(
        self,
        fresidues: Callable[[Sequence[float], Union[float, str]], np.ndarray],
        x0: Sequence[float],
        args: Tuple[Union[float, str], ...] = (),
        options: Optional[OptionsDictionary] = None,
    ):
        r = SolverResults()
        fresidues(x0, *args)
        r.x = x0
        r.success = False
        r.message = "Dummy abstract solver is not solving anything"
        return r


class SuccessSolver(AbstractSolver):
    def resolution_method(
        self,
        fresidues: Callable[[Sequence[float], Union[float, str]], np.ndarray],
        x0: Sequence[float],
        args: Tuple[Union[float, str], ...] = (),
        options: Optional[OptionsDictionary] = None,
    ):
        r = SolverResults()
        fresidues(x0, *args)
        r.x = x0
        r.success = True
        r.message = "Dummy abstract solver is not solving anything"
        return r


def test_AbstractSolver_setup():
    d = FailureSolver("dummy")

    assert len(d.children) == 0
    assert d.problem is None
    assert isinstance(d.initial_values, np.ndarray)
    assert len(d.initial_values) == 0
    assert isinstance(d.solution, dict)
    assert len(d.solution) == 0
    assert d._default_driver_name == "runner"
