from __future__ import annotations

import logging
from typing import (
    Any, Callable, Dict, Optional,
    Sequence, Tuple, TypeVar, Union,
)
import numpy
from scipy import optimize

from cosapp.core.numerics.basics import SolverResults
from cosapp.core.numerics.enum import NonLinearMethods

from .non_linear_solver import (
    AbstractNonLinearSolver,
    NewtonRaphsonSolver,
    ScipyRootSolver,
)

logger = logging.getLogger(__name__)


ConcreteSolver = TypeVar("ConcreteSolver", bound=AbstractNonLinearSolver)
RootFunction = Callable[[Sequence[float], Any], numpy.ndarray]


def root(
    fun: RootFunction,
    x0: Sequence[float],
    args: Tuple[Any] = tuple(),
    method: NonLinearMethods = NonLinearMethods.POWELL,
    options: Dict[str, Any] = {},
    callback: Optional[Callable[[], None]] = None,
) -> Union[SolverResults, optimize.OptimizeResult]:

    if method == NonLinearMethods.NR:
        solver = NewtonRaphsonSolver.from_options(options)
    else:
        solver = ScipyRootSolver(method.value, **options)

    solver.update_options(options)
    return solver.solve(fun, x0, args, callback=callback)
