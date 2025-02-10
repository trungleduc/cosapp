from enum import Enum
from typing import Any, Dict

from cosapp.utils.state_io import object__getstate__

# TODO
# class SolverStep(Enum):
#     """Step in progress within the numerical solver.
#
#     INIT : Initialization phase
#     JACOBIAN : Jacobian matrix building phase
#     ITERATION : Convergence iteration
#     """
#     INIT = 'init'
#     JACOBIAN = 'jacobian'
#     ITERATION = 'iteration'


class NonLinearMethods(Enum):
    """Enumeration of non-linear algorithm available.

    POWELL : Modified Powell method using MINPACK’s hybrd and hybrj routines
    BROYDEN_GOOD : Broyden’s first Jacobian approximation. This method is also known as “Broyden’s good method”.
    NR : Simple Newton-Raphson
    """

    POWELL = "hybr"
    BROYDEN_GOOD = "broyden1"
    NR = "cosapp"

    def __json__(self) -> Dict[str, Any]:
        """Creates a state of the object.

        The state type depend on the object, see
        https://docs.python.org/3/library/pickle.html#object.__getstate__
        for further details.

        Returns
        -------
        Dict[str, Any]:
            state
        """
        return object__getstate__(self)
