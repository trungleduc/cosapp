from abc import abstractmethod
from typing import Any, Dict

import numpy
from scipy.linalg import LinAlgWarning, lu_factor, lu_solve
from scipy.sparse.linalg import splu

from cosapp.utils.json import jsonify
from cosapp.utils.options_dictionary import HasOptions
from cosapp.utils.state_io import object__getstate__


class AbstractLinearSolver(HasOptions):

    def __getstate__(self) -> Dict[str, Any]:
        """Creates a state of the object.

        The state type does NOT match type specified in
        https://docs.python.org/3/library/pickle.html#object.__getstate__
        to allow custom serialization.

        Returns
        -------
        Dict[str, Any]:
            state
        """
        return object__getstate__(self)

    def __json__(self) -> Dict[str, Any]:
        """Creates a JSONable dictionary representation of the object.

        Break circular dependencies by removing some slots from the
        state.

        Returns
        -------
        Dict[str, Any]
            The dictionary
        """
        state = self.__getstate__().copy()

        return jsonify(state)

    @abstractmethod
    def solve(self, x: numpy.ndarray) -> numpy.ndarray:
        """Solves the linear problem."""
        pass

    @property
    @abstractmethod
    def need_jacobian(self) -> bool:
        """Gets whether the implementation needs eager computation of
        a Jacobian matrix or not."""
        pass

    @abstractmethod
    def has_valid_state(self, size: int) -> bool:
        """Gets whether the implementation is in valid state or not."""
        pass

    @abstractmethod
    def eval(self, dx: numpy.ndarray) -> numpy.ndarray:
        """Evaluates a residue delta from an input delta."""
        pass


class GradientBasedLS(AbstractLinearSolver):
    @property
    @abstractmethod
    def jacobian(self) -> numpy.ndarray:
        """Gets the Jacobian matrix."""
        pass

    @jacobian.setter
    @abstractmethod
    def jacobian(self, value: numpy.ndarray) -> None:
        """Sets the Jacobian matrix."""
        pass


class DenseLUSolver(GradientBasedLS):
    """Linear solver based on dense LU decomposition."""

    def __init__(self):
        super().__init__()
        self._jac = None
        self._lu_piv = None

    @property
    def need_jacobian(self) -> bool:
        """Gets whether the implementation needs eager computation of
        a Jacobian matrix or not."""
        return True

    def setup(self, jac: numpy.ndarray) -> None:
        """Performs setup of the linear solver from the Jacobian matrix."""
        lu, piv = lu_factor(jac, check_finite=True)
        min_diag = numpy.abs(lu.diagonal()).min()
        if min_diag < 1e-14:
            raise LinAlgWarning(
                f"Quasi-singular Jacobian matrix; min diag element of U matrix is {min_diag}"
            )
        self._jac = jac
        self._lu_piv = lu, piv

    @property
    def jacobian(self) -> numpy.ndarray:
        """Gets the Jacobian matrix."""
        return self._jac

    @jacobian.setter
    def jacobian(self, value: numpy.ndarray) -> None:
        """Sets the Jacobian matrix."""
        self._jac = value
        self.setup(self._jac)

    def has_valid_state(self, size: int) -> bool:
        """Gets whether the implementation is in valid state or not."""
        return self._lu_piv is not None and self._lu_piv[0].shape == (size, size)

    def solve(self, x: numpy.ndarray) -> numpy.ndarray:
        """Solves the linear problem."""
        return -lu_solve(self._lu_piv, x)

    def eval(self, dx: numpy.ndarray) -> numpy.ndarray:
        """Evaluates a residue delta from an input delta."""
        return self._jac.dot(dx)


class SparseLUSolver(AbstractLinearSolver):
    """Linear solver based on sparse LU decomposition."""

    def __init__(self):
        super().__init__()
        self._splu = None
        self._jac = None

    def setup(self, jac: numpy.ndarray) -> None:
        """Performs setup of the linear solver from the Jacobian matrix."""
        self._splu = splu(jac)

    @property
    def need_jacobian(self) -> bool:
        """Gets whether the implementation needs eager computation of
        a Jacobian matrix or not."""
        return True

    @property
    def jacobian(self) -> numpy.ndarray:
        """Gets the Jacobian matrix."""
        return self._jac

    @jacobian.setter
    def jacobian(self, value: numpy.ndarray) -> None:
        """Sets the Jacobian matrix."""
        self._jac = value
        self.setup(self._jac)

    def has_valid_state(self, size: int) -> bool:
        """Gets whether the implementation is in valid state or not."""
        return self._splu is not None and self._splu.shape == (size, size)

    def solve(self, x: numpy.ndarray) -> numpy.ndarray:
        """Solves the linear problem."""
        return -self._splu.solve(x)

    def eval(self, dx: numpy.ndarray) -> numpy.ndarray:
        """Evaluates a residue delta from an input delta."""
        L, U = self._splu.L, self._splu.U
        return L.dot(U).dot(dx)
