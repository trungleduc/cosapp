import numpy
import copy
from typing import Any, Dict, List, Optional, Union, Optional
from dataclasses import dataclass, field

from cosapp.core import MathematicalProblem
from cosapp.core.numerics.boundary import Boundary
from cosapp.drivers.driver import Driver
from cosapp.systems.system import System, SystemConnector
from cosapp.ports.port import BasePort
from cosapp.utils.state_io import object__getstate__

import logging
logger = logging.getLogger(__name__)


def default_array_factory(shape=0, dtype=float):
    """Default factory for dataclass fields."""
    def factory():
        return numpy.empty(shape, dtype=dtype)
    return factory


@dataclass
class SolverResults:
    """Data class for solver solution

    Attributes
    ----------
    - x [numpy.ndarray[float]]:
        Solution vector.
    - r [numpy.ndarray[float]]:
        Residue vector.
    - success [bool]:
        Whether or not the solver exited successfully.
    - message [str]:
        Description of the cause of the termination.
    - tol [float, optional]:
        Tolerance level; `NaN` if not available.
    - n_iter [int]:
        Number of iterations.
    """
    x: numpy.ndarray = field(default_factory=default_array_factory())
    r: numpy.ndarray = field(default_factory=default_array_factory())
    success: bool = False
    message: str = ""
    tol: float = numpy.nan
    n_iter: int = 0

    def __getstate__(self) -> Union[Dict[str, Any], tuple[Optional[Dict[str, Any]], Dict[str, Any]]]:
        """Creates a state of the object.

        The state type depend on the object, see
        https://docs.python.org/3/library/pickle.html#object.__getstate__
        for further details.
        
        Returns
        -------
        Union[Dict[str, Any], tuple[Optional[Dict[str, Any]], Dict[str, Any]]]:
            state
        """
        return object__getstate__(self)

    def __json__(self) -> Dict[str, Any]:
        """Creates a JSONable dictionary representation of the object.

        Returns
        -------
        Dict[str, Any]
            The dictionary
        """
        return self.__getstate__().copy()


class FixedPointSolver(Driver):
    """Driver attempting to solve the internal cyclic dependencies
    of its owner `System` by iteratively running it.

    Parameters
    ----------
    name : str
        Name of the driver
    owner : System,
        :py:class:`~cosapp.systems.system.System` to which driver belongs.
    **options : Any
        Keyword arguments used to set driver options
    """

    __slots__ = ('initial_values', 'problem', 'results', '_loop_connectors')

    def __init__(
        self,
        name='solver',
        owner: Optional[System]=None,
        **options,
    ) -> None:
        """Initialize driver

        Parameters
        ----------
        name: str, optional
            Name of the `Driver` (default: 'solver').
        owner: `System`. optional
            :py:class:`~cosapp.systems.system.System` to which the driver belongs (defaults: `None`).
        **options:
            Additional keywords arguments forwarded to base class.
        """
        super().__init__(name, owner, **options)
        
        self.problem: MathematicalProblem = None
        self.results = SolverResults()
        self.initial_values: Dict[str, Boundary] = dict()  # Initial guess for the iteratives
        self._loop_connectors: List[SystemConnector] = []

    def _declare_options(self) -> None:
        super()._declare_options()
        self.options.declare(
            'tol', 1e-6, dtype=float, allow_none=False,
            desc='Absolute tolerance (in max-norm) for the residual.'
        )
        self.options.declare(
            'max_iter', 50, dtype=int, lower=1,
            desc='The maximum number of iterations.',
        )
        self.options.declare(
            'factor', 1.0, dtype=float, lower=1e-3,
            desc='Relaxation factor; applies: next value = factor * current + (1 - factor) * previous.'
        )
        self.options.declare(
            'history', False, dtype=bool,
            desc='Should the recorder (if any) capture all iterations, or just the last one?'
        )
        self.options.declare(
            'force_init', False, dtype=bool,
            desc='Whether or not initial conditions should be applied at the beginning of the resolution.'
        )

    def set_init(self, modifications: Dict[str, Any]) -> None:
        """Define initial values for one or more variables.

        The variable can be contextual `child1.port2.var`. The only rule is that it should belong to
        the owner `System` of this driver or any of its descendants.

        Parameters
        ----------
        modifications : dict[str, Any]
            Dictionary of (variable name, value)

        Examples
        --------
        >>> driver.set_init({'myvar': 42, 'foo.dummy': 'banana'})
        """
        if self.owner is None:
            raise AttributeError(
                f"Driver {self.name!r} must be attached to a System to be assigned initial values."
            )

        if not isinstance(modifications, dict):
            raise TypeError(
                "Initial values must be specified through a dictionary of the kind {varname: value}."
            )

        for variable, value in modifications.items():
            boundary = Boundary(self.owner, variable, default=value, inputs_only=False)

            # Check if boundary.name already exists
            actual = self.initial_values.setdefault(boundary.name, boundary)

            if actual is not boundary:
                # Update already existing boundary with new default value and mask
                actual.update_default_value(boundary.default_value, boundary.mask)

    def get_init(self) -> Dict[str, Any]:
        """Get the initial values used by the solver.

        Returns
        -------
        dict[str, Any]
            Dictionary of initial values, referenced by variable names.
        """
        init = {
            varname: boundary.value
            for varname, boundary in self.initial_values.items()
        }
        return init

    def apply_init(self) -> None:
        """Apply initial values in owner system."""
        for boundary in self.initial_values.values():
            boundary.set_to_default()
    
    def setup_run(self):
        """Method called once before starting any simulation."""
        super().setup_run()
        
        owner = self.owner
        owner.open_loops()
        self.problem = owner.assembled_problem()

        self._loop_connectors = loop_connectors = []
        is_open = lambda connector: not connector.is_active
        for system in owner.tree():
            loop_connectors.extend(filter(is_open, system.all_connectors()))

        if self.options['force_init']:
            self.apply_init()

    def run_once(self) -> None:
        """Run solver once, assuming driver has already been initialized.
        """
        with self.log_context(" - run_once"):
            if self.is_active():
                self._precompute()

                logger.debug(f"Call {self.name}.compute_before()")
                self.compute_before()

                # Sub-drivers are executed at each iteration in `compute`,
                # so the child loop before `self.compute()` is omitted.
                logger.debug(f"Call {self.name}.compute()")
                self._compute_calls += 1
                self.compute()

                self._postcompute()
                self.computed.emit()
            
            else:
                logger.debug(f"Skip {self.name} execution - Inactive")

    def compute(self) -> None:
        """Execute drivers on all child `System` belonging to the driver `System` owner.
        """
        owner: System = self.owner
        subdrivers = self.children.values()

        if subdrivers:
            def update_system() -> None:
                for subdriver in subdrivers:
                    logger.debug(f"Call {subdriver.name}.run_once()")
                    subdriver.run_once()
        else:
            def update_system() -> None:
                owner.run_children_drivers()

        if (recorder := self._recorder):
            def record_state(ref: str, *args, **kwargs):
                recorder.record_state(ref, *args, **kwargs)
        else:
            record_state = lambda *args, **kwargs: None

        problem = self.problem
        results = self.results

        if problem.is_empty():
            record_history = False
            update_system()
            results.message = "No iterations were necessary"
            results.success = True
            results.n_iter = 0

        else:
            loop_connectors = self._loop_connectors
            # Deactivate all loop connectors for the first iteration
            for connector in loop_connectors:
                connector.deactivate()

            tol = self.options['tol']
            factor = self.options['factor']
            max_iter = self.options['max_iter']
            record_history = self.options['history']

            residues = problem.residues.values()
            converged = False

            if factor == 1.0:
                def update_unknowns():
                    for connector in loop_connectors:
                        connector.activate()
                        connector.transfer()

            else:
                def update_unknowns():
                    for connector in loop_connectors:
                        sink: BasePort = connector.sink
                        previous_values = {
                            varname: copy.copy(getattr(sink, varname))
                            for varname in connector.sink_variables()
                        }
                        connector.activate()
                        connector.transfer()
                        # Apply relaxation
                        for varname, previous in previous_values.items():
                            current = getattr(sink, varname)
                            setattr(sink, varname, factor * current + (1.0 - factor) * previous)
                        connector.deactivate()

            try:
                for i in range(max_iter):
                    update_system()
                    for residue in residues:
                        residue.update()
                    r = problem.residue_vector()
                    r_norm = numpy.linalg.norm(r, numpy.inf)
                    logger.debug(f"iteration #{i},\t{r = }")
                    if record_history:
                        record_state(f"{self.name} (iter #{i})")
                    update_unknowns()
                    converged = bool(r_norm < tol)
                    if converged:
                        break

            except:
                pass  # silent fail, for now
            
            finally:
                if converged:
                    self.status = "Converged"
                    self.error_code = "0"
                    message = f"Fixed-point resolution has converged in {i} iterations ({r_norm = :.2e})"
                    logger.info(message)
                else:
                    self.status = "Not converged"
                    self.error_code = "1"
                    message = f"Fixed-point resolution has not converged in {max_iter} iterations"
                    logger.warn(message)
                # Store solver results
                results.success = converged
                results.message = message
                results.n_iter = i
                results.tol = tol
                results.x = problem.unknown_vector()
                results.r = r

        if not record_history:
            record_state(self.name, self.status, self.error_code)

    def _postcompute(self) -> None:
        self.owner.close_loops()
        super()._postcompute()

    def is_standalone(self) -> bool:
        """Is this Driver able to solve a system?

        Returns
        -------
        bool
            Ability to solve a system or not.
        """
        return True
