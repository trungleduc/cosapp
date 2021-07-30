"""
`Driver`s for `System` optimization calculation.
"""
import logging
from collections import OrderedDict
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy
import scipy.optimize

from cosapp.core.eval_str import EvalString
from cosapp.core.numerics.basics import SolverResults
from cosapp.drivers.abstractsolver import AbstractSolver
from cosapp.drivers.optionaldriver import OptionalDriver
from cosapp.drivers.runoptim import RunOptim
from cosapp.recorders.recorder import BaseRecorder
from cosapp.utils.options_dictionary import OptionsDictionary

logger = logging.getLogger(__name__)


# TODO
# [ ] Pull unknowns from Systems
class Optimizer(AbstractSolver):
    """Driver running an optimization problem on its `System` owner.

    In general, the optimization problems are of the form::

        minimize f(x) subject to

        g_i(x) >= 0,  i = 1,...,m
        h_j(x)  = 0,  j = 1,...,p

    where ``x`` is a vector of one or more variables. ``g_i(x)`` are the inequality constraints.
    ``h_j(x)`` are the equality constrains.

    Optionally, the lower and upper bounds for each element in x can also be specified.

    Parameters
    ----------
    name : str
        Name of the driver
    owner : System, optional
        :py:class:`~cosapp.systems.system.System` to which this driver belong; default None
    **kwargs : Any
        Keyword arguments will be used to set driver options

    Attributes
    ----------
    name : str
        Name of the driver
    parent : Driver, optional
        Top driver containing this driver; default None.
    owner : System
        :py:class:`~cosapp.systems.system.System` to which this driver belong.
    children : OrderedDict[Driver]
        Drivers belonging to this one.

    options : OptionsDictionary
      |  Options for the current driver
      |  **verbose** : int, {0, 1}
      |    Verbosity level of the driver; default 0 (i.e. minimal information)
      |  **eps** : float, [1.5e-8, 1.]
      |    Step size used for numerical approximation of the Jacobian; default 1.5e-8
      |  **ftol** : float, [1.5e-8, 1.]
      |    Iteration termination criteria (f^k - f^{k+1})/max{\|f^k\|,\|f^{k+1}\|,1} <= ftol;
      |    default 1e-6
      |  **max_iter** : int, [1, [
      |    Maximum number of iterations; default 100

    solution :  # TODO

    Notes
    -----
    This optimizer is a wrapper around ``scipy.optimize.minimize`` function. For more details please
    refer to:
    https://docs.scipy.org/doc/scipy-1.0.0/reference/generated/scipy.optimize.minimize.html

    """

    def __init__(self,
        name: str,
        owner: "Optional[cosapp.systems.System]" = None,
        **kwargs
    ) -> None:
        """Initialize a driver

        Parameters
        ----------
        name: str, optional
            Name of the `Module`
        owner : System, optional
            :py:class:`~cosapp.systems.system.System` to which this driver belong; default None
        **kwargs : Dict[str, Any]
            Optional keywords arguments
        """
        super().__init__(name, owner, **kwargs)

        self.options.declare(
            'method',
            None,
            values=['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP',
                    'trust-constr', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov'],
            allow_none=True,
            desc="Type of solver. If not given, chosen to be one of 'BFGS', 'L-BFGS-B', 'SLSQP', depending if "
                "the problem has constraints or bounds.")
        self.options.declare('eps', 1.5e-08, dtype=float, lower=1.5e-8, upper=1.,
            desc='Step size used for numerical approximation of the Jacobian.')
        self.options.declare('ftol', 1.0e-6, dtype=float, lower=1.5e-8, upper=1.,
            desc='The iteration stops when (f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol.')
        self.options.declare('maxiter', 100, dtype=int, lower=1, desc='Maximum number of iterations.')
        self.options.declare('monitor', False, dtype=bool, allow_none=False,
            desc='Defines if intermediate system state should be recorded.')

        self._filter_options(kwargs, aliases={'tol': 'ftol', 'max_iter': 'maxiter'})

        self.add_child(RunOptim(self._default_driver_name))

    def _fun_wrapper(self, expression: EvalString) -> Callable[[numpy.ndarray], float]:
        """Wrapper around objective and constraint expression to propagate the x values in the
            `System` owner.

        Parameters
        ----------
        expression : EvalString
            The expression to be evaluated.

        Returns
        -------
        Callable[[numpy.ndarray], float]
            Callable function usable by scipy.optimize.minimize
        """

        def wrapper(x: numpy.ndarray, *args) -> float:
            """Wrapper around objective and constraint function to propagate the x values in the
            `System` owner.

            Parameters
            ----------
            x : 1D-scalar numpy.ndarray
                Points at which the function needs to be evaluated
            args : Tuple
                Any additional fixed parameters needed to completely specify the function

            Returns
            -------
            Float
                Value of the function for x
            """
            # Propagate x value in the System owner, if it has changed
            counter = 0
            need_update = False
            for unknown in self.problem.unknowns.values():
                if unknown.mask is None:
                    unknown.set_default_value(x[counter])
                    counter += 1
                else:
                    n = numpy.count_nonzero(unknown.mask)
                    unknown.set_default_value(x[counter:counter + n])
                    counter += n

                # Set the variable to the new x
                if not numpy.array_equal(unknown.value, unknown.default_value):
                    unknown.set_to_default()
                    need_update = True

            if need_update:
                for child in self.exec_order:
                    self.children[child].run_once()

            return expression.eval()

        return wrapper

    def _fresidues(self, x: numpy.ndarray, update_residues_ref: bool = True) -> float:
        """
        Method used by the solver to take free variables values as input and values of the objective function (after
        running the System).

        Parameters
        ----------
        x : numpy.ndarray
            The list of values to set to the free variables of the `System`
        update_residues_ref : bool
            Request residues to update their reference

        Returns
        -------
        float
            Objective function value
        """
        return super()._fresidues(x, update_residues_ref)[0]

    def set_iteratives(self, x: Sequence[float]) -> None:
        x = numpy.asarray(x)
        counter = 0
        for unknown in self.problem.unknowns.values():
            if unknown.mask is None:
                unknown.set_default_value(x[counter])
                counter += 1
            else:
                n = numpy.count_nonzero(unknown.mask)
                unknown.set_default_value(x[counter : counter + n])
                counter += n
            # Set variable to new x
            if not numpy.array_equal(unknown.value, unknown.default_value):
                unknown.set_to_default()

    def resolution_method(self,
        fresidues: Callable[[Sequence[float]], float],
        x0: numpy.ndarray,
        args: Tuple[Union[float, str], bool] = (),
        options: Optional[OptionsDictionary] = None,
        bounds = None,
        constraints = None,
    ) -> SolverResults:
        """Function call to cancel the residues.

        Parameters
        ----------
        fresidues : Callable[[Sequence[float], Union[float, str]], float]
            Residues function taking two parameters (evaluation vector, time/ref) and returning the residues
        x0 : numpy.ndarray
            The initial values vector to converge to the solution
        args : Tuple[Union[float, str], bool], optional
            A tuple of additional argument for fresidues starting with the time/ref parameter
        options : OptionsDictionary, optional
            Options for the numerical resolution method

        Returns
        -------
        SolverResults
            Solution container
        """
        sub_options = {
            'ftol': options['ftol'],
            # TODO ftol is not understood by unconstrained solver. So set `gtol` but warning about `ftol` is emitted
            'gtol': options['ftol'],
            'eps': options['eps'],
            'maxiter': options['maxiter'],
            'disp': bool(options['verbose'])
        }

        output = scipy.optimize.minimize(
            fresidues, x0,
            args=args,
            method=options['method'],
            bounds=bounds,
            constraints=constraints,
            options=sub_options,
        )
        return output

    def _precompute(self):
        """List unknowns and gather initial values."""
        super()._precompute()
        OptionalDriver.set_inhibited(True)

        for child in self.children.values():
            if isinstance(child, RunOptim):
                self.problem.extend(child.get_problem(), copy=False)
                self.initial_values = numpy.append(self.initial_values, child.get_init(self.force_init))

    def compute(self) -> None:
        """Execute the optimization."""
        self.status = ''
        self.error_code = '0'

        # Check that there is only one objective
        if self.problem.residues_vector.size != 1:
            self.status = 'ERROR'
            self.error_code = '9'
            raise ArithmeticError(
                "Optimizer can only target a unique cost value; got {} values.".format(
                    self.problem.residues_vector.size)
            )

        # Gather and simplify the bounds
        limits = self._get_solver_limits()
        unique_lower = numpy.unique(limits['lower_bound'])
        unique_upper = numpy.unique(limits['upper_bound'])
        def get_bounds(lower, upper) -> tuple:
            return (
                None if lower == -numpy.inf else lower,
                None if upper ==  numpy.inf else upper
            )
        if unique_lower.size == 1 and unique_upper.size == 1:
            bounds = get_bounds(unique_lower[0], unique_upper[0])
            if bounds == (None, None):
                bounds = None
        else:
            bounds = tuple()

        if bounds is not None:
            bounds = [
                get_bounds(lower, upper)
                for lower, upper in zip(limits['lower_bound'], limits['upper_bound'])
            ]

        # Create constraint
        constraints = []
        # TODO The following is really ugly. Constraints should be merged from all children through the
        #  MathematicalProblem extension.
        for c in list(self.children.values())[0].constraints:
            constraints.append({
                'type': c['type'],
                'fun': self._fun_wrapper(EvalString(c['formula'], self.owner))
            })
        constraints = tuple(constraints)

        if len(self.initial_values) > 0:
            if not self.options['monitor']:
                BaseRecorder.paused = True

            results = self.resolution_method(
                self._fresidues,
                self.initial_values,
                args = (False,),
                bounds = bounds,
                constraints = constraints,
                options = self.options,
            )

            if not results.success:
                self.status = 'ERROR'
                self.error_code = '9'
                self.solution = {}
                logger.error(f"The solver failed: {results.message}")
            else:
                self.solution = dict([(key, unknown.default_value) for key, unknown in self.problem.unknowns.items()])
                self._print_solution()

            # Call to record the results
            if not self.options['monitor']:
                BaseRecorder.paused = False
                for child in self.children.values():
                    child.run_once()

        else:
            logger.warning('No design variable has been specified for the optimization.')

    def _postcompute(self) -> None:
        """Undo pull inputs and reset iteratives sets."""
        OptionalDriver.set_inhibited(False)
        super()._postcompute()

    def _print_solution(self) -> None:  # TODO better returning a string
        """Print the solution in the log."""
        if self.options['verbose']:
            logger.info(f"Objective function: {self.problem.residues_vector[0]:.5g}")
            logger.info(f"Parameters [{len(self.solution)}]: ")
            for name, value in self.solution.items():
                logger.info(f"   # {name}: {value}")
            constraints = list(self.children.values())[0].constraints
            if constraints:
                logger.info(f"Constraints [{len(constraints)}]: ")
                for constraint in constraints:
                    expr = constraint['formula']
                    value = EvalString(expr, self.owner).eval()
                    logger.info(f"   # {expr}: {value}")
