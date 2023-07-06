import copy
import numpy
import abc
import csv
from io import StringIO
from numbers import Number
from typing import (
    Any, Callable, Dict, List, Optional,
    Sequence, Tuple, Union, Iterable,
    TypeVar,
)

from cosapp.core.numerics.basics import MathematicalProblem, SolverResults
from cosapp.core.numerics.enum import NonLinearMethods
from cosapp.core.numerics.root import root
from cosapp.drivers.driver import Driver
from cosapp.drivers.abstractsolver import AbstractSolver
from cosapp.drivers.runsinglecase import RunSingleCase
from cosapp.drivers.utils import DesignProblemHandler
from cosapp.utils.helpers import check_arg
from cosapp.utils.logging import LogFormat, LogLevel

import logging
logger = logging.getLogger(__name__)

AnyDriver = TypeVar("AnyDriver", bound=Driver)


class BaseSolverBuilder(abc.ABC):
    """Base interface for numerical system building strategy used by `NonLinearSolver`.
    Implementations will differ for single- or multi-point design.
    """
    def __init__(self, solver: AbstractSolver):
        self.solver = solver
        self.problem = MathematicalProblem(solver.name, solver.owner)
        self.initial_values = numpy.empty(0)
        self.is_design_unknown: Dict[str, bool] = dict()

    @abc.abstractmethod
    def build_system(self) -> None:
        pass

    @abc.abstractmethod
    def update_residues(self) -> None:
        pass

    def update_unknowns(self, x: numpy.ndarray) -> None:
        counter = 0
        for name, unknown in self.problem.unknowns.items():
            if unknown.mask is None:
                unknown.set_default_value(x[counter])
                counter += 1
            else:
                n = numpy.count_nonzero(unknown.mask)
                unknown.set_default_value(x[counter : counter + n])
                counter += n
            # Update design unknowns
            if self.is_design_unknown[name]:
                unknown.set_to_default()


class NonLinearSolver(AbstractSolver):
    """Solve mathematical problem with algebraic variables.

    Attributes
    ----------
    compute_jacobian : bool
        Should the Jacobian matrix be computed? ; default True
    jac_lup : ndarray, optional
        LU decomposition of latest Jacobian matrix (if available, None otherwise)
    jac : ndarray, optional
        Latest Jacobian matrix computed (if available, None otherwise)
    """

    __slots__ = (
        '__method', '__option_aliases', '__trace', '__results',
        '__design_unknowns', 'compute_jacobian', 'jac_lup', 'jac',
        '__builder',
    )

    def __init__(self, 
        name: str, 
        owner: Optional["cosapp.systems.System"] = None, 
        method: Union[NonLinearMethods, str] = NonLinearMethods.NR, 
        **kwargs
    ) -> None:
        """Initialize driver

        Parameters
        ----------
        name: str, optional
            Name of the `Driver`.
        owner: System, optional
            :py:class:`~cosapp.systems.system.System` to which this driver belong; defaults to `None`.
        method : Union[NonLinearMethods, str]
            Resolution method to use
        **kwargs:
            Additional keywords arguments forwarded to base class.
        """
        super().__init__(name, owner, **kwargs)

        if isinstance(method, str):
            method = NonLinearMethods[method]
        else:
            check_arg(method, 'method', NonLinearMethods)
        self.__method = method
        self.__option_aliases = dict()
        self.__set_method(method, **kwargs)
        self.__trace: List[Dict[str, Any]] = list()
        self.__results: SolverResults = None
        self.__builder: BaseSolverBuilder = None

        self.compute_jacobian = True  # type: bool
            # desc='Should the Jacobian matrix be computed?'

        self.jac_lup = (None, None)  # type: Tuple[Optional[numpy.ndarray], Optional[numpy.ndarray]]
            # desc='LU decomposition of latest Jacobian matrix (if available).'
        self.jac = None  # type: Optional[numpy.ndarray]
            # desc='Latest computed Jacobian matrix (if available).'

    def reset_problem(self) -> None:
        """Reset mathematical problem"""
        super().reset_problem()
        self.__design_unknowns = dict()

    @property
    def method(self) -> NonLinearMethods:
        """NonLinearMethods : Selected solver algorithm."""
        return self.__method

    @property
    def results(self) -> SolverResults:
        """SolverResults: structure containing solver results,
        together with additional detail.
        """
        return self.__results

    def add_child(self, child: AnyDriver, execution_index: Optional[int]=None, desc="") -> AnyDriver:
        """Add a child `Driver` to the current `Driver`.

        When adding a child `Driver`, it is possible to specified its position in the execution order.

        Child `Port`, `inwards` and `outwards` can also be pulled at the parent level by providing either
        the name of the port/inward/outward or a list of them or the name mapping of the child element
        (dictionary keys) to the parent element (dictionary values).
        If the argument is not a dictionary, the name in the parent system will be the same as in the child.

        Parameters
        ----------
        - child: Driver
            `Driver` to add to the current `Driver`
        - execution_index: int, optional
            Index of the execution order list at which the `Module` should be inserted;
            default latest.
        - desc [str, optional]:
            Sub-driver description in the context of its parent driver.

        Notes
        -----
        The added child will have its owner set to match the one of the current driver.
        """
        self.compute_jacobian = True
        return super().add_child(child, execution_index, desc)

    def is_standalone(self) -> bool:
        """Is this Driver able to solve a system?

        Returns
        -------
        bool
            Ability to solve a system or not.
        """
        return True

    def resolution_method(self,
        fresidues: Callable[[Sequence[float], Union[float, str], bool], numpy.ndarray],
        x0: Sequence[float],
        args: Tuple[Union[float, str]] = (),
        options: Optional[Dict[str, Any]] = None
    ) -> SolverResults:

        if self.method == NonLinearMethods.NR:
            self.options.update(self._get_solver_limits())

        common_keys = set(self.options).intersection(options)
        this_options = dict((key, options[key]) for key in common_keys)

        if self.method == NonLinearMethods.NR:
            this_options['compute_jacobian'] = self.compute_jacobian
            this_options['jac_lup'] = self.jac_lup
            this_options['jac'] = self.jac

        results = root(fresidues, x0, args=args, method=self.method, options=this_options)

        if results.jac_lup[0] is not None:
            self.jac_lup = results.jac_lup
            self.jac = results.jac
            self.compute_jacobian = False

        return results

    def setup_run(self) -> None:
        super().setup_run()
        self._init_problem()

    def _print_solution(self) -> None:  # TODO better returning a string
        """Print the solution in the log."""
        if self.options['verbose']:
            # TODO move it in MathematicalProblem
            logger.info(f"Parameters [{len(self.solution)}]: ")
            for k, v in self.solution.items():
                logger.info(f"   # {k}: {v}")

            for name, residue in self.problem.residues.items():
                value = numpy.asarray(residue.value)
                if value.ndim > 0:
                    spacing = " " * len(name)
                    msg = f"Residues [{len(self.problem.residues)}]: \n   # ({name}"
                    msg += f"\n   #  {spacing}".join(f", {v:.5g}" for v in value)
                    msg += ")\n"
                else:
                    logger.info(f"   # ({name}, {value:.5g})")

            tol = numpy.linalg.norm(self.problem.residue_vector(), numpy.inf)
            try:
                option = self.__option_aliases['tol']  # should never fail, in principle
                target = self.options[option]
            except:
                target = 0
            logger.debug(f" # Current tolerance {tol} for target {target}")

    def _init_problem(self):
        """Initialize mathematical problem"""
        logger.debug(
            "\n".join([
                "*" * 40,
                "*", 
                "* Assemble mathematical problem",
                "*",
                "*" * 40,
            ])
        )
        run_cases = list(
            filter(
                lambda driver: isinstance(driver, RunSingleCase),
                self.children.values(),
            )
        )
        if run_cases:
            builder = MultipointSolverBuilder(self, run_cases)
        else:
            builder = StandaloneSolverBuilder(self)
        
        builder.build_system()
        self.problem = builder.problem
        self.initial_values = builder.initial_values
        self.__builder = builder
        self.touch_unknowns()
        logger.debug(
            "\n".join([
                "Mathematical problem:",
                f"{'<empty>' if self.problem.is_empty() else self.problem}",
            ])
        )

    def _fresidues(self, x: Sequence[float]) -> numpy.ndarray:
        """
        Method used by the solver to take free variables values as input and values of residues as
        output (after running the System).

        Parameters
        ----------
        x : Sequence[float]
            The list of values to set to the free variables of the `System`
        update_residues_ref : bool
            Request residues to update their reference

        Returns
        -------
        numpy.ndarray
            The list of residues of the `System`
        """
        x = numpy.asarray(x)
        logger.debug(f"Call fresidues with x = {x!r}")
        self.set_iteratives(x)

        if self.children:
            for subdriver in self.children.values():
                logger.debug(f"Call {subdriver.name}.run_once()")
                subdriver.run_once()
        else:
            self.owner.run_children_drivers()

        self.__builder.update_residues()
        residues = self.problem.residue_vector()
        logger.debug(f"Residues: {residues!r}")
        return residues

    def set_iteratives(self, x: Sequence[float]) -> None:
        x = numpy.asarray(x)
        self.__builder.update_unknowns(x)

    def compute(self) -> None:
        """Run the resolution method to find free vars values that zero out residues
        """
        # Reset status
        self.status = ''
        self.error_code = '0'

        try:
            self.problem.validate()
        except ArithmeticError:
            self.status = 'ERROR'
            self.error_code = '9'
            raise

        if len(self.initial_values) > 0:
            self.solution = {}

            # compute first order
            results = self.resolution_method(
                self._fresidues,
                self.initial_values,
                options=self.options,
            )

            self.__results = results
            self.__trace = getattr(results, "trace", list())

            if results.success:
                self.status = ''
                self.error_code = '0'
                logger.info(f"solver : {self.name}{results.message}")
            
            else:
                self.status = 'ERROR'
                self.error_code = '9'

                error_msg = f"The solver failed: {results.message}"

                error_desc = getattr(results, 'jac_errors', {})
                if error_desc:
                    if (indices := error_desc.get('unknowns', [])):
                        unknown_names = self.problem.unknown_names()
                        error_msg += (
                            f" \nThe {len(indices)} following parameter(s)"
                            f" have no influence: {[unknown_names[i] for i in indices]} \n{indices}"
                        )
                    if (indices := error_desc.get('residues', [])):
                        residue_names = self.problem.residue_names()
                        error_msg += (
                            f" \nThe {len(indices)} following residue(s)"
                            f" are not influenced: {[residue_names[i] for i in indices]}"
                        )

                if self.parent is not None:
                    raise ArithmeticError(error_msg)
                else:
                    logger.error(error_msg)

            self.solution = dict(
                (key, unknown.default_value)
                for key, unknown in self.problem.unknowns.items()
            )
            self._print_solution()

        else:
            logger.debug('No parameters/residues to solve. Fallback to children execution.')
            for child in self.children.values():
                child.run_once()
            self.owner.run_children_drivers()

        if self._recorder is not None:
            if self.children:
                for child in self.children.values():
                    child.run_once()
                    self._recorder.record_state(child.name, self.status, self.error_code)
            else:
                self._recorder.record_state(self.name, self.status, self.error_code)

    def log_debug_message(self,
        handler: "HandlerWithContextFilters",
        record: logging.LogRecord,
        format: LogFormat = LogFormat.RAW
    ) -> bool:
        """Callback method on the driver to log more detailed information.
        
        This method will be called by the log handler when :py:meth:`~cosapp.utils.logging.LoggerContext.log_context`
        is active if the logging level is lower or equals to VERBOSE_LEVEL. It allows
        the object to send additional log message to help debugging a simulation.

        Parameters
        ----------
        handler : HandlerWithContextFilters
            Log handler on which additional message should be published.
        record : logging.LogRecord
            Log record
        format : LogFormat
            Format of the message

        Returns
        -------
        bool
            Should the provided record be logged?
        """
        message = record.getMessage()
        activate = getattr(record, "activate", None)
        emit_record = super().log_debug_message(handler, record, format)

        try:
            self.options["history"] = activate
        except (KeyError, TypeError):
            pass

        if message.endswith("call_setup_run") or message.endswith("call_clean_run"):
            emit_record = False

        elif activate == True:
            emit_record = False

        elif activate == False:
            emit_record = False

            message = ""
            unknown_trace = None
            residue_trace = None
            residue_names = self.problem.residue_names()
            unknown_names = self.problem.unknown_names()
            for i, record in enumerate(self.__trace):
                if i == 0:
                    unknown_trace = record["x"]
                    residue_trace = record["residues"]
                else:
                    unknown_trace = numpy.vstack((unknown_trace, record["x"]))
                    residue_trace = numpy.vstack((residue_trace, record["residues"]))
                try:
                    jac_record = record["jac"]
                except KeyError:
                    continue
                else:
                    message += f"Iteration {i}\n"
                    with StringIO() as container:
                        numpy.savetxt(container, jac_record, delimiter=",")
                        jacobian = container.getvalue()
                    unknowns = ", ".join(unknown_names)
                    message += f"New Jacobian matrix:\n,{unknowns}\n"
                    for residue, line in zip(residue_names, jacobian.splitlines()):
                        message += f"{residue}, {line}\n"

            def format_record(records: numpy.ndarray, headers: List[str]) -> str:
                records = numpy.atleast_2d(records)
                with StringIO() as stream:
                    writer = csv.writer(stream, delimiter=",", lineterminator="\n")
                    writer.writerows([headers, *records])
                    return stream.getvalue()

            if unknown_trace is not None:
                message += "Unknowns\n{}\n".format(
                    format_record(unknown_trace, unknown_names)
                )
            if residue_trace is not None:
                message += "Residues\n{}\n".format(
                    format_record(residue_trace, residue_names)
                )

            handler.log(
                LogLevel.FULL_DEBUG,
                message,
                name=logger.name,
            )

        return emit_record

    def __set_method(self, method, **options):
        check_arg(method, 'method', NonLinearMethods)

        self.__method = method
        self.__option_aliases = dict()

        if self.__method == NonLinearMethods.NR:
            self.options.declare('tol', 'auto', dtype=(float, str), allow_none=True,
                desc='Absolute tolerance (in max-norm) for the residual.')
            self.options.declare('max_iter', 500, dtype=int,
                desc='The maximum number of iterations.')
            # Note: use a power of 2 for `eps`, to guaranty machine-precision accurate gradients in linear problems
            self.options.declare('eps', 2**(-16), dtype=float, allow_none=True, lower=2**(-30),
                desc='Relative step length for the forward-difference approximation of the Jacobian.')
            self.options.declare('factor', 1.0, dtype=Number, allow_none=True, lower=1e-3, upper=1.0,
                desc='A parameter determining the initial step bound factor * norm(diag * x). Should be in interval [0.001, 1].')
            self.options.declare('partial_jac', True, dtype=bool, allow_none=False,
                desc='Defines if partial Jacobian updates can be computed before a complete Jacobian matrix update.')
            self.options.declare('partial_jac_tries', 10, dtype=int, allow_none=False, lower=1, upper=10,
                desc='Defines how many partial Jacobian updates can be tried before a complete Jacobian matrix update.')
            self.options.declare('jac_update_tol', 0.01, dtype=float, allow_none=False, lower=0, upper=1,
                desc='Tolerance level for partial Jacobian matrix update, based on nonlinearity estimation.')
            self.options.declare('recorder', None, allow_none=True,
                desc='A recorder to store solver intermediate results.')
            self.options.declare('lower_bound', None, dtype=numpy.ndarray, allow_none=True,
                desc='Min values for parameters iterated by solver.')
            self.options.declare('upper_bound', None, dtype=numpy.ndarray, allow_none=True,
                desc='Max values for parameters iterated by solver.')
            self.options.declare('abs_step', None, dtype=numpy.ndarray, allow_none=True,
                desc='Max absolute step for parameters iterated by solver.')
            self.options.declare('rel_step', None, dtype=numpy.ndarray, allow_none=True,
                desc='Max relative step for parameters iterated by solver.')
            self.options.declare('history', False, dtype=bool, allow_none=False,
                desc='Request saving the resolution trace.')
            self.options.declare('tol_update_period', 4, dtype=int, lower=1, allow_none=False,
                desc="Tolerance update period, in iteration number, when tol='auto'.")
            self.options.declare('tol_to_noise_ratio', 16, dtype=Number, lower=1.0, allow_none=False,
                desc="Tolerance-to-noise ratio, when tol='auto'.")

        elif self.__method == NonLinearMethods.POWELL:
            self.__option_aliases = {
                'tol': 'xtol',
                'max_eval': 'maxfev',
            }
            self.options.declare('xtol', 1.0e-7, dtype=float,
                desc="The calculation will terminate if the relative error between two consecutive iterations is at most tol.")
            self.options.declare('maxfev', 0, dtype=int,
                desc='The maximum number of calls to the function. If zero, assumes 100 * (N + 1),'
                    ' where N is the number of elements in x0.')
            self.options.declare('eps', None, dtype=float, allow_none=True,
                desc='A suitable step length for the forward-difference approximation of the Jacobian (for fprime=None).'
                    ' If eps is less than machine precision u, it is assumed that the relative errors in the'
                    ' functions are of the order of u.')
            self.options.declare('factor', 0.1, dtype=float, lower=0.1, upper=100.,
                desc='A parameter determining the initial step bound factor * norm(diag * x). Should be in the interval [0.1, 100].')

        elif self.__method == NonLinearMethods.BROYDEN_GOOD:
            self.__option_aliases = {
                'tol': 'fatol',
                'num_iter': 'nit',
                'max_iter': 'maxiter',
                'min_rel_step': 'xtol',
                'min_abs_step': 'xatol',
            }
            self.options.declare('nit', 100, dtype=int,
                desc='Number of iterations to perform. If omitted (default), iterate unit tolerance is met.')
            self.options.declare('maxiter', 200, dtype=int,
                desc='Maximum number of iterations to make. If more are needed to meet convergence, NoConvergence is raised.')
            self.options.declare('disp', False, dtype=bool,
                desc='Print status to stdout on every iteration.')
            self.options.declare('fatol', 6e-6, dtype=float,
                desc='Absolute tolerance (in max-norm) for the residual. If omitted, default is 6e-6.')
            self.options.declare('line_search', 'armijo', dtype=str, allow_none=True,
                desc='Which type of a line search to use to determine the step '
                    'size in the direction given by the Jacobian approximation. Defaults to ‘armijo’.')
            self.options.declare('jac_options', {'reduction_method': 'svd'},
                dtype=dict, allow_none=True,
                desc='Options for the respective Jacobian approximation. restart, simple or svd')

        self._filter_options(options, self.__option_aliases)

    def extend(self, problem: MathematicalProblem, *args, **kwargs) -> MathematicalProblem:
        """Extend solver inner problem.
        
        Parameters
        ----------
        - problem [MathematicalProblem]:
            Source mathematical problem.
        - *args, **kwargs:
            Additional arguments forwarded to `MathematicalProblem.extend`.

        Returns
        -------
        - MathematicalProblem:
            The extended problem.
        """
        return self._raw_problem.extend(problem, *args, **kwargs)

    def add_unknown(self,
        name: Union[str, Iterable[Union[dict, str]]],
        *args, **kwargs,
    ) -> MathematicalProblem:
        """Add design unknown(s).

        More details in `cosapp.core.MathematicalProblem.add_unknown`.

        Parameters
        ----------
        - name [str or Iterable of dictionary or str]:
            Name of the variable or collection of variables to be added.
        - *args, **kwargs:
            Additional arguments forwarded to `MathematicalProblem.add_unknown`.

        Returns
        -------
        - MathematicalProblem:
            The updated problem.
        """
        return self._raw_problem.add_unknown(name, *args, **kwargs)

    def add_equation(self,
        equation: Union[str, Iterable[Union[dict, str]]],
        *args, **kwargs,
    ) -> MathematicalProblem:
        """Add off-design equation(s).

        More details in `cosapp.core.MathematicalProblem.add_equation`.

        Parameters
        ----------
        - equation [str or Iterable of str of the kind 'lhs == rhs']:
            Equation or collection of equations to be added.
        - *args, **kwargs:
            Additional arguments forwarded to `MathematicalProblem.add_equation`.

        Returns
        -------
        - MathematicalProblem:
            The updated problem.
        """
        return self._raw_problem.add_equation(equation, *args, **kwargs)

    def add_target(self,
        expression: Union[str, Iterable[str]],
        *args, **kwargs,
    ) -> MathematicalProblem:
        """Add deferred off-design equation(s).

        More details in `cosapp.core.MathematicalProblem.add_target`.

        Parameters
        ----------
        - expression: str
            Targetted expression
        - *args, **kwargs : Forwarded to `MathematicalProblem.add_target`

        Returns
        -------
        MathematicalProblem
            The modified mathematical problem
        """
        return self._raw_problem.add_target(expression, *args, **kwargs)


class StandaloneSolverBuilder(BaseSolverBuilder):
    """System building strategy for solvers with no
    `RunSingleCase` sub-drivers. In this case, the actual
    mathematical problem is entirely defined by unknowns and
    equations declared at solver level.
    """
    def build_system(self):
        solver = self.solver
        system = self.solver.owner
        problem = self.problem
        handler = DesignProblemHandler(system)
        handler.design.extend(solver.raw_problem)
        handler.offdesign.extend(system.assembled_problem())
        handler.prune()
        problem.clear()
        problem.extend(handler.merged_problem(), copy=False)
        self.initial_values = problem.unknown_vector()
        self.is_design_unknown = dict.fromkeys(problem.unknowns, True)

    def update_residues(self):
        for residue in self.problem.residues.values():
            residue.update()


class MultipointSolverBuilder(BaseSolverBuilder):
    """System building strategy for solvers with one or more
    `RunSingleCase` sub-drivers. In this case, the actual
    mathematical problem is a combination of unknowns and
    equations declared at solver level, and of local design
    and off-design problems declared at case level.
    """
    def __init__(self, solver: NonLinearSolver, points: List[RunSingleCase]):
        super().__init__(solver)
        self.points = points

    def build_system(self):
        solver = self.solver
        system = self.solver.owner
        problem = self.problem
        handler = DesignProblemHandler(system)
        handler.design.extend(solver.raw_problem, equations=False)
        handler.offdesign.extend(solver.raw_problem, unknowns=False)
        handler.prune()  # resolve aliasing
        # Create assembled problem
        problem.clear()
        problem.extend(handler.design)

        self.initial_values = handler.design.unknown_vector()

        if len(self.points) > 1:
            # If more than one `RunSingleCase` drivers are present,
            # use a name wrapper to distinguish dict entries for
            # residues and off-design unknowns in global problem
            get_key_wrapper = lambda case: (
               lambda key: f"{case.name}[{key}]"
            )
        else:
            get_key_wrapper = lambda case: None

        # Set of unknown names to detect design/off-design conflicts
        design_unknown_names = set(handler.design.unknowns)

        for point in self.points:
            local = point.processed_problems
            # Check that local problem does not introduce design/off-design conflicts
            design_unknown_names.update(local.design.unknowns)
            common = design_unknown_names.intersection(local.offdesign.unknowns)
            if common:
                kind = "unknown"
                if len(common) > 1:
                    names = ", ".join(map(repr, sorted(common)))
                    names = f"({names}) are"
                    kind += "s"
                else:
                    names = f"{common.pop()!r} is"
                raise ValueError(
                    f"{names} defined as design and off-design {kind} in {point.name!r}"
                )
            # Enforce solver-level off-design problem to child case
            point.add_offdesign_problem(handler.offdesign)
            # Assemble case problems (design & off-design)
            wrapper = get_key_wrapper(point)
            problem.extend(local.design, copy=False, residue_wrapper=wrapper)
            problem.extend(local.offdesign, copy=False, unknown_wrapper=wrapper, residue_wrapper=wrapper)
            self.initial_values = numpy.append(self.initial_values, point.get_init(solver.force_init))

        self.is_design_unknown = dict.fromkeys(problem.unknowns, False)
        for name in design_unknown_names:
            self.is_design_unknown[name] = True

    def update_residues(self):
        # Nothing to do, since residues are updated
        # by `RunSingleCase` sub-drivers
        pass
