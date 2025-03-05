import numpy
import abc
import csv
import inspect
from io import StringIO
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    Iterable,
    TypeVar,
    Type,
)

from cosapp.core.numerics.basics import MathematicalProblem, SolverResults
from cosapp.core.numerics.enum import NonLinearMethods
from cosapp.drivers.driver import Driver, System
from cosapp.drivers.abstractsolver import AbstractSolver
from cosapp.drivers.runsinglecase import RunSingleCase
from cosapp.drivers.utils import DesignProblemHandler
from cosapp.utils.logging import LogFormat, LogLevel, HandlerWithContextFilters
from cosapp.utils.options_dictionary import HasOptions

from cosapp.core.numerics.solve import (
    AbstractNonLinearSolver,
    NewtonRaphsonSolver,
    ScipyRootSolver,
)

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
            n = unknown.size
            value = x[counter] if unknown.is_scalar else x[counter: counter + n]
            unknown.update_default_value(value, checks=False)
            counter += n

            # Update design unknowns
            if self.is_design_unknown[name]:
                unknown.set_to_default()


class BaseSolverRecorder(abc.ABC):
    """Abstract interface for solver recorder"""

    @abc.abstractmethod
    def record(self, context: str) -> None:
        """Record owner system data using solver's recorder"""


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
        "__method",
        "__option_aliases",
        "__trace",
        "__results",
        "__design_unknowns",
        "compute_jacobian",
        "__builder",
        "_solver",
    )

    def __init__(
        self,
        name: str,
        owner: Optional[System] = None,
        method: Union[
            NonLinearMethods, Type[AbstractNonLinearSolver]
        ] = NonLinearMethods.NR,
        **options,
    ) -> None:
        """Initialize driver

        Parameters
        ----------
        name: str, optional
            Name of the `Driver`.
        owner: System, optional
            :py:class:`~cosapp.systems.system.System` to which this driver belong; defaults to `None`.
        method : Union[NonLinearMethods, Type[AbstractNonLinearSolver]]
            Resolution method to use
        **kwargs:
            Additional keywords arguments forwarded to base class.
        """
        if isinstance(method, str):
            method = NonLinearMethods[method]

        if isinstance(method, NonLinearMethods):
            if method == NonLinearMethods.NR:
                self._solver = NewtonRaphsonSolver()
            else:
                self._solver = ScipyRootSolver(method=method)

        elif inspect.isclass(method) and issubclass(method, AbstractNonLinearSolver):
            self._solver = method()

        else:
            raise TypeError(
                "Argument 'method' must be either a `NonLinearMethods`"
                f" or a derived class of `AbstractNonLinearSolver`; got {method!r}."
            )

        self.__method = method
        self.__trace: List[Dict[str, Any]] = list()
        self.__results: SolverResults = None
        self.__builder: BaseSolverBuilder = None

        self.compute_jacobian = True  # type: bool
        # desc='Should the Jacobian matrix be computed?'

        super().__init__(name, owner, **options)

    def _get_nested_objects_with_options(self) -> Iterable[HasOptions]:
        """Gets nested objects having options."""
        return (self._solver, )

    @classmethod
    def _slots_not_jsonified(cls) -> tuple[str]:
        """Returns slots that must not be JSONified."""
        return (
            *super()._slots_not_jsonified(),
            "_NonLinearSolver__builder",
        )

    def reset_problem(self) -> None:
        """Reset mathematical problem"""
        super().reset_problem()
        self.__design_unknowns = dict()

    @property
    def non_linear_solver(self) -> Any:
        return self._solver

    @property
    def linear_solver(self) -> Any:
        return self._solver._linear_solver

    @property
    def jac(self) -> Any:
        ls = self.linear_solver
        if ls.need_jacobian:
            return ls.jacobian
        return None

    @jac.setter
    def jac(self, value) -> None:
        ls = self.linear_solver
        if ls.need_jacobian:
            ls.jacobian = value

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

    def add_child(
        self, child: AnyDriver, execution_index: Optional[int] = None, desc=""
    ) -> AnyDriver:
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

    def resolution_method(
        self,
        fresidues: Callable[[Sequence[float], Union[float, str], bool], numpy.ndarray],
        x0: Sequence[float],
        args: Tuple[Union[float, str]] = (),
        options: Optional[Dict[str, Any]] = None,
        callback: Optional[Callable[[], None]] = None,
    ) -> SolverResults:
        """Solves the mathematical problem with a non linear method."""

        self._solver.set_options()  # required if options were changed after instantiation
        self._solver.log_level = (
            LogLevel.INFO if self.options.verbose else LogLevel.DEBUG
        )

        results = self._solver.solve(fresidues, x0, args, callback=callback)

        if results.success:
            self.compute_jacobian = False

        return results

    def setup_run(self) -> None:
        super().setup_run()
        self._init_problem()

    def _print_solution(self) -> None:  # TODO better returning a string
        """Print the solution in the log."""
        if self.options.verbose > 0:
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
                target = self.options.tol
            except:
                target = 0
            logger.debug(f" # Current tolerance {tol} for target {target}")

    def _init_problem(self):
        """Initialize mathematical problem"""
        logger.debug(
            "\n".join(
                [
                    "*" * 40,
                    "*",
                    "* Assemble mathematical problem",
                    "*",
                    "*" * 40,
                ]
            )
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
            "\n".join(
                [
                    "Mathematical problem:",
                    f"{'<empty>' if self.problem.is_empty() else self.problem}",
                ]
            )
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
        logger.debug(f"Call fresidues with {x = !r}")
        self.set_iteratives(x)
        self._update_system()
        self.__builder.update_residues()
        residues = self.problem.residue_vector()
        logger.debug(f"Residues: {residues!r}")
        return residues

    def set_iteratives(self, x: Sequence[float]) -> None:
        x = numpy.asarray(x)
        self.__builder.update_unknowns(x)

    def compute(self) -> None:
        """Run the resolution method to find free variable values that cancel the residues"""
        # Reset status
        self.status = ""
        self.error_code = "0"

        try:
            self.problem.validate()
        except ArithmeticError:
            self.status = "ERROR"
            self.error_code = "9"
            raise

        record_history = self.options.get("history", False)
        must_resolve = len(self.initial_values) > 0

        if must_resolve:
            self.solution = {}

            if record_history and self._recorder:
                # Record system data at each solver iteration
                callback = SolverRecorderCallback(self)
            else:
                callback = None

            # compute first order
            results = self.resolution_method(
                self._fresidues,
                self.initial_values,
                options=self._get_solver_options(),
                callback=callback,
            )
            self.__results = results
            self.__trace = getattr(results, "trace", list())

            if results.success:
                self.status = ""
                self.error_code = "0"
                logger.info(f"solver : {self.name}{results.message}")

            else:
                self.status = "ERROR"
                self.error_code = "9"

                error_msg = f"The solver failed: {results.message}"

                error_desc = getattr(results, "jac_errors", {})
                if error_desc:
                    indices = error_desc.get("unknowns", [])
                    if len(indices) > 0:
                        unknown_names = self.problem.unknown_names()
                        error_msg += (
                            f" \nThe {len(indices)} following parameter(s)"
                            f" have no influence: {[unknown_names[i] for i in indices]}"
                        )
                    indices = error_desc.get("residues", [])
                    if len(indices) > 0:
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
            logger.debug("No parameters/residues to solve. Fallback to system update.")
            self._update_system()

        if not record_history or not must_resolve:
            # Record system data at the end of the simulation
            callback = SolverRecorderCallback(self)
            callback.record()

    def _get_solver_options(self) -> Dict[str, Any]:
        options = self._filter_options(self.__option_aliases)

        if self.method == NonLinearMethods.NR:
            self.options.update(self._get_solver_limits())
            # self.options["compute_jacobian"] = self.compute_jacobian

        return options

    def log_debug_message(
        self,
        handler: HandlerWithContextFilters,
        record: logging.LogRecord,
        format: LogFormat = LogFormat.RAW,
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

    def _declare_options(self):
        """Declare solver options, and options aliases possibly necessary to use numpy functions"""
        super()._declare_options()
        self.__option_aliases = dict()

    def extend(
        self, problem: MathematicalProblem, *args, **kwargs
    ) -> MathematicalProblem:
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

    def add_unknown(
        self,
        name: Union[str, Iterable[Union[dict, str]]],
        *args,
        **kwargs,
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

    def add_equation(
        self,
        equation: Union[str, Iterable[Union[dict, str]]],
        *args,
        **kwargs,
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

    def add_target(
        self,
        expression: Union[str, Iterable[str]],
        *args,
        **kwargs,
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
            get_key_wrapper = lambda case: (lambda key: f"{case.name}[{key}]")
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
            problem.extend(
                local.offdesign,
                copy=False,
                unknown_wrapper=wrapper,
                residue_wrapper=wrapper,
            )
            self.initial_values = numpy.append(
                self.initial_values, point.get_init(solver.force_init)
            )

        self.is_design_unknown = dict.fromkeys(problem.unknowns, False)
        for name in design_unknown_names:
            self.is_design_unknown[name] = True

    def update_residues(self):
        # Nothing to do, since residues are updated
        # by `RunSingleCase` sub-drivers
        pass


class EmptySolverRecorder(BaseSolverRecorder):
    """Specialization for solvers with no recorder."""

    def record(self, context: str) -> None:
        pass


class StandaloneSolverRecorder(BaseSolverRecorder):
    """Solver recorder specialization for solvers with no sub-drivers."""

    def __init__(self, solver: NonLinearSolver) -> None:
        self._solver = solver

    def record(self, context: str) -> None:
        solver = self._solver
        if not context:
            context = solver.name
        solver.recorder.record_state(context, solver.status, solver.error_code)


class CompositeSolverRecorder(BaseSolverRecorder):
    """Solver recorder specialization for solvers with sub-drivers."""

    def __init__(self, solver: NonLinearSolver) -> None:
        self._solver = solver

    def record(self, context: str) -> None:
        solver = self._solver
        recorder = solver.recorder
        if context:
            context = f" ({context})"
        for child in solver.children.values():
            child.run_once()
            recorder.record_state(
                f"{child.name}{context}", solver.status, solver.error_code
            )


class SolverRecorderCallback:
    """Callback functor to monitor solver residues"""

    def __init__(self, solver: NonLinearSolver) -> None:
        self.reset_iter()
        if solver.recorder is None:
            self.recorder = EmptySolverRecorder()
        elif solver.children:
            self.recorder = CompositeSolverRecorder(solver)
        else:
            self.recorder = StandaloneSolverRecorder(solver)

    def reset_iter(self) -> None:
        """Reset inner iteration counter"""
        self._iter = 0

    def record(self, context="") -> None:
        """Record owner system data using solver's recorder"""
        self.recorder.record(context)

    def __call__(self, *args, **kwargs) -> None:
        """Callback function"""
        self.record(f"iter {self._iter}")
        self._iter += 1
