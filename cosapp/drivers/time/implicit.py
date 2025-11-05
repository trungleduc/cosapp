import abc
import numpy
from typing import Optional, Iterable, Any

from cosapp.drivers.time.base import AbstractTimeDriver, System
from cosapp.core import MathematicalProblem
from cosapp.core.numerics.solve import NewtonRaphsonSolver
from cosapp.utils.options_dictionary import HasOptions


class ImplicitTimeDriver(AbstractTimeDriver):
    """Abstract base class for implicit integrators."""

    __slots__ = (
        "_has_intrinsic_problem",
        "_intrinsic_problem",
        "_transient_problem",
        "_design_problem",
        "_curr_res",
        "_solver",
        "_prev_x",
    )

    def __init__(self,
        name = "Implicit time driver",
        owner: Optional[System] = None,
        time_interval: Optional[tuple[float, float]] = None,
        dt: Optional[float] = None,
        record_dt: bool = False,
        **options,
    ):
        self._has_intrinsic_problem = False
        self._intrinsic_problem: MathematicalProblem = None
        self._transient_problem: MathematicalProblem = None
        self._design_problem: MathematicalProblem = None
        self._solver = NewtonRaphsonSolver()
        self._prev_x = numpy.empty(0)
        self._curr_res = numpy.empty(0)
        super().__init__(name, owner, time_interval, dt, record_dt, **options)

    @property
    def problem(self):
        """Mathematical problem handled by the driver, gathering the
        intrinsic problem of the owner system and its transient variables.
        """
        return self._transient_problem

    def setup_run(self) -> None:
        super().setup_run()
        self._reset_time_problem()
        self._update_solver_options()

    def _reset_time_problem(self) -> None:
        # Get intrinsic problem of owner system
        context = self._owner
        self._intrinsic_problem = initial_problem = context.assembled_problem()

        # Extend the intrinsic problem with the design problem of the driver
        design_problem = self.__get_design_problem()
        initial_problem.extend(design_problem, copy=False)

        self._has_intrinsic_problem = not initial_problem.is_empty()

        # Add transient variables as unknowns of the time-dependent problem
        self._transient_problem = problem = context.new_problem()
        problem.extend(initial_problem, copy=False)
        transients = self._var_manager.problem.transients.values()
        for transient in transients:
            path = context.get_path_to_child(transient.context)
            name = f"{path}.{transient.name}" if path else transient.name
            problem.add_unknown(name, max_abs_step=transient.max_abs_step)

        self._prev_x = problem.unknown_vector()
        super()._reset_time_problem()

    def _initialize(self):
        super()._initialize()

        # Equilibrate system @ t=0 (if necessary)
        initial_problem = self._intrinsic_problem
        initial_problem.validate()
        initial_problem.activate_targets()
        self._solve_intrinsic_problem()

        # Initialize the unknown vector
        transient_problem = self._transient_problem
        transient_problem.activate_targets()
        transient_problem.update_residues()
        self._prev_x = self._transient_problem.unknown_vector()

    def _solve_intrinsic_problem(self, x0: Optional[numpy.ndarray]=None) -> None:
        """Solve the intrinsic problem of the owner system."""
        if self._has_intrinsic_problem:
            if x0 is None:
                x0 = self._intrinsic_problem.unknown_vector()
            self._solver.solve(self._fresidues_init, x0=x0)

    @abc.abstractmethod
    def _time_residues(self, dt: float, current: bool) -> numpy.ndarray:
        """Computes and returns the current- or next-time component
        of the transient problem residue vector.
        
        Parameters:
        -----------
        - dt [float]:
            Time step
        - current [bool]:
            If `True`, compute the part of the residues known at time n.
            If `False`, compute the time (n + 1) part of the residues.
        """

    @staticmethod
    def _update_unknowns(problem: MathematicalProblem, x: numpy.ndarray):
        """Update the unknowns in `problem` from unknown vector `x`."""
        counter = 0
        for unknown in problem.unknowns.values():
            n = unknown.size
            value = x[counter] if unknown.is_scalar else x[counter: counter + n]
            unknown.update_default_value(value, checks=False)
            unknown.set_to_default()
            counter += n

    def _update_transients(self, dt: float):
        """Integrate transient variable over time step `dt`."""
        self._curr_res = self._time_residues(dt, current=True)
        prev_x = self._prev_x
        if len(prev_x) > 0:
            result = self._solver.solve(self._fresidues, x0=prev_x, args=(self.time, dt))
            prev_x[:] = result.x

    def _postcompute(self) -> None:
        # Synch unknown values with final system state,
        # in case the simulation was ended by an event.
        for unknown in self.problem.unknowns.values():
            unknown.update_default_value(unknown.value, checks=False)
            unknown.set_to_default()
        super()._postcompute()

    def _fresidues_init(self, x: numpy.ndarray) -> numpy.ndarray:
        """Function returning the residue vector of the initial
        (intrinsic) problem of the owner system.

        Parameters
        ----------
        x : numpy.array[float]
            Unknown vector at time t=0

        Returns
        -------
        numpy.ndarray
            Residue vector
        """
        problem = self._intrinsic_problem
        self._update_unknowns(problem, x)
        self._update_system()
        problem.update_residues()
        return problem.residue_vector()

    def _fresidues(self, x: numpy.ndarray, t: float, dt: float) -> numpy.ndarray:
        """Function returning the residue vector of the full transient problem.

        Parameters
        ----------
        - x [numpy.array[float]]:
            Unknown vector at time t
        - t [float]:
            Current time
        - dt [float]:
            Time step

        Returns
        -------
        numpy.ndarray
            Residue vector
        """
        problem = self._transient_problem
        self._update_unknowns(problem, x)
        self._set_time(t + dt)
        problem.update_residues()
        inner_res = problem.residue_vector()
        next_res = self._time_residues(dt, current=False)
        time_res = next_res - self._curr_res
        return numpy.concatenate((inner_res, time_res))

    def _get_nested_objects_with_options(self) -> Iterable[HasOptions]:
        """Gets nested objects having options."""
        return (self._solver, )

    def _update_solver_options(self) -> dict[str, Any]:
        mapping = {
            "lower_bound": "lower_bound",
            "upper_bound": "upper_bound",
            "abs_step": "max_abs_step",
            "rel_step": "max_rel_step",
        }
        options = dict.fromkeys(mapping, [])
        for unknown in self._transient_problem.unknowns.values():
            for name, attr_name in mapping.items():
                const = getattr(unknown, attr_name)
                array = numpy.full_like(unknown.value, const, dtype=type(const))
                options[name] = numpy.concatenate((options[name], array.flatten()))

        self.options.update(options)
        self._solver.set_options()

    def is_standalone(self) -> bool:
        """Is this Driver able to solve a system?

        Returns
        -------
        bool
            Ability to solve a system or not.
        """
        return True

    def _pre_update_system(self) -> None:
        """Solve intrinsic problem (if any) during transitions
        before calling method `_update_system`.
        """
        self._solve_intrinsic_problem()

    def add_unknown(self, name: str, *args, **kwargs):
        """Add an unknown variable to the design problem.

        Parameters
        ----------
        name: str
            Name of the unknown variable to add.
        *args, **kwargs:
            Additional arguments forwarded to the unknown.

        Returns
        -------
        `MathematicalProblem`: design problem of the driver.
        """
        design_problem = self.__get_design_problem()
        return design_problem.add_unknown(name, *args, **kwargs)

    def add_equation(self, equation: str, *args, **kwargs):
        """Add an equation to the design problem.

        Parameters
        ----------
        equation: str
            Equation of the kind `lhs == rhs`.
        *args, **kwargs:
            Additional arguments forwarded to the equation.
        Returns
        -------
        `MathematicalProblem`: design problem of the driver.
        """
        design_problem = self.__get_design_problem()
        return design_problem.add_equation(equation, *args, **kwargs)

    def add_problem(self, problem: MathematicalProblem, copy=False):
        """Add a mathematical problem to the design problem.

        Parameters
        ----------
        problem: MathematicalProblem
            Problem to add.
        copy: bool
            If `True`, the problem is copied before being added.

        Returns
        -------
        `MathematicalProblem`: design problem of the driver.
        """
        design_problem = self.__get_design_problem()
        design_problem.extend(problem, copy=copy)
        return design_problem

    def clear_problem(self) -> None:
        """Clear the design problem associated to the driver (if any)."""
        problem = self._design_problem
        if problem is not None:
            problem.clear()

    def __get_design_problem(self) -> MathematicalProblem:
        """Get the design problem associated to the driver.
        Creates it if it does not exist.

        Returns
        -------
        `MathematicalProblem`: Design problem of the driver.

        Raises
        ------
        `RuntimeError`: If the driver has no owner system.
        """
        if self._design_problem is None:
            context = self._owner
            if context is None:
                raise RuntimeError("Cannot create design problem: driver has no owner system.")
            self._design_problem = context.new_problem()
        return self._design_problem
