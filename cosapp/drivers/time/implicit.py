import abc
import numpy
from cosapp.drivers.time.base import AbstractTimeDriver, System
from cosapp.core import MathematicalProblem
from cosapp.core.numerics.solve import NewtonRaphsonSolver
from cosapp.utils.options_dictionary import HasOptions
from typing import Optional, Iterable, Any


class ImplicitTimeDriver(AbstractTimeDriver):
    """Abstract base class for implicit integrators."""

    __slots__ = (
        "_has_intrinsic_problem",
        "_intrinsic_problem",
        "_transient_problem",
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
        self._intrinsic_problem = context.assembled_problem()
        self._has_intrinsic_problem = not self._intrinsic_problem.is_empty()

        # Add transient variables as unknowns of the time-dependent problem
        self._transient_problem = problem = context.new_problem()
        problem.extend(self._intrinsic_problem, copy=False)
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
        x0 = initial_problem.unknown_vector()
        self._has_intrinsic_problem = (x0.size > 0)
        if self._has_intrinsic_problem:
            self._solver.solve(self._fresidues_init, x0=x0)
        
        # Initialize the unknown vector
        self._transient_problem.update_residues()
        self._prev_x = self._transient_problem.unknown_vector()

    @abc.abstractmethod
    def _time_residues(self, dt: float, current: bool):
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
        result = self._solver.solve(self._fresidues, x0=self._prev_x, args=(self.time, dt))
        self._prev_x[:] = result.x

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
        if self._has_intrinsic_problem:
            x0 = self._intrinsic_problem.unknown_vector()
            self._solver.solve(self._fresidues_init, x0=x0)
