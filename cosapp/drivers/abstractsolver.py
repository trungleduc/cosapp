import abc
import json
import logging
from copy import copy
from numbers import Number
from typing import (Any, AnyStr, Callable, Dict, List, NoReturn, Optional,
                    Sequence, Tuple, Union)

import numpy

from cosapp.core.numerics.basics import MathematicalProblem, SolverResults
from cosapp.drivers.driver import Driver
from cosapp.drivers.iterativecase import IterativeCase
from cosapp.drivers.runonce import RunOnce
from cosapp.utils.options_dictionary import OptionsDictionary

logger = logging.getLogger(__name__)


class AbstractSolver(Driver):
    """
    Solve a `System`

    Parameters
    ----------
    name : str
        Name of the driver
    owner : System, optional
        :py:class:`~cosapp.systems.system.System` to which this driver belong; default None
    **kwargs : Any
        Keyword arguments will be used to set driver options
    """

    __slots__ = ('force_init', 'problem', 'initial_values', 'solution')

    def __init__(self,
        name: str,
        owner: "Optional[cosapp.systems.System]" = None,
        **kwargs
    ) -> NoReturn:
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

        self.force_init = False  # type: bool
            # desc="Force the initial values to be used."

        self.problem = None  # type: Optional[MathematicalProblem]
            # desc="Mathematical problem to solve."
        self.initial_values = numpy.empty(0, dtype=numpy.float)  # type: numpy.ndarray
            # desc="List of initial values for all iteratives.",
        self.solution = {}  # type: Dict[str, float]
            # desc="Dictionary (name, value) of the latest solution reached."

    def _filter_options(self, kwargs, aliases: Dict[str, str] = dict()):
        """
        Translate option names into self.options using an alias dictionary, to handle cases where
        a common option name, such as 'tol', is passed to a specific solver/function with a different name.
        
        For example, in scipy.optimize.root(), the convergence criterion may be referred to as 'ftol', 'gtol'...
        depending on the invoked algorithm (Levenberg-Marquardt, Powell, Broyden's good, etc.).
        """
        keys = list(kwargs.keys())  # make a copy of keys only

        for key in keys:
            value = kwargs.pop(key)
            try:
                key = aliases[key]
            except KeyError:
                pass
            if key in self.options:
                self.options[key] = value
            else:
                raise KeyError("Unknown solver option '{}'".format(key))

    @property
    def _default_driver_name(self) -> str:
        """str : Name for the default driver."""
        return "runner"

    def _get_solver_limits(self) -> Dict[str, numpy.ndarray]:
        """Returns the step limitations for all iteratives.

        There are 4 types of limits defined:
        - lower_bound: lower bound of the iteratives
        - upper_bound: upper bound of the iteratives
        - abs_step: maximal absolute step of the iteratives
        - rel_step: maximal relative step of the iteratives

        Returns
        -------
        Dict[str, numpy.ndarray]
            Dictionary with the limits for all iteratives.
        """
        options = {
            name : []
            for name in ("lower_bound", "upper_bound", "abs_step", "rel_step")
        }

        for unknown in self.problem.unknowns.values():
            def append(name, const):
                array = numpy.full_like(unknown.value, const, dtype=type(const))
                options[name] = numpy.concatenate((options[name], array.flatten()))
            append("lower_bound", unknown.lower_bound)
            append("upper_bound", unknown.upper_bound)
            append("abs_step", unknown.max_abs_step)
            append("rel_step", unknown.max_rel_step)

        return options

    def compute_before(self):
        """Contains the customized `Module` calculation, to execute before children.
        """
        logger.debug(f"Set unknowns initial values: {self.initial_values}")
        x = self.initial_values
        counter = 0
        # Pass the x vector to all driver so they set the new x (at appropriate time)
        for child in self.children.values():
            if isinstance(child, IterativeCase):
                counter += child.set_iteratives(x[counter:])

    def _precompute(self) -> NoReturn:
        """Set up the mathematical problem."""
        # TODO we should check that all variables are of numerical types
        super()._precompute()

        self.initial_values = numpy.empty(0, dtype=numpy.float)
        self.problem = MathematicalProblem(self.name, self.owner)

    def _fresidues(self,
        x: Sequence[float],
        update_residues_ref: bool = True,
    ) -> numpy.ndarray:
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
        counter = 0
        # Pass the x vector to all driver so they set the new x (at appropriate time)
        for child in self.children.values():
            if isinstance(child, IterativeCase):
                counter += child.set_iteratives(x[counter:])

        # Run all points
        for child in self.exec_order:
            logger.debug(f"Call {child}.run_once")
            self.children[child].run_once()

        residues = self.problem.residues_vector
        logger.debug(f"Residues: {residues!r}")
        return residues

    @abc.abstractmethod
    def resolution_method(self,
        fresidues: Callable[[Sequence[float], Union[float, str], bool], numpy.ndarray],
        x0: Sequence[float],
        args: Tuple[Union[float, str]] = (),
        options: Optional[OptionsDictionary] = None,
    ) -> SolverResults:
        """Function call to cancel the residues.

        Parameters
        ----------
        fresidues : Callable[[Sequence[float], Union[float, str]], numpy.ndarray]
            Residues function taking two parameters (evaluation vector, time/ref) and returning the residues
        x0 : Sequence[float]
            The initial values vector to converge to the solution
        args : Tuple[Union[float, str], bool], optional
            A tuple of additional argument for fresidues starting with the time/ref parameter and the need to update
            residues reference
        options : OptionsDictionary, optional
            Options for the numerical resolution method

        Returns
        -------
        SolverResults
            Solution container
        """
        pass

    # Don't clean initial_values and problem => could be useful for debugging
    # def _postcompute(self) -> NoReturn:
    #     """Undo pull inputs and reset iteratives sets."""
    #     self.initial_values = numpy.empty(0, dtype=numpy.float)
    #     self.problem = None
    #     super()._postcompute()

    def save_solution(self, file: Optional[str] = None) -> Dict[str, Union[Number, List[Number]]]:
        """Save the latest solver solution.

        If `file` is specified, the solution will be saved in it in JSON format.

        Parameters
        ----------
        file : str, optional
            Filename to save the answer in; default None (i.e. data will not be saved)

        Returns
        -------
        Dict[str, Union[Number, List[Number]]]
            Dictionary of the latest solution
        """
        latest_answer = dict()

        for k, v in self.solution.items():
            if isinstance(v, numpy.ndarray):
                v = v.tolist()
            latest_answer[k] = copy(v)

        if file:
            with open(file, "w") as outfile:
                json.dump(latest_answer, outfile)

        return latest_answer

    def load_solution(self,
        solution: Union[Dict[str, Union[Number, numpy.ndarray]], AnyStr],
        case: Optional[str] = None,
    ):
        """Load the provided solution to initialize the solver.

        The solution can be provided directly as a dictionary or from a filename to be read.

        Parameters
        ----------
        solution : Dict[str, Union[Number, numpy.ndarray]] or str
            Dictionary of the latest solution to load or the filename in JSON format to read from.
        case : str, optional
            Case to initialize with the solution; default None (i.e. will be guessed from variable name)
        """
        # TODO Fred is it better to set the initial values or to override the previous solution?
        #   In case the solution does not cover all offdesign unknowns, the later should be better.
        from cosapp.systems import System

        if not isinstance(solution, (dict, str)):
            raise TypeError(
                "Answer expected as dict or str name of the json file, got {}".format(
                    type(solution).__qualname__
                )
            )

        if isinstance(solution, str):
            with open(solution, "r") as f:
                data = json.load(f)
        else:
            data = solution

        with System.set_master(repr(self.owner)) as is_master:
            if is_master:
                self.owner.open_loops()

            try:
                if case is None:
                    for name, value in data.items():
                        for child in filter(
                            lambda d: isinstance(d, RunOnce), self.children.values()
                        ):
                            if name.startswith(child.name):  # We got a offdesign variable
                                name = name[len(child.name) + 1 : -1]
                                if name in self.owner:
                                    child.set_init({name: numpy.asarray(value)})
                                    break
                            elif (
                                name in child.design.unknowns
                            ):  # We may have a design variable
                                child.set_init({name: numpy.asarray(value)})
                                break
                else:
                    child = self.children[case]
                    if not isinstance(child, RunOnce):
                        raise TypeError(
                            "Only driver derived from RunOnce can be initialized; got '{}' for driver '{}'.".format(
                                type(child).__qualname__, case
                            )
                        )

                    for name, value in data.items():
                        if name.startswith(child.name):
                            name = name[len(child.name) + 1 : -1]
                        if name in self.owner:
                            child.set_init({name: numpy.asarray(value)})

            finally:  # Ensure to clean the system
                if is_master:
                    self.owner.close_loops()
