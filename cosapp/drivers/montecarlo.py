from __future__ import annotations

import numpy
import logging
from collections.abc import Iterable, Iterator
from typing import Any, Optional, Union
from functools import singledispatchmethod
from scipy.stats import qmc

from cosapp.ports.port import BasePort
from cosapp.drivers.abstractsetofcases import AbstractSetOfCases, System
from cosapp.drivers.abstractsolver import AbstractSolver
from cosapp.utils.distributions import Distribution
from cosapp.utils.helpers import check_arg
from cosapp.systems.system import SystemConnector

logger = logging.getLogger(__name__)


class RandomVariable:
    """Class representing a random variable in a system of interest"""

    __slots__ = ("name", "_ref", "distribution", "connector")

    def __init__(self, system: System, varname: str, distribution: Optional[Distribution]=None):
        ref = system.name2variable[varname]
        port = ref.mapping

        if not isinstance(port, BasePort):
            raise TypeError(f"{varname!r} is not a variable of {system.name!r}.")

        if not port.is_input:
            raise TypeError(f"{varname!r} is not an input variable.")

        if distribution is None:
            distribution = port.get_details(ref.key).distribution
        elif not isinstance(distribution, Distribution):
            raise TypeError(
                f"Distribution for '{system.name}.{varname}' is expected to be of type `Distribution`"
                f"; got a {type(distribution).__name__}."
            )

        self._ref = ref
        self.name = varname
        self.distribution: Optional[Distribution] = distribution
        self.connector: Optional[SystemConnector] = None

        # Test if the variable is aliased or connected
        try:
            alias = system.input_mapping[varname]

        except KeyError:
            parent: System = port.owner.parent
            if parent is not None:
                port_varname = ref.key
                connectors = parent.all_connectors()
                for connector in filter(lambda c: c.sink is port, connectors):
                    if port_varname in connector.sink_variables():
                        self.connector = connector
                        break

        else:
            self._ref = alias

    @property
    def value(self):
        return self._ref.value

    @value.setter
    def value(self, value):
        self._ref.value = value
    
    def add_noise(self, quantile: Optional[float]=None) -> float:
        delta = self.draw(quantile)
        self.set_perturbation(delta)
        return delta

    def draw(self, quantile: Optional[float]=None) -> float:
        return self.distribution.draw(quantile)

    def set_perturbation(self, value) -> None:
        if (connector := self.connector):
            connector.set_perturbation(self._ref.key, value)
        else:
            self._ref.value += value

    def __repr__(self):
        distribution = self.distribution
        return f"RandomVariable({self.name}, {distribution=})"


# TODO linearization does not support multipoint cases
# TODO Does not work for vector variables (at least partially connected one)
# TODO Does it work if a subsystem mutates to higher/lower fidelity
# TODO We don't forbid using an unknown
class MonteCarlo(AbstractSetOfCases):
    """
    This driver execute a MonteCarlo simulation on its system.
    """

    __slots__ = (
        'draws', 'linear', '_random_variables', 'response_varnames', '_solver',
        'X0', 'Y0', 'A', 'perturbations', 'reference_case_solution',
    )

    def __init__(self, name: str, owner: Optional[System]=None, **options) -> None:
        """Initialize driver

        Parameters
        ----------
        name: str, optional
            Name of the `Driver`.
        owner: System, optional
            :py:class:`~cosapp.systems.system.System` to which this driver belong; defaults to `None`.
        **kwargs:
            Additional keywords arguments forwarded to base class.
        """
        super().__init__(name, owner, **options)
        self.draws = 200   # desc="Number of cases performed for Montecarlo calculations."
        self.linear = False  # desc="True for linearisation of system before Montecarlo calculation. Default False."

        self._random_variables: dict[str, RandomVariable] = {}  # desc="Random variables in the system."
        self.response_varnames: list[str] = []  #  desc="Variable names to study through Monte Carlo calculations."
        self._solver: AbstractSolver = None  # desc="Solver acting. Used for re-init of case."
        self.reference_case_solution: dict[str, float] = {}

        self.X0: numpy.ndarray = None  # desc="Vector of imposed disturbed values"
        self.Y0: numpy.ndarray = None  # desc="Vector of output evaluated disturbed values."            
        self.A: numpy.ndarray = None   # desc="Matrice of influence of imposed disturbed values on results."
        self.perturbations = numpy.empty(0)  # desc="Array of perturbations applied on the system."

    @classmethod
    def _slots_not_jsonified(cls) -> tuple[str]:
        """Returns slots that must not be JSONified."""
        return (*super()._slots_not_jsonified(), "_random_variables")

    @singledispatchmethod
    def add_random_variable(self, name, distribution: Optional[Distribution]=None) -> None:
        """Add variable to be randomly perturbated, following an optional distribution.

        The perturbation distribution is defined by the variable distribution details.

        Parameters
        ----------
        varname : Union[str, Iterable[str], dict[str, Distribution]]
            Name of the variable to be perturbed, or iterable thereof.
            If the argument is a dict, it is interpreted as a {name: distribution} specification
        distribution : cosapp.utils.distributions.Distribution | None
            Distribution of the variable(s) to be perturbed.
            If `None` (default), distributions are deduced from variable names.
            This argument is only meaningful when a variable name or a list thereof is provided.

        Examples
        --------
        >>> from cosapp.utils.distributions import Normal, Uniform
        >>>
        >>> montecarlo = system.add_driver(MonteCarlo("montecarlo"))
        >>>
        >>> montecarlo.add_random_variable("port.v")
        >>> montecarlo.add_random_variable("port.v", Normal(...))
        >>>
        >>> montecarlo.add_random_variable(["x", "port.v"])
        >>> montecarlo.add_random_variable(["x", "port.v"], Normal(...))
        >>>
        >>> montecarlo.add_random_variable({
        >>>     "x": Normal(...),
        >>>     "port.v": Uniform(...),
        >>> })
        """
        raise TypeError("Variable name is expected to be a string or an iterable of strings")

    @add_random_variable.register(dict)
    def _add_random_variable_dict(self, specs: dict[str, Distribution], /) -> None:
        """Add several random variables to be randomly perturbated, with specified distributions.
        """
        for varname, distribution in specs.items():
            self.__add_new_random_var(varname, distribution)

    @add_random_variable.register(str)
    def _(self, varname: str, distribution: Optional[Distribution]=None) -> None:
        """Add a variable to be randomly perturbated, following an optional distribution.
        """
        self.__add_new_random_var(varname, distribution)

    @add_random_variable.register(Iterable)
    def _(self, varnames: Iterable[str], distribution: Optional[Distribution]=None) -> None:
        """Add several variables to be randomly perturbated, following an optional distribution.
        """
        for varname in varnames:
            check_arg(varname, f"{varname} in 'varnames'", str)
            self.__add_new_random_var(varname, distribution)
    
    def __add_new_random_var(self, varname: str, distribution: Optional[Distribution]=None) -> None:
        """Add owner[varname] to random variable collection"""
        check_arg(varname, 'varname', str, stack_shift=1)
        self.check_owner_attr(varname)
        self._random_variables[varname] = RandomVariable(self._owner, varname, distribution)

    def add_response(self, name: Union[str, Iterable[str]]) -> None:
        """Add a variable for which the statistical response will be calculated.

        Parameters
        ----------
        name : Union[str, Iterable[str]]
            List of variable names to add
        """
        def add_unique_response_var(name: str):
            self.check_owner_attr(name)
            if name not in self.response_varnames:
                self.response_varnames.append(name)

        check_arg(name, 'name', (str, set, list))

        if isinstance(name, str):
            add_unique_response_var(name)
        else:
            for n in name:
                check_arg(n, f"{n} in 'name'", str)
                add_unique_response_var(n)

    def clear_random_variables(self) -> None:
        """Purge all random variables"""
        self._random_variables.clear()

    def random_variable_data(self) -> dict[str, Distribution]:
        """Builds and returns the dictionary associating distributions to random variable names"""
        return {
            varname: variable.distribution
            for varname, variable in self._random_variables.items()
        }

    @property
    def random_variables(self) -> Iterator[RandomVariable]:
        """Generator yielding all random variable names"""
        return self._random_variables.values()

    @property
    def random_variable_names(self) -> Iterator[str]:
        """Generator yielding all random variable names"""
        return self._random_variables.keys()

    def setup_run(self) -> None:
        """Actions performed prior to the `compute` call."""
        super().setup_run()
        unset_distributions = []
        for varname, random_variable in self._random_variables.items():
            if random_variable.distribution is None:
                unset_distributions.append(varname)
        if unset_distributions:
            varnames = ", ".join(unset_distributions)
            raise ValueError(f"No distribution was specified for {varnames}")

    def _build_cases(self) -> None:
        """Build the list of cases to run during execution
        """
        sobol = qmc.Sobol(d=len(self._random_variables), scramble=False)
        sobol.random()
        self.cases = sobol.random(self.draws)

    def _reset_transients(self):
        """Reattribute initial transient values."""
        owner = self._owner
        for varname, value in self._transients.items():
            setattr(owner, varname, value)

    def _precompute(self):
        """Save reference and build cases."""
        super()._precompute()
        self.run_children()

        self._solver = None
        for child in self.children.values():
            if isinstance(child, AbstractSolver):
                self._solver = child
                self.reference_case_solution = child.save_solution()
                break

        if self.linear:  # precompute linear system
            n_input = len(self._random_variables)
            n_output = len(self.response_varnames)
            if n_output == 0:
                raise ValueError("You need to define response variables to use MonteCarlo linear mode.")

            # Store variable references (faster than calling owner's getattr/setattr)
            name2variable = self._owner.name2variable
            response_variables = [name2variable[name] for name in self.response_varnames]
            random_variables = list(self.random_variables)

            self.X0 = X0 = numpy.array([variable.value for variable in random_variables])
            self.Y0 = Y0 = numpy.array([variable.value for variable in response_variables])
            self.A = A = numpy.zeros((n_output, n_input))

            # Reference for influence matrix computation through center differentiation scheme
            variation = 0.5 * (numpy.max(self.cases, axis=0) - numpy.min(self.cases, axis=0))

            for i, random_variable in enumerate(random_variables):
                random_variable.value = X0[i] + variation[i]
                self.run_children()

                for j, response in enumerate(response_variables):
                    A[j, i] = 0.5 * (response.value - Y0[j]) / variation[i]

                random_variable.value = X0[i] - variation[i]
                self.run_children()

                for j, response in enumerate(response_variables):
                    A[j, i] -= 0.5 * (response.value - Y0[j]) / variation[i]

                # Restore system value
                random_variable.value = X0[i]

            self.Y0 = numpy.array([variable.value for variable in response_variables])
        
        self._reset_transients()

    def _precase(self, case_idx, case) -> None:
        """Hook to be called before running each case.
        
        Parameters
        ----------
        case_idx : int
            Index of the case
        case : Any
            Parameters for this case
        """
        super()._precase(case_idx, case)

        # Set perturbation
        self.perturbations = numpy.zeros(len(self._random_variables))
        for i, variable in enumerate(self._random_variables.values()):
            self.perturbations[i] = variable.add_noise(case[i])

        if (solution := self.reference_case_solution):
            self._solver.load_solution(solution)

    @staticmethod
    def _compute_sequential(mc: MonteCarlo) -> None:
        """Contains the customized `Module` calculation, to execute after children."""
        for index, case in enumerate(mc.cases):
            if len(case) > 0:
                mc._precase(index, case)
                if mc.linear:
                    mc.__run_linear()
                else:
                    mc.run_children()
                mc._postcase(index, case)

    def __run_linear(self) -> None:
        """Approximate MonteCarlo simulation using partial derivatives matrix."""
        # TODO this is not great as we set variables in the system breaking its consistency.
        n_output = len(self.response_varnames)

        if n_output > 0:
            owner = self._owner
            self.X0 = X0 = numpy.array([variable.value for variable in self._random_variables.values()])
            X = numpy.zeros(n_output)
            Y = self.Y0 + numpy.matmul(self.A, X - X0)

            for j, name in enumerate(self.response_varnames):
                setattr(owner, name, Y[j])

    def _postcase(self, index: int, case: Any) -> None:
        """Hook to be called before running each case.
        
        Parameters
        ----------
        case_idx : int
            Index of the case
        case : Any
            Parameters for this case
        """
        # Store the results
        super()._postcase(index, case)

        # Remove the perturbation
        for variable, delta in zip(self._random_variables.values(), self.perturbations):
            if variable.connector is None:
                variable.set_perturbation(-delta)
            else:
                variable.connector.clear_noise()

        self._reset_transients()
