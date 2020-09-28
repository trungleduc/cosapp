import logging
from collections import OrderedDict
from typing import Any, Dict, NoReturn, Optional, Union

import numpy

from cosapp.core.eval_str import AssignString
from cosapp.core.numerics.basics import MathematicalProblem
from cosapp.core.numerics.boundary import Boundary
from cosapp.drivers.iterativecase import IterativeCase
from cosapp.ports.enum import PortType
from cosapp.ports.port import ExtensiblePort
from cosapp.ports.variable import ArrayIndices
from cosapp.systems import System
from cosapp.utils.helpers import check_arg

logger = logging.getLogger(__name__)


class RunSingleCase(IterativeCase):
    """Set new boundary conditions and equations on the system.

    By default, it has a :py:class:`~cosapp.drivers.runonce.RunOnce` driver as child to run the system.

    Attributes
    ----------
    case_values : List[AssignString]
        List of requested variable assignments to set up the case
    initial_values : Dict[str, Tuple[Any, Optional[numpy.ndarray]]]
        List of variables to set with the values to set and associated indices selection

    Parameters
    ----------
    name : str
        Name of the driver
    owner : System, optional
        :py:class:`~cosapp.systems.system.System` to which this driver belong; default None
    **kwargs : Any
        Keyword arguments will be used to set driver options
    """

    __slots__ = ('__case_values', 'offdesign', 'problem')

    def __init__(self,
        name: str,
        owner: Optional[System] = None,
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
        self.__case_values = []  # type: List[AssignString]
            # desc="List of assignments 'lhs <- rhs' to perform in the present case.")
        self.owner = owner
        self.offdesign = MathematicalProblem(self.name + "- offdesign", self.owner)  # type: MathematicalProblem
            # desc="Additional mathematical problem to solve for on this case only.")
        self.problem = None  # type: Optional[MathematicalProblem]
            # desc='Full mathematical problem to be solved on this case.'

    def setup_run(self):
        """Method called once before starting any simulation."""
        super().setup_run()
        
        full_name = lambda name: self.name + name.join('[]')

        def raise_ValueError(kind, name):
            raise ValueError(
                "{} {!r} is defined as design and offdesign {} in driver {!r}".format(
                    kind.capitalize(), name, kind.lower(), self.name))

        self.problem = MathematicalProblem(self.name, self.owner)
        # Add design equations
        for name, unknown in self.design.unknowns.items():
            self.problem.unknowns[name] = unknown
        for name, residue in self.design.residues.items():
            self.problem.residues[full_name(name)] = residue
        # Add off-design equations
        for name, unknown in self.offdesign.unknowns.items():
            if name in self.problem.unknowns:
                raise_ValueError("variable", name)
            self.problem.unknowns[full_name(name)] = unknown
        for name, residue in self.offdesign.residues.items():
            fullname = full_name(name)
            if fullname in self.problem.residues:
                raise_ValueError("equation", name)
            self.problem.residues[fullname] = residue
        # Get common off-design problem to be solved on each case
        common_system = self.owner.get_unsolved_problem()
        # Add common off-design equations taken into account switch in frozen status
        for name, unknown in common_system.unknowns.items():
            if name in self.problem.unknowns:
                raise_ValueError("variable", name)
            # Common unknowns must be duplicated to avoid modification by one point to the others
            self.problem.unknowns[full_name(name)] = unknown.copy()
        for name, residue in common_system.residues.items():
            fullname = full_name(name)
            if fullname in self.problem.residues:
                raise_ValueError("equation", name)
            # Common residues must be duplicated to avoid modification by one point to the others
            self.problem.residues[fullname] = residue.copy()

    def _precompute(self) -> NoReturn:
        """Actions to carry out before the :py:meth:`~cosapp.drivers.runonce.RunOnce.compute` method call.

        It sets the boundary conditions and changes variable status.
        """
        super()._precompute()

        # Set the boundary conditions
        for assignment in self.case_values:
            value, changed = assignment.exec()
            if changed:
                self.owner.set_dirty(PortType.IN)

        # Set the offdesign variables
        for name, unknown in self.get_problem().unknowns.items():
            if name not in self.design.unknowns and not numpy.array_equal(unknown.value, unknown.default_value):
                unknown.set_to_default()

    def clean_run(self):
        """Method called once after any simulation."""
        self.problem = None

    @IterativeCase.owner.setter
    def owner(self, value: Optional[System]) -> NoReturn:
        # Trick to call super setter (see: https://bugs.python.org/issue14965)
        if self.owner is not value:
            if self.owner is not None:
                logger.warning(
                    "System owner of Driver {!r} has changed. Design and offdesign equations have been cleared.".format(
                        self.name))
            self.offdesign = MathematicalProblem(self.offdesign.name, value)
        super(RunSingleCase, RunSingleCase).owner.__set__(self, value)

    def get_problem(self) -> MathematicalProblem:
        """Returns the full mathematical for the case.

        Returns
        -------
        MathematicalProblem
            The full mathematical problem to solve for the case
        """
        if self.problem is None:
            logger.warning("RunSingleCase.get_problem called with no prior call to RunSingleCase.setup_run.")
            return MathematicalProblem(self.name, self.owner)
        else:
            return self.problem

    def set_values(self, modifications: Dict[str, Any]) -> NoReturn:
        """Enter the set of variables defining the case, from a dictionary of the kind {'variable1': value1, ...}
        Note: will erase all previously defined values. Use 'add_values' to append new case values.

        The variable can be contextual `child1.port2.var`. The only rule is that it should belong to
        the owner `System` of this driver or any of its descendants.

        Parameters
        ----------
        modifications : Dict[str, Any]
            Dictionary of (variable name, value)

        Examples
        --------
        >>> driver.set_values({'myvar': 42, 'port.dummy': 'banana'})
        """
        self.clear_values()
        self.add_values(modifications)

    def add_values(self, modifications: Dict[str, Any]) -> NoReturn:
        """Add a set of variables to the list of case values, from a dictionary of the kind {'variable1': value1, ...}

        The variable can be contextual `child1.port2.var`. The only rule is that it should belong to
        the owner `System` of this driver or any of its descendants.

        Parameters
        ----------
        modifications : Dict[str, Any]
            Dictionary of (variable name, value)

        Examples
        --------
        >>> driver.add_values({'myvar': 42, 'port.dummy': 'banana'})
        """
        check_arg(modifications, 'modifications', dict)

        for variable, value in modifications.items():
            self.add_value(variable, value)

    def add_value(self, variable: str, value: Any) -> NoReturn:
        """Add a single variable to list of case values.

        The variable can be contextual `child1.port2.var`. The only rule is that it should belong to
        the owner `System` of this driver or any of its descendants.

        Parameters
        ----------
        variable : str
            Name of the variable
        value : Any
            Value to be used.

        Examples
        --------
        >>> driver.add_value('myvar', 42)
        """
        if self.owner is None:
            raise AttributeError("Driver {!r} must be attached to a System to set case values.".format(self.name))
        else:
            Boundary.parse(self.owner, variable)  # checks that variable is valid
            self.__case_values.append(AssignString(variable, value, self.owner))

    def clear_values(self):
        self.__case_values.clear()

    @property
    def case_values(self):
        return self.__case_values
