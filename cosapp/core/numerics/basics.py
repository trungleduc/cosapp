import logging
from collections import OrderedDict
from numbers import Number
from typing import Any, Dict, Iterable, NoReturn, Optional, Sequence, Union, Tuple

import numpy

from cosapp.core.variableref import VariableReference
from cosapp.core.numerics.boundary import Unknown, TimeUnknown, TimeDerivative
from cosapp.core.numerics.residues import Residue

logger = logging.getLogger(__name__)


class SolverResults:
    """Storage class for solver solution

    Attributes
    ----------
    x : Sequence[float]
        Solution vector
    success : bool
        Whether or not the solver exited successfully.
    message : str
        Description of the cause of the termination.
    fun : ndarray
        Values of objective function.
    jac_lup : (ndarray[float], ndarray[int]), optional
        LU decomposition of Jacobian given as tuple (LU, perm); (None, None) if not available.
    jac : ndarray, optional
        Values of function Jacobian; None if not available.
    jac_errors : dict, optional
        Dictionary with singular Jacobian matrix error indexes
    jac_calls : int
        Number of calls to the Jacobian matrix calculation
    fres_calls : int
        Number of calls to the residues function
    trace : List[Dict[str, Any]]
        Resolution history (if requested) with the evolution of x, residues and jacobian
    """

    def __init__(self):
        self.x = list()  # type: Sequence[float]
        self.success = False  # type: bool
        self.message = ''  # type: str
        self.fun = numpy.array([], dtype=numpy.float)  # type: numpy.ndarray
        self.jac_lup = (None, None)  # type: Optional[(numpy.ndarray, numpy.ndarray)]
        self.jac = None  # type: Optional[numpy.ndarray]
        self.jac_errors = dict()
        self.jac_calls = 0  # type: int
        self.fres_calls = 0  # type: int
        self.trace = list()  # type: List[Dict[str, Any]]


class MathematicalProblem:
    """Container object for unknowns and equations.

    Parameters
    ----------
    name : str
        Name of the mathematical problem
    context : cosapp.systems.System
        Context in which the mathematical problem will be evaluated.
    """

    def __init__(self, name: str, context: 'Optional[cosapp.systems.System]') -> NoReturn:
        # TODO add point label to associate set of equations with Single Case
        self._name = name  # type: str
        self._context = context  # type: Optional[cosapp.systems.System]
        self._unknowns = OrderedDict()  # type: Dict[str, Unknown]
        self._residues = OrderedDict()  # type: Dict[str, Residue]
        self._transients = OrderedDict()  # type: Dict[str, TimeUnknown]
        self._rates = OrderedDict()  # type: Dict[str, TimeDerivative]

    def __str__(self) -> str:
        msg = ""
        if len(self.unknowns) > 0:
            msg += "Unknowns\n"
            for name in self.unknowns:
                msg += "  " + name + "\n"
        if len(self.residues) > 0:
            msg += "Equations\n"
            for residue in self.residues.values():
                msg += "  " + str(residue) + "\n"
        return msg

    def __repr__(self) -> str:
        msg = ""
        if len(self.unknowns) > 0:
            msg += "Unknowns\n"
            for unknown in self.unknowns.values():
                msg += "  " + repr(unknown) + "\n"
        if len(self.residues) > 0:
            msg += "Equations\n"
            for residue in self.residues.values():
                msg += "  " + str(residue) + "\n"
        return msg

    @property
    def name(self) -> str:
        """str : Mathematical system name."""
        return self._name

    @property
    def context(self) -> 'Optional[cosapp.systems.System]':
        """cosapp.systems.System or None : Context in which the mathematical system is evaluated."""
        return self._context

    @context.setter
    def context(self, context: 'Optional[cosapp.systems.System]'):
        if self.context is None:
            self._context = context
        else:
            raise ValueError("Context is already set to '{}'.".format(self.context.name))

    @property
    def residues(self) -> Dict[str, Residue]:
        """Dict[str, Residue] : Residue equations."""
        return self._residues

    @property
    def residues_names(self) -> Tuple[str]:
        """Tuple[str] : Names of residues, flattened to have the same size as `residues_vector`."""
        names = []
        for name, residue in self.residues.items():
            n_values = numpy.size(residue.value)
            if n_values > 1:
                names.extend(['{}[{}]'.format(name, i) for i in range(n_values)])
            else:
                names.append(name)
        return tuple(names)

    @property
    def residues_vector(self) -> numpy.ndarray:
        """numpy.ndarray : Vector of residues."""
        residues = numpy.empty(0)
        for residue in self.residues.values():
            residues = numpy.append(residues, residue.value)
        return residues

    @property
    def shape(self) -> Tuple[int, int]:
        """(int, int) : Number of unknowns and equations."""
        n_unknowns = 0
        for unknown in self.unknowns.values():
            n_unknowns += numpy.size(unknown.value)
        return n_unknowns, self.residues_vector.size

    @property
    def unknowns(self) -> Dict[str, Unknown]:
        """Dict[str, Unknown] : Unknown numerical features defined for this system."""
        return self._unknowns

    @property
    def unknowns_names(self) -> Tuple[str]:
        """Tuple[str] : Names of unknowns flatten to have the same size as `residues_vector`."""
        names = []
        for name, unknown in self.unknowns.items():
            n_values = numpy.size(unknown.value)
            if n_values > 1:
                names.extend(['{}[{}]'.format(name, i) for i in numpy.arange(n_values)[unknown.mask]])
            else:
                names.append(name)
        return tuple(names)

    def add_unknown(
            self,
            name: Union[str, Iterable[Union[dict, str, Unknown]]],
            max_abs_step: Number = numpy.inf,
            max_rel_step: Number = numpy.inf,
            lower_bound: Number = -numpy.inf,
            upper_bound: Number = numpy.inf,
            mask: Optional[numpy.ndarray] = None,
    ) -> "MathematicalProblem":
        """Add unknown variables.

        You can set variable one by one or provide a list of dictionary to set multiple variable at once. The
        dictionary key are the arguments of this method.

        Parameters
        ----------
        name : str or Iterable of dictionary or str
            Name of the variable or list of variable to add
        max_rel_step : float, optional
            Maximal relative step by which the variable can be modified by the numerical solver; default numpy.inf
        max_abs_step : float, optional
            Maximal absolute step by which the variable can be modified by the numerical solver; default numpy.inf
        lower_bound : float, optional
            Lower bound on which the solver solution is saturated; default -numpy.inf
        upper_bound : float, optional
            Upper bound on which the solver solution is saturated; default numpy.inf
        mask : numpy.ndarray or None
            Mask of unknown values in the vector variable

        Returns
        -------
        MathematicalProblem
            The modified MathematicalSystem
        """

        def add_unknown(
            context: 'cosapp.systems.System',
            name: str,
            max_abs_step: Number = numpy.inf,
            max_rel_step: Number = numpy.inf,
            lower_bound: Number = -numpy.inf,
            upper_bound: Number = numpy.inf,
            mask: Optional[numpy.ndarray] = None,
        ):
            """Set numerical limitations on variables.

            Parameters
            ----------
            context: cosapp.systems.System
                Object in which the unknown is defined
            name : str
                Name of the variable
            max_rel_step : float, optional
                Maximal relative step by which the variable can be modified by the numerical solver; default numpy.inf
            max_abs_step : float, optional
                Maximal absolute step by which the variable can be modified by the numerical solver; default numpy.inf
            lower_bound : float, optional
                Lower bound on which the solver solution is saturated; default -numpy.inf
            upper_bound : float, optional
                Upper bound on which the solver solution is saturated; default numpy.inf
            mask : numpy.ndarray or None
                Mask of unknown values in the vector variable
            """
            if name in self._unknowns:
                raise ArithmeticError(
                    "Variable {!r} is defined multiple times as unknown variable in {!r}.".format(name, self.name))

            unknown = Unknown(context, name, max_abs_step, max_rel_step, lower_bound, upper_bound, mask)
            # TODO we have a trouble here if a vector variable is defined as unknown partially multiple time
            #   Example a = [1, 2, 3] with Unknown1 = a[0] & Unknown2 = a[2]
            self._unknowns[unknown.name] = unknown

        if self.context is None:
            raise AttributeError("Owner System is required to define unknowns.")

        if isinstance(name, str):
            add_unknown(self.context, name, max_abs_step, max_rel_step, lower_bound, upper_bound, mask)
        else:
            for unknown in name:
                if isinstance(unknown, Unknown):
                    current_to_context = self.context.get_path_to_child(unknown.context)
                    new_name = '.'.join((current_to_context, name)) if current_to_context else unknown.name
                    if new_name in self._unknowns:
                        logger.warning(
                            "Unknown {!r} already exists in mathematical system {!r}. "
                            "It will be overwritten.".format(new_name, self.name)
                        )
                    self._unknowns[new_name] = unknown
                elif isinstance(unknown, str):
                    add_unknown(self.context, unknown, max_abs_step, max_rel_step, lower_bound, upper_bound, mask)
                else:
                    add_unknown(self.context, **unknown)

        return self

    def add_equation(self,
        equation: Union[str, Iterable[Union[dict, str, Tuple[str, str]]]],
        name: Optional[str] = None,
        reference: Union[Number, numpy.ndarray, str] = 1,
    ) -> "MathematicalProblem":
        """Add residue equation.

        You can add residue equation one by one or provide a list of dictionary to add multiple equation at once. The
        dictionary key are the arguments of this method.

        Parameters
        ----------
        equation : str or Iterable of str of the kind 'lhs == rhs'
            Equation or list of equations to add
        name : str, optional
            Name of the equation; default None => 'lhs == rhs'
        reference : Number, numpy.ndarray or "norm", optional
            Reference value(s) used to normalize the equation; default is 1.
            If value is "norm", actual reference value is estimated from order of magnitude.

        Returns
        -------
        MathematicalProblem
            The modified MathematicalSystem
        """

        if self.context is None:
            raise AttributeError("Owner System is required to define equations.")

        def add_residue(equation, name=None, reference=1):
            """Add residue from equation."""
            residue = Residue(self.context, equation, name, reference)
            self._residues[residue.name] = residue

        if isinstance(equation, str):
            add_residue(equation, name, reference)
        else:
            for eq in equation:
                if isinstance(eq, str):
                    add_residue(eq)
                elif isinstance(eq, dict):
                    add_residue(**eq)
                else:
                    add_residue(*eq)

        return self

    @property
    def transients(self) -> Dict[str, TimeUnknown]:
        """Dict[str, TimeUnknown] : Unknown time-dependent numerical features defined for this system."""
        return self._transients

    def add_transient(self,
        name: str,
        der: Any,
        max_time_step: Union[Number, str] = numpy.inf,
        max_abs_step: Union[Number, str] = numpy.inf,
        pulled_from: Optional[VariableReference] = None,
    ) -> "MathematicalProblem":
        """Add a time-dependent unknown.

        Parameters
        ----------
        name : str
            Name of the new time-dependent (transient) variable
        der : Any
            Name of the variable or evaluable expression defining the time derivative of transient variable
        max_time_step : Number or evaluable expression (str), optional
            Maximal time step for the time integration of the transient variable; numpy.inf by default.
        max_abs_step : Number or evaluable expression compatible with transient variable, optional
            Maximum variable step admissible over one time step; numpy.inf by default.
        pulled_from : VariableReference, optional
            Original time unknown before pulling; None by default.

        Returns
        -------
        MathematicalProblem
            The modified MathematicalSystem
        """

        if self.context is None:
            raise AttributeError("Owner System is required to define time-dependent unknowns.")

        if name in self._transients:
            raise ArithmeticError(f"Variable {name!r} is already defined as a time-dependent unknown of {self.name!r}.")

        self._transients[name] = TimeUnknown(self.context, name, der, max_time_step, max_abs_step, pulled_from=pulled_from)
        return self

    @property
    def rates(self) -> Dict[str, TimeDerivative]:
        """Dict[str, TimeDerivative] : Time derivatives computed during system evolution."""
        return self._rates

    def add_rate(self, name: str, source: Any, initial_value: Any = None) -> "MathematicalProblem":
        """Add a time derivative.

        Parameters
        ----------
        name : str
            Name of the new time-dependent (transient) variable
        source : Any
            Name of the variable or evaluable expression whose time derivative will be computed

        Returns
        -------
        MathematicalProblem
            The modified MathematicalSystem
        """

        if self.context is None:
            raise AttributeError("Owner System is required to define a time derivative.")

        if name in self._rates:
            raise ArithmeticError('Variable "{}" is already defined as a time-dependent unknown of "{}".'.format(name, self.name))

        self._rates[name] = TimeDerivative(self.context, name, source, initial_value)
        return self

    def extend(self, other: 'MathematicalProblem', copy: bool = True) -> 'MathematicalProblem':
        """Extend the current mathematical system with the other one.

        Parameters
        ----------
        other : MathematicalProblem
            The other mathematical system to add
        copy : bool, optional
            Should the objects be copied; default True

        Returns
        -------
        MathematicalProblem
            The resulting mathematical system
        """
        current_to_context = self.context.get_path_to_child(other.context)

        if len(current_to_context) > 0:
            full_path = lambda name: '.'.join((current_to_context, name))
            residue_fullname = lambda name: full_path(name if name.endswith(')') else name.join('()'))
        else:
            full_path = residue_fullname = lambda name: name

        if copy:
            get = lambda obj: obj.copy()
        else:
            get = lambda obj: obj

        def connect(self_dict, other_dict, get_fullname):
            for name, elem in other_dict.items():
                fullname = get_fullname(name)
                if fullname in self_dict:
                    raise ValueError("'{}' already exists in system '{}'.".format(fullname, self.name))
                self_dict[fullname] = get(elem)

        connect(self._unknowns, other.unknowns, full_path)
        connect(self._residues, other.residues, residue_fullname)
        connect(self._transients, other.transients, full_path)
        connect(self._rates, other.rates, full_path)

        return self

    def clear(self) -> NoReturn:
        """Clear all mathematical elements in this problem."""
        self._unknowns.clear()
        self._residues.clear()
        self._transients.clear()
        self._rates.clear()
        # self._extrema.clear()

    def copy(self) -> 'MathematicalProblem':
        """Copy the `MathematicalSystem` object.

        Returns
        -------
        MathematicalProblem
            The duplicated mathematical problem.
        """
        new = MathematicalProblem(self.name, self.context)
        new.extend(self)
        return new

    def to_dict(self) -> Dict[str, Any]:
        """Returns a JSONable representation of the mathematical problem.
        
        Returns
        -------
        Dict[str, Any]
            JSONable representation
        """
        return {
            "unknowns": dict([(name, unknown.to_dict()) for name, unknown in self.unknowns.items()]),
            "equations": dict([(name, equation.to_dict()) for name, equation in self.residues.items()]),
            "transients": dict([(name, transient.to_dict()) for name, transient in self.transients.items()]),
            "rates": dict([(name, rate.to_dict()) for name, rate in self.rates.items()])
        }

    def validate(self) -> NoReturn:
        """Verifies that there are as much unknowns as equations defined.

        Raises
        ------
        ArithmeticError
            If the mathematical system is not closed.
        """
        n_unknowns, n_equations = self.shape
        if n_unknowns != n_equations:
            msg = ('Nonlinear problem {} error: Mismatch between numbers of params [{}] and residues [{}]'
                   ''.format(self.name, n_unknowns, n_equations))
            logger.error(msg)
            logger.error('Residues: {}'.format(list(self.residues)))
            logger.error('Variables: {}'.format(list(self.unknowns)))
            raise ArithmeticError(msg)
