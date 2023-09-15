from __future__ import annotations
import logging
from collections import OrderedDict
from numbers import Number
from dataclasses import dataclass, field
from typing import (
    Any, Dict, Iterable, Optional,
    Sequence, Union, Tuple, List,
    Set, Callable, NamedTuple,
)

import numpy

from cosapp.core.variableref import VariableReference
from cosapp.core.numerics.boundary import Boundary, Unknown, TimeUnknown, TimeDerivative
from cosapp.core.numerics.residues import Residue, DeferredResidue
from cosapp.utils.naming import natural_varname
from cosapp.utils.helpers import check_arg

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
    - fun [numpy.ndarray[float]]:
        Values of the objective function.
    - success [bool]:
        Whether or not the solver exited successfully.
    - message [str]:
        Description of the cause of the termination.
    - tol [float, optional]:
        Tolerance level; `NaN` if not available.
    - jac [numpy.ndarray[float], optional]
        Values of function Jacobian; None if not available.
    - jac_lup [tuple(numpy.ndarray[float], numpy.ndarray[int]), optional]
        LU decomposition of Jacobian given as tuple (LU, perm); (None, None) if not available.
    - jac_calls [int]:
        Number of calls to the Jacobian matrix calculation.
    - fres_calls [int]:
        Number of calls to the residues function.
    - jac_errors [dict, optional]:
        Dictionary with singular Jacobian matrix error indexes.
    - trace [list[dict[str, Any]]]:
        Resolution history (if requested) with the evolution of x, residues and jacobian.
    """
    x: numpy.ndarray = field(default_factory=default_array_factory())
    fun: numpy.ndarray = field(default_factory=default_array_factory())
    success: bool = False
    message: str = ''
    tol: float = numpy.nan
    jac: Optional[numpy.ndarray] = None
    jac_lup: Optional[Tuple[numpy.ndarray, numpy.ndarray]] = (None, None)
    jac_calls: int = 0
    fres_calls: int = 0
    jac_errors: dict = field(default_factory=dict)
    trace: List[Dict[str, Any]] = field(default_factory=list)


class WeakDeferredResidue(NamedTuple):
    deferred: DeferredResidue
    weak: bool = False

    @property
    def target(self) -> str:
        """str: targetted quantity"""
        return self.deferred.target

    @property
    def context(self):
        """System: evaluation context of residue"""
        return self.deferred.context

    @property
    def variables(self) -> Set[str]:
        """Set[str]: names of variables involved in residue"""
        return self.deferred.variables

    def target_value(self) -> Any:
        """Evaluates and returns current value of target"""
        return self.deferred.target_value()

    def equation(self) -> str:
        """Returns target equation with updated lhs value"""
        return self.deferred.equation()

    def make_residue(self, reference=None) -> Residue:
        """Generates the residue corresponding to equation 'target == value(target)'"""
        return self.deferred.make_residue(reference)


class MathematicalProblem:
    """Container object for unknowns and equations.

    Parameters
    ----------
    name : str
        Name of the mathematical problem
    context : cosapp.systems.System
        Context in which the mathematical problem will be evaluated.
    """
    def __init__(self, name: str, context: Optional["System"]) -> None:
        from cosapp.systems import System
        check_arg(context, 'context', (System, type(None)))
        # TODO add point label to associate set of equations with Single Case
        self._name = name  # type: str
        self._context: System = context
        self._unknowns = OrderedDict()  # type: Dict[str, Unknown]
        self._residues = OrderedDict()  # type: Dict[str, Residue]
        self._transients = OrderedDict()  # type: Dict[str, TimeUnknown]
        self._rates = OrderedDict()  # type: Dict[str, TimeDerivative]
        self._targets = OrderedDict()  # type: Dict[str, WeakDeferredResidue]

    def __repr__(self) -> str:
        lines = []
        indent = "  "
        def format_unknown(items: Tuple[str, Unknown]) -> str:
            key, unknown = items
            value = unknown.default_value
            if value is None:
                value = unknown.value
            return f"{indent}{key} = {value}"

        if self.unknowns:
            lines.append(f"Unknowns [{self.n_unknowns}]")
            lines.extend(
                map(format_unknown, self.unknowns.items())
            )
        if self.residues or self.deferred_residues:
            lines.append(f"Equations [{self.n_equations}]")
            lines.extend(
                f"{indent}{key} := {residue.value}"
                for key, residue in self.residues.items()
            )
            lines.extend(
                f"{indent}{deferred.equation()} (target)"
                for deferred in self.deferred_residues.values()
            )
        return "\n".join(lines) if lines else "empty problem"

    @property
    def name(self) -> str:
        """str : Mathematical system name."""
        return self._name

    @property
    def context(self) -> Optional['cosapp.systems.System']:
        """cosapp.systems.System or None: Context in which mathematical objects are evaluated."""
        return self._context

    @context.setter
    def context(self, context: Optional['cosapp.systems.System']):
        if self._context is None:
            self._context = context
        elif context is not self._context:
            raise ValueError(f"Context is already set to {self._context.name!r}.")

    @property
    def residues(self) -> Dict[str, Residue]:
        """Dict[str, Residue]: Residue dictionary defined in problem."""
        return self._residues

    @property
    def unknowns(self) -> Dict[str, Unknown]:
        """Dict[str, Unknown]: Unknown dictionary defined in problem."""
        return self._unknowns

    def residue_vector(self) -> numpy.ndarray:
        """numpy.ndarray: Residue values stacked into a vector."""
        return self.__as_vector(self.residues)

    def unknown_vector(self):
        """numpy.ndarray: Unknown values stacked into a vector."""
        return self.__as_vector(self.unknowns)

    def __as_vector(self, collection: dict):
        values = tuple(
            numpy.ravel(element.value) for element in collection.values()
        )
        return numpy.concatenate(values) if values else numpy.empty(0)

    def residue_names(self) -> Tuple[str]:
        """Tuple[str]: Names of residues, flattened to have the same size as `residue_vector()`."""
        names = []
        for name, residue in self.residues.items():
            n_values = numpy.size(residue.value)
            if n_values > 1:
                names.extend(f"{name}[{i}]" for i in range(n_values))
            else:
                names.append(name)
        return tuple(names)

    def unknown_names(self) -> Tuple[str]:
        """Tuple[str]: Names of unknowns flatten to have the same size as `unknown_vector()`."""
        names = []
        for unknown in self.unknowns.values():
            if unknown.mask is None:
                names.append(unknown.name)
            else:
                basename = unknown.basename
                ref_size = numpy.size(unknown.ref.value)
                names.extend(
                    f"{basename}[{i}]" for i in numpy.arange(ref_size)[unknown.mask.flatten()]
                )
        return tuple(names)

    @property
    def n_unknowns(self) -> int:
        """int: Number of unknowns."""
        return sum(
            numpy.size(unknown.value) 
            for unknown in self.unknowns.values()
        )

    @property
    def n_equations(self) -> int:
        """int: Number of equations (including deferred equations)."""
        n_equations = sum(
            numpy.size(residue.value)
            for residue in self.residues.values()
        )
        n_equations += sum(
            numpy.size(deferred.target_value())
            for deferred in self.deferred_residues.values()
        )
        return n_equations

    @property
    def shape(self) -> Tuple[int, int]:
        """(int, int) : Number of unknowns and equations."""
        return (self.n_unknowns, self.n_equations)

    def is_empty(self) -> bool:
        return self.shape == (0, 0)

    def add_unknown(self,
        name: Union[str, Iterable[Union[dict, str, Unknown]]],
        max_abs_step: Number = numpy.inf,
        max_rel_step: Number = numpy.inf,
        lower_bound: Number = -numpy.inf,
        upper_bound: Number = numpy.inf,
        mask: Optional[numpy.ndarray] = None,
    ) -> MathematicalProblem:
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
        self.__check_context("unknowns")
        context = self.context

        def add_unknown(
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
            # TODO we have a problem here if a vector variable is defined as unknown partially multiple times
            #   Example a = [1, 2, 3] with Unknown1 = a[0] & Unknown2 = a[2]

            unknown = Unknown(context, name, max_abs_step, max_rel_step, lower_bound, upper_bound, mask)
            if unknown.name in self._unknowns:
                logger.info(
                    f"Variable {name!r} is already declared as unknown in {self.name!r}."
                )
            else:
                self._unknowns[unknown.name] = unknown

        params = (max_abs_step, max_rel_step, lower_bound, upper_bound, mask)

        if isinstance(name, str):
            add_unknown(name, *params)
        else:
            for unknown in name:
                if isinstance(unknown, Unknown):
                    current_to_context = context.get_path_to_child(unknown.context)
                    new_name = f"{current_to_context}.{name}" if current_to_context else unknown.name
                    if new_name in self._unknowns:
                        logger.warning(
                            "Unknown {!r} already exists in mathematical system {!r}. "
                            "It will be overwritten.".format(new_name, self.name)
                        )
                    self._unknowns[new_name] = unknown
                elif isinstance(unknown, str):
                    add_unknown(unknown, *params)
                else:
                    add_unknown(**unknown)

        return self

    def add_equation(self,
        equation: Union[str, Iterable[Union[dict, str, Tuple[str, str]]]],
        name: Optional[str] = None,
        reference: Union[Number, numpy.ndarray, str] = 1,
    ) -> MathematicalProblem:
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
        self.__check_context("equations")
        context = self.context

        def add_residue(equation, name=None, reference=1):
            """Add residue from equation."""
            residue = Residue(context, equation, name, reference)
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

    def add_target(self,
        expression: Union[str, Iterable[str]],
        reference: Union[Number, numpy.ndarray, str] = 1,
        weak = False,
    ) -> MathematicalProblem:
        """Add deferred equation.

        Parameters
        ----------
        expression: str
            Targetted expression
        reference : Number, numpy.ndarray or "norm", optional
            Reference value(s) used to normalize the (deferred) equation; default is 1.
            If value is "norm", actual reference value is estimated from order of magnitude.
        weak: bool, optional
            If True, the target is disregarded if the corresponding variable is connected; default is `False`.

        Returns
        -------
        MathematicalProblem
            The modified MathematicalSystem
        """
        self.__check_context("targets")
        context = self.context

        def register(name, reference=1):
            deferred = DeferredResidue(context, name, reference)
            if len(deferred.variables) > 1:
                raise NotImplementedError(
                    f"Targets are only supported for single variables; got {deferred.variables}"
                )
            key = self.target_key(deferred.target)
            self._targets[key] = WeakDeferredResidue(deferred, weak)

        if isinstance(expression, str):
            register(expression, reference)
        else:
            for target in expression:
                register(target)

        return self

    def get_target_equations(self) -> List[str]:
        return [deferred.equation() for deferred in self._targets.values()]

    def get_target_residues(self) -> Dict[str, Residue]:
        return dict(
            (key, deferred.make_residue())
            for key, deferred in self._targets.items()
        )

    def activate_targets(self) -> None:
        """Activate deferred residues (targets) and incorporate them
        in the mathematical problem residue list.
        Warning: This operation is irreversible, as targets are purged.
        """
        targets = self._targets
        residues = self._residues
        for key in list(targets):
            deferred = targets.pop(key)
            residue = deferred.make_residue()
            new_key = self.target_key(residue.name)
            residues[new_key] = residue

    @staticmethod
    def target_key(target: str) -> str:
        """Returns dict key to be used for targetted quantity `target`"""
        return f"{target} (target)"

    @property
    def deferred_residues(self) -> Dict[str, WeakDeferredResidue]:
        """Dict[str, WeakDeferredResidue]: Dict of deferred residues defined for this system."""
        return self._targets

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
    ) -> MathematicalProblem:
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
        self.__check_context("transient unknowns")

        if name in self._transients:
            raise ArithmeticError(f"Variable {name!r} is already defined as a time-dependent unknown of {self.name!r}.")

        self._transients[name] = TimeUnknown(
            self.context,
            name,
            der,
            max_time_step,
            max_abs_step,
            pulled_from=pulled_from,
        )
        return self

    @property
    def rates(self) -> Dict[str, TimeDerivative]:
        """Dict[str, TimeDerivative] : Time derivatives computed during system evolution."""
        return self._rates

    def add_rate(self, name: str, source: Any, initial_value: Any = None) -> MathematicalProblem:
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
        self.__check_context("rates")

        if name in self._rates:
            raise ArithmeticError(f"Variable {name!r} is already defined as a time-dependent unknown of {self.name!r}.")

        self._rates[name] = TimeDerivative(self.context, name, source, initial_value)
        return self

    def __check_context(self, name: str):
        if self.context is None:
            raise AttributeError(f"Owner System is required to define {name}.")

    def extend(self,
        other: MathematicalProblem,
        copy = True,
        unknowns = True,
        equations = True,
        overwrite = False,
        unknown_wrapper: Optional[Callable[[str], str]] = None,
        residue_wrapper: Optional[Callable[[str], str]] = None,
    ) -> MathematicalProblem:
        """Extend the current mathematical system with the other one.

        Parameters
        ----------
        - other [MathematicalProblem]:
            The other mathematical system to add
        - copy [bool, optional]:
            Should the objects be copied; default is `True`.
        - unknowns [bool, optional]:
            If `False`, unknowns are discarded; default is `True`.
        - equations [bool, optional]:
            If `False`, equations are discarded; default is `True`.
        - overwrite [bool, optional]:
            If `False` (default), common unknowns/equations raise `ValueError`.
            If `True`, attributes are silently overwritten.

        Returns
        -------
        MathematicalProblem
            The resulting mathematical system
        """
        no_wrapping = (unknown_wrapper is residue_wrapper is None)
        if other is self and not copy and no_wrapping:
            return self  # quick return

        if unknown_wrapper is None:
            unknown_wrapper = lambda key: key
        if residue_wrapper is None:
            residue_wrapper = lambda key: key

        from cosapp.systems import System
        context: System = self.context

        sys_paths: Dict[int, str] = dict()

        def get_path(system: System) -> str:
            nonlocal sys_paths
            key = id(system)
            try:
                path = sys_paths[key]
            except KeyError:
                sys_paths[key] = path = context.get_path_to_child(system)
            return path

        def var_key_format(path: str, varname: str):
            return f"{path}.{varname}" if path else varname

        def res_key_format(path: str, eqname: str):
            return f"{path}: {eqname}" if path else eqname

        def make_key(obj: Union[Boundary, Residue], key_format, wrapper) -> str:
            """Generic key formatter for unknowns and residues."""
            path = get_path(obj.context)
            key = key_format(path, obj.name)
            return wrapper(key)

        def variable_key(variable: Boundary) -> str:
            """Generate dict key from `variable` context."""
            return make_key(variable, var_key_format, unknown_wrapper)

        def residue_key(residue: Residue) -> str:
            """Generate dict key from `residue` context."""
            return make_key(residue, res_key_format, residue_wrapper)

        get = (lambda obj: obj.copy()) if copy else (lambda obj: obj)

        def transfer_unknowns(kind: str):
            """Transfer unknowns from other mathematical problem"""
            source = getattr(other, kind)
            destination = getattr(self, kind)
            for unknown in source.values():
                key = variable_key(unknown)
                if not overwrite and key in destination:
                    raise ValueError(f"{key!r} already exists in {self.name!r}.")
                destination[key] = get(unknown)

        if unknowns:
            transfer_unknowns('unknowns')
            transfer_unknowns('transients')
            transfer_unknowns('rates')

        if equations:
            residues = self._residues
            for name, residue in other.residues.items():
                key = residue_key(residue)
                if not overwrite and key in residues:
                    raise ValueError(f"{key!r} already exists in {self.name!r}.")
                residues[key] = get(residue)

            connectors = list(self.context.incoming_connectors())
            name2variable = other.context.name2variable
            path = get_path(other.context)

            for deferred in other._targets.values():
                targetted = list(deferred.variables)[0]
                name = deferred.target.replace(targetted, var_key_format(path, targetted))  # default
                ref = name2variable[targetted]
                port, varname = ref.mapping, ref.key
                # Check if targetted variable is a pulled output
                if port.is_output:
                    for connector in connectors:
                        pulled = (
                            connector.source is port
                            and varname in connector.source_variables()
                        )
                        if pulled:
                            alias_name = natural_varname(
                                f"{connector.sink.name}.{connector.sink_variable(varname)}"
                            )
                            original = name
                            if deferred.target == targetted:
                                name = alias_name
                            else:
                                # target is an expression involving `targetted`
                                name = name.replace(var_key_format(path, targetted), alias_name)
                            logger.info(
                                f"Target on {original!r} will be based on {name!r} in the context of {self.context.full_name()!r}"
                            )
                            break
                self.add_target(name, weak=deferred.weak)

        return self

    def clear(self) -> None:
        """Clear all mathematical elements in this problem."""
        self._unknowns.clear()
        self._residues.clear()
        self._transients.clear()
        self._rates.clear()
        self._targets.clear()

    def copy(self, activate_targets=False) -> MathematicalProblem:
        """Copy the `MathematicalSystem` object.

        Returns
        -------
        MathematicalProblem
            The duplicated mathematical problem.
        """
        new = MathematicalProblem(self.name, self.context)
        new.extend(self, copy=True)
        if activate_targets:
            new.activate_targets()
        return new

    def to_dict(self) -> Dict[str, Any]:
        """Returns a JSONable representation of the mathematical problem.
        
        Returns
        -------
        Dict[str, Any]
            JSONable representation
        """
        return {
            "unknowns": dict((name, unknown.to_dict()) for name, unknown in self.unknowns.items()),
            "equations": dict((name, equation.to_dict()) for name, equation in self.residues.items()),
            "transients": dict((name, transient.to_dict()) for name, transient in self.transients.items()),
            "rates": dict((name, rate.to_dict()) for name, rate in self.rates.items())
        }

    def validate(self) -> None:
        """Verifies that there are as much unknowns as equations defined.

        Raises
        ------
        ArithmeticError
            If the mathematical system is not closed.
        """
        n_unknowns, n_equations = self.shape
        if n_unknowns != n_equations:
            msg = "Nonlinear problem {} error: Mismatch between numbers of params [{}] and residues [{}]".format(
                self.name, n_unknowns, n_equations
            )
            logger.error(msg)
            logger.error(f"Residues: {list(self.residues) + list(self.deferred_residues)}")
            logger.error(f"Variables: {list(self.unknowns)}")
            raise ArithmeticError(msg)
