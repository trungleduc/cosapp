from __future__ import annotations
import abc
import logging
import numpy
from itertools import chain
from collections import OrderedDict
from numbers import Number
from dataclasses import dataclass, field
from typing import (
    Any, Union, Iterable, Optional,
    Dict, Tuple, List, Set,
    Callable, NamedTuple,
    TYPE_CHECKING, TypeVar,
)
if TYPE_CHECKING:
    from cosapp.systems import System

from cosapp.core.variableref import VariableReference
from cosapp.core.numerics.boundary import Boundary, Unknown, TimeUnknown, TimeDerivative
from cosapp.core.numerics.residues import Residue, DeferredResidue
from cosapp.core.numerics.utils import TransferHelper
from cosapp.utils.naming import natural_varname
from cosapp.utils.helpers import check_arg
from cosapp.utils.json import jsonify
from cosapp.utils.state_io import object__getstate__


logger = logging.getLogger(__name__)


Self = TypeVar("Self", bound="BaseProblem")
TimeVar = TypeVar("TimeVar", TimeUnknown, TimeDerivative)


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

    def __json__(self) -> Dict[str, Any]:
        """Creates a JSONable dictionary representation of the object.

        Returns
        -------
        Dict[str, Any]
            The dictionary
        """
        return object__getstate__(self).copy()


class WeakDeferredResidue(NamedTuple):
    deferred: DeferredResidue
    weak: bool = False

    @property
    def target(self) -> str:
        """str: targetted quantity"""
        return self.deferred.target

    @property
    def context(self) -> System:
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


class BaseProblem(abc.ABC):
    """Container object for unknowns and equations.

    Parameters
    ----------
    name : str
        Name of the mathematical problem
    context : cosapp.systems.System
        Context in which the mathematical problem will be evaluated.
    """
    def __init__(self, name: str, context: Optional[System]) -> None:
        from cosapp.systems import System
        check_arg(context, 'context', (System, type(None)))
        self._name = name  # type: str
        self._context: System = context

    def __getstate__(self) -> Dict[str, Any]:
        """Creates a state of the object.

        The state type depend on the object, see
        https://docs.python.org/3/library/pickle.html#object.__getstate__
        for further details.

        Returns
        -------
        Dict[str, Any]:
            state
        """
        return object__getstate__(self)

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Sets the object from a provided state.

        Parameters
        ----------
        state : dict[str, Any]
            State
        """
        self.__dict__.update(state)

    def __json__(self) -> Dict[str, Any]:
        """Creates a JSONable dictionary representation of the object.

        Returns
        -------
        Dict[str, Any]
            The dictionary
        """
        state = self.__getstate__().copy()
        state.pop("_context")
        return jsonify(state)
    
    @property
    def name(self) -> str:
        """str : Mathematical system name."""
        return self._name

    @property
    def context(self) -> System:
        """cosapp.systems.System or None: Context in which mathematical objects are evaluated."""
        return self._context

    @context.setter
    def context(self, context: Optional[System]):
        if self._context is None:
            from cosapp.systems import System
            check_arg(context, 'context', (System, type(None)))
            self._context = context
        elif context is not self._context:
            raise ValueError(f"Context is already set to {self._context.name!r}.")

    def _check_context(self, attr_name: str) -> None:
        if self._context is None:
            raise AttributeError(f"Owner System is required to define {attr_name}.")

    @abc.abstractmethod
    def is_empty(self) -> bool:
        pass

    @abc.abstractmethod
    def extend(
        self: Self,
        other: Self,
        copy = True,
        overwrite = False,
        **kwargs,
    ) -> Self:
        """Extend the current problem with another one.

        Parameters
        ----------
        - other [BaseProblem]:
            The other mathematical system to add
        - copy [bool, optional]:
            Should the objects be copied; default is `True`.
        - overwrite [bool, optional]:
            If `False` (default), common attributes raise `ValueError`.
            If `True`, attributes are silently overwritten.

        Returns
        -------
        BaseProblem
            The resulting problem
        """
        return self

    @abc.abstractmethod
    def clear(self) -> None:
        """Clear all mathematical elements in problem."""
        pass

    def copy(self: Self) -> Self:
        """Copy the problem into a new one.

        Returns
        -------
        BaseProblem
            The duplicated problem.
        """
        new = type(self)(self.name, self.context)
        new.extend(self, copy=True)
        return new

    @abc.abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Returns a JSONable representation of the problem.
        
        Returns
        -------
        dict[str, Any]
            JSONable representation
        """
        pass


class MathematicalProblem(BaseProblem):
    """Container object for unknowns and equations.

    Parameters
    ----------
    name : str
        Name of the mathematical problem
    context : cosapp.systems.System
        Context in which the mathematical problem will be evaluated.
    """
    def __init__(self, name: str, context: Optional[System]) -> None:
        super().__init__(name, context)
        # TODO add point label to associate set of equations with Single Case
        self._unknowns = OrderedDict()  # type: Dict[str, Unknown]
        self._residues = OrderedDict()  # type: Dict[str, Residue]
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

    def __eq__(self, other: MathematicalProblem) -> bool:
        try:
            return all(
                getattr(self, name) == getattr(other, name)
                for name in ("_context", "_unknowns", "_residues", "_targets")
            )
        except:
            return False

    @property
    def residues(self) -> Dict[str, Residue]:
        """Dict[str, Residue]: Residue dictionary defined in problem."""
        return self._residues

    @property
    def unknowns(self) -> Dict[str, Unknown]:
        """Dict[str, Unknown]: Unknown dictionary defined in problem."""
        return self._unknowns

    def update_residues(self) -> None:
        """Update the value of all residues"""
        for residue in self._residues.values():
            residue.update()

    def residue_vector(self) -> numpy.ndarray:
        """numpy.ndarray: Residue values stacked into a vector."""
        return self.__as_vector(self._residues)

    def unknown_vector(self):
        """numpy.ndarray: Unknown values stacked into a vector."""
        return self.__as_vector(self._unknowns)

    def __as_vector(self, collection: dict[str, Union[Unknown, Residue]]):
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
            if unknown.is_scalar:
                names.append(unknown.name)
            else:
                basename = unknown.basename
                names.extend(f"{basename}[{i}]" for i in unknown.ref._mask_idx)
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
        return self.n_unknowns == self.n_equations == 0

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
        self._check_context("unknowns")
        context = self._context

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
        self._check_context("equations")
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
        self._check_context("targets")
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

    def extend(self,
        other: MathematicalProblem,
        copy = True,
        overwrite = False,
        unknowns = True,
        equations = True,
        unknown_wrapper: Optional[Callable[[str], str]] = None,
        residue_wrapper: Optional[Callable[[str], str]] = None,
    ) -> MathematicalProblem:
        """Extend the current mathematical problem with another one.

        Parameters
        ----------
        - other [MathematicalProblem]:
            The other mathematical system to add
        - copy [bool, optional]:
            Should the objects be copied; default is `True`.
        - overwrite [bool, optional]:
            If `False` (default), common unknowns/equations raise `ValueError`.
            If `True`, attributes are silently overwritten.
        - unknowns [bool, optional]:
            If `False`, unknowns are discarded; default is `True`.
        - equations [bool, optional]:
            If `False`, equations are discarded; default is `True`.

        Returns
        -------
        MathematicalProblem
            The merged mathematical problem
        """
        no_wrapping = (unknown_wrapper is residue_wrapper is None)
        if other is self and not copy and no_wrapping:
            return self  # quick return

        if unknown_wrapper is None:
            unknown_wrapper = lambda key: key
        if residue_wrapper is None:
            residue_wrapper = lambda key: key

        context: System = self.context
        other_context: System = other.context

        helper = TransferHelper(context)
        var_key_format = TransferHelper.variable_key_format

        def transfer(attr_name: str, transfer_func: Callable, name_wrapper: Callable[[str], str]):
            source: dict = getattr(other, attr_name)
            destination: dict = getattr(self, attr_name)
            transferred = transfer_func(
                source.values(),
                name_wrapper=name_wrapper,
                copy=copy,
            )
            if not overwrite:
                common = set(destination).intersection(transferred)
                if common:
                    if len(common) == 1:
                        message = f"{common.pop()!r} already exists"
                    else:
                        message = f"{sorted(common)} already exist"
                    raise ValueError(f"{message} in {self.name!r}.")
            destination.update(transferred)

        if unknowns:
            transfer('unknowns', helper.transfer_unknowns, unknown_wrapper)

        if equations:
            transfer('residues', helper.transfer_residues, residue_wrapper)

            connectors = list(context.incoming_connectors())
            name2variable = other_context.name2variable
            path = helper.path_finder.get_path(other_context)

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
                                f"Target on {original!r} will be based on {name!r} in the context of {context.full_name()!r}"
                            )
                            break
                self.add_target(name, weak=deferred.weak)

        return self

    def clear(self) -> None:
        """Clear all mathematical elements in this problem."""
        self._unknowns.clear()
        self._residues.clear()
        self._targets.clear()

    def copy(self, activate_targets=False) -> MathematicalProblem:
        """Copy the `MathematicalProblem` object.

        Returns
        -------
        MathematicalProblem
            The duplicated mathematical problem.
        """
        new = super().copy()
        if activate_targets:
            new.activate_targets()
        return new

    def to_dict(self) -> Dict[str, Any]:
        """Returns a JSONable representation of the mathematical problem.
        
        Returns
        -------
        dict[str, Any]
            JSONable representation
        """
        def value_to_dict(items: tuple[str, Union[Unknown, Residue]]):
            return items[0], items[1].to_dict()

        return {
            "unknowns": dict(map(value_to_dict, self.unknowns.items())),
            "equations": dict(map(value_to_dict, self.residues.items()))
        }

    def validate(self) -> None:
        """Verifies that there are as many unknowns as equations defined.

        Raises
        ------
        ArithmeticError
            If the mathematical problem is not square.
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


class TimeProblem(BaseProblem):
    """Container object for time-dependent variables.

    Parameters
    ----------
    name : str
        Name of the mathematical problem
    context : cosapp.systems.System
        Context in which the mathematical problem will be evaluated.
    """
    def __init__(self, name: str, context: Optional[System]) -> None:
        super().__init__(name, context)
        self._transients = OrderedDict()  # type: Dict[str, TimeUnknown]
        self._rates = OrderedDict()  # type: Dict[str, TimeDerivative]

    def __repr__(self) -> str:
        lines = []
        indent = "  "
        if self._transients:
            lines.append(f"Transients")
            lines.extend(
                f"{indent}{transient.name} (derivative: {transient.der})"
                for transient in self._transients.values()
            )
        if self._rates:
            lines.append(f"Rates")
            lines.extend(
                f"{indent}{rate.name} (source: {rate.source_expr})"
                for rate in self._rates.values()
            )
        return "\n".join(lines) if lines else "empty time problem"

    @property
    def transients(self) -> Dict[str, TimeUnknown]:
        """Dict[str, TimeUnknown] : Unknown time-dependent numerical features defined for this system."""
        return self._transients

    def is_empty(self) -> bool:
        return len(self._transients) == len(self._rates) == 0

    def add_transient(self,
        name: str,
        der: Any,
        max_time_step: Union[Number, str] = numpy.inf,
        max_abs_step: Union[Number, str] = numpy.inf,
        pulled_from: Optional[VariableReference] = None,
    ) -> TimeProblem:
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
        self._check_context("transient unknowns")

        if name in self._transients:
            raise ArithmeticError(
                f"Variable {name!r} is already defined as a time-dependent unknown of {self.name!r}."
            )

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

    def add_rate(self, name: str, source: Any, initial_value: Any = None) -> TimeProblem:
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
        self._check_context("rates")

        if name in self._rates:
            raise ArithmeticError(
                f"Variable {name!r} is already defined as a time-dependent unknown of {self.name!r}."
            )

        self._rates[name] = TimeDerivative(self.context, name, source, initial_value)
        return self

    @property
    def n_unknowns(self) -> int:
        """int: Number of unknowns."""
        return sum(
            numpy.size(unknown.value) 
            for unknown in chain(self._transients.values(), self._rates.values())
        )

    def extend(self,
        other: TimeProblem,
        copy = True,
        overwrite = False,
    ) -> TimeProblem:
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
        if other is self and not copy:
            return self  # quick return

        helper = TransferHelper(self.context)

        def transfer_unknowns(attr_name: str):
            """Transfer unknowns from other time problem"""
            source: Dict[str, TimeVar] = getattr(other, attr_name)
            unknowns: Dict[str, TimeVar] = getattr(self, attr_name)
            transferred_unknowns = helper.transfer_unknowns(source.values(), copy=copy)
            if not overwrite:
                common = set(unknowns).intersection(transferred_unknowns)
                if common:
                    raise ValueError(f"{sorted(common)} already exist in {self.name!r}.")
            unknowns.update(transferred_unknowns)

        transfer_unknowns('transients')
        transfer_unknowns('rates')

        return self

    def clear(self) -> None:
        """Clear all mathematical elements in this problem."""
        self._transients.clear()
        self._rates.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Returns a JSONable representation of the mathematical problem.
        
        Returns
        -------
        Dict[str, Any]
            JSONable representation
        """
        def value_to_dict(items: tuple[str, TimeVar]):
            return items[0], items[1].to_dict()

        return {
            "transients": dict(map(value_to_dict, self._transients.items())),
            "rates": dict(map(value_to_dict, self._rates.items()))
        }
