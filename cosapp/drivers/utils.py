from __future__ import annotations
import enum
from typing import Any, Optional, Union, Dict, Tuple, List, Set, NamedTuple, TYPE_CHECKING
from collections.abc import Collection

from cosapp.core.numerics.boundary import Unknown
from cosapp.core.numerics.basics import MathematicalProblem
from cosapp.utils.helpers import check_arg
from cosapp.utils.parsing import multi_split
from cosapp.utils.state_io import object__getstate__
if TYPE_CHECKING:
    from cosapp.systems import System

import logging
logger = logging.getLogger(__name__)


class SystemAnalyzer:
    """Class containing data collected on a system,
    to be shared between different drivers.
    """
    __slots__ = ('__system',)

    def __init__(self, system: Optional[System]=None):
        self.__reset()
        self.system = system

    def __getstate__(self) -> Union[Dict[str, Any], tuple[Optional[Dict[str, Any]], Dict[str, Any]]]:
        """Creates a state of the object.

        The state type depend on the object, see
        https://docs.python.org/3/library/pickle.html#object.__getstate__
        for further details.
        
        Returns
        -------
        Union[Dict[str, Any], tuple[Optional[Dict[str, Any]], Dict[str, Any]]]:
            state
        """
        return object__getstate__(self)

    @property
    def system(self) -> System:
        "System: system of interest"
        return self.__system

    @system.setter
    def system(self, system) -> None:
        if system is self.__system:
            return
        from cosapp.base import System
        check_arg(system, 'system', (System, type(None)))
        self.__reset(system)

    def __reset(self, system=None) -> None:
        """Reset system and data, with no type check"""
        self.__system = system
        self.clear()

    def clear(self) -> None:
        """Hook function to clear internal data, if any.
        Called at object instanciation.
        """
        pass

    def check_system(self) -> None:
        """Check that object is associated to a system.
        
        Raises
        ------
        `ValueError` if object is not associated to a system.
        """
        if self.__system is None:
            raise ValueError("object is not associated to any system")


def dealias_problem(problem: MathematicalProblem, name=None) -> MathematicalProblem:
    """Resolve unknown aliasing in `problem` due to
    pulling connectors (if any) in context system.

    Parameters:
    -----------
    - problem [MathematicalProblem]:
        Mathematical problem to be de-aliased.
    - name [str, optional]:
        Name of output `MathematicalProblem` object.
    
    Returns:
    --------
    MathematicalProblem:
        New mathematical problem, with identical equations as source
        problem, but de-aliased unknowns.
    """
    if problem.context is None:
        raise ValueError(f"problem is not defined on a system")
    if not name:
        name = f"{problem.name}[filtered]"
    from cosapp.base import System
    context: System = problem.context
    input_mapping = context.input_mapping

    def get_free_unknown(unknown: Unknown) -> Union[Unknown, None]:
        """Checks if `unknown` is aliased by pulling.
        If so, returns alias unknown; else, returns original unknown.
        """
        if unknown.context is context:
            contextual_name = unknown.basename
        else:
            contextual_name = f"{context.get_path_to_child(unknown.context)}.{unknown.basename}"
        try:
            alias = input_mapping[contextual_name]
            var_name = ""
        except KeyError:
            try:
                alias = input_mapping[unknown.portname]
                var_name = ".".join(list(set(unknown.basename.split(".")) - set(unknown.portname.split("."))))
            except KeyError:
                logger.warning(f"Skip connected unknown {contextual_name!r}")
                return None
        aliased = (alias is not unknown.variable_reference)
        if aliased:
            try:
                path = context.get_path_to_child(alias.context)
            except ValueError:
                # `alias.context` is not in `system` tree
                logger.warning(
                    f"Unknown {unknown.contextual_name()!r} is aliased by {alias.contextual_name!r}"
                    f", defined outside the context of {context.name!r}"
                    f"; it is likely to be overwritten after the computation."
                )
            else:
                alias_name = f"{alias.name}.{var_name}" if var_name else alias.name
                alias_contextual_name = f"{path}.{alias.name}" if path else alias_name
                unknown = unknown.transfer(context, alias_contextual_name)
                logger.info(f"Replace unknown {contextual_name!r} by {alias_contextual_name!r}")

        return unknown
    
    # Create output
    filtered = context.new_problem(name)
    # Add equations
    filtered.extend(problem, unknowns=False)
    # Add unknowns
    for key, unknown in problem.unknowns.items():
        free_unknown = get_free_unknown(unknown)
        if free_unknown is None:
            continue
        aliased = (free_unknown is not unknown)
        if aliased:
            key = free_unknown.name
            filtered.unknowns[key] = free_unknown
        else:
            filtered.unknowns[key] = unknown.copy()

    return filtered


class DesignProblemHandler(SystemAnalyzer):
    """Class managing tied design and off-design problems,
    including unknown aliasing.
    """

    __slots__ = ("design", "offdesign")

    def __json__(self) -> Dict[str, Any]:
        """Creates a JSONable dictionary representation of the object.
        
        Break circular dependencies by removing some slots from the 
        state.

        Returns
        -------
        Dict[str, Any]
            The dictionary
        """
        _, slots = self.__getstate__()
        slots.pop("_SystemAnalyzer__system")
        return slots

    def clear(self) -> None:
        """Reset inner problems"""
        self.design = self.new_problem('design')
        self.offdesign = self.new_problem('offdesign')

    @classmethod
    def make(cls, design: MathematicalProblem, offdesign: MathematicalProblem) -> DesignProblemHandler:
        handler = cls(design.context)
        handler.problems = (design, offdesign)
        return handler

    @property
    def problems(self) -> Tuple[MathematicalProblem, MathematicalProblem]:
        """Tuple[MathematicalProblem, MathematicalProblem]: design and off-design problems as a tuple"""
        return self.design, self.offdesign

    @problems.setter
    def problems(self, problems: Tuple[MathematicalProblem, MathematicalProblem]) -> None:
        """Setter for (design, offdesign) tuple"""
        wrong_type = lambda problem: not isinstance(problem, MathematicalProblem)
        if any(map(wrong_type, problems)):
            raise TypeError(
                f"expected `MathematicalProblem` instances; got {tuple(map(type, problems))}."
            )
        self.check_context(*problems)
        self.design, self.offdesign = problems

    def check_context(self, design: MathematicalProblem, offdesign: MathematicalProblem) -> None:
        ok = design.context is offdesign.context is self.system
        if not ok:
            raise ValueError(
                f"Design and off-design problems must be defined in the context of {self.system}."
            )

    def new_problem(self, name: str) -> MathematicalProblem:
        """Create new `MathematicalProblem` instance"""
        return MathematicalProblem(name, self.system)

    def copy_problems(self, prune=True) -> Tuple[MathematicalProblem, MathematicalProblem]:
        """Export design and off-design problems.
        
        Parameters:
        -----------
        - prune, Optional[bool]:
            If `True` (default), resolve unknown aliasing first.
            If `False`, returned problems are copies of object attributes.

        Results:
        --------
        - (design, offdesign): Copies of design and off-design problems,
            as a tuple of `MathematicalProblem` objects.
        """
        if prune:
            design = dealias_problem(self.design, name='design')
            offdesign = dealias_problem(self.offdesign, name='offdesign')
        else:
            design = self.design.copy()
            offdesign = self.offdesign.copy()
        return design, offdesign

    def copy(self, prune=True) -> DesignProblemHandler:
        """Returns a copy of the current object.
        
        Parameters:
        -----------
        - prune, Optional[bool]:
            If `True` (default), resolve unknown aliasing first.
            If `False`, returned handler contains copies of object problems.
        """
        design, offdesign = self.copy_problems(prune)
        return DesignProblemHandler.make(design, offdesign)

    def prune(self) -> None:
        """Remove connected unknowns and resolve unknown aliasing
        in design and off-desing problems.
        """
        self.design, self.offdesign = self.copy_problems(prune=True)

    def merged_problem(self, name="merged", offdesign_prefix="offdesign", copy=True) -> MathematicalProblem:
        """Merge design and off-design problems into a single `MathematicalProblem` instance.

        Parameters
        ----------
        - name [str, optional]:
            Merged problem name (default: 'merged').
        - offdesign_prefix [str, optional]:
            If not empty or `None`, applies a prefix to dict keys in off-design unknowns and equations.
        - copy [bool, optional]:
            Perform copies if `True` (default).
        """
        design, offdesign = self.copy_problems() if copy else self.problems

        def check(attr, kind):
            design_attr = getattr(design, attr)
            offdesign_attr = getattr(offdesign, attr)
            common = set(design_attr).intersection(offdesign_attr)
            if common:
                if len(common) > 1:
                    names = ", ".join(repr(v) for v in sorted(common))
                    names = f"({names}) are"
                    kind += "s"
                else:
                    names = f"{common.pop()!r} is"
                raise ValueError(
                    f"{names} defined as design and off-design {kind}"
                )
        check('unknowns', kind='unknown')
        check('residues', kind='equation')
        
        merged = self.new_problem(name)
        no_rename = lambda name: name
        local_name = (lambda name: f"{offdesign_prefix}[{name}]") if offdesign_prefix else no_rename

        def add_problem(problem: MathematicalProblem, rename_unknowns=True) -> None:
            nonlocal merged
            rename = local_name if rename_unknowns else no_rename

            # Add unknowns
            for name in list(problem.unknowns.keys()):
                merged.unknowns[rename(name)] = problem.unknowns.get(name)
            
            # Add residues
            for name in list(problem.residues.keys()):
                merged.residues[local_name(name)] = problem.residues.get(name)

            for name, residue in problem.get_target_residues().items():
                merged.residues[local_name(name)] = residue

        add_problem(design, rename_unknowns=False)
        add_problem(offdesign, rename_unknowns=True)
        return merged

    def extend(self, other: DesignProblemHandler, prune=True, copy=True, overwrite=False) -> DesignProblemHandler:
        """Extend both design and off-design problems from `other` handler.
        
        Parameters
        ----------
        - prune [bool, optional]:
            If `True` (default), added problems are pruned before being added.
        - copy [bool, optional]:
            Determines whether problem copies should be made before extension.
        - overwrite [bool, optional]:
            Overwrite option, forwarded to `MathematicalProblem.extend`.
        """
        if prune or copy:
            design, offdesign = other.copy_problems(prune)
        else:
            design, offdesign = other.problems
        # Extend inner problems; copy is unnecessary at this point,
        # as `design` and `offdesign` are consistent with argument `copy`.
        options = dict(copy=False, overwrite=overwrite)
        self.design.extend(design, **options)
        self.offdesign.extend(offdesign, **options)
        return self


@enum.unique
class ConstraintType(enum.Enum):
    """Enum covering constraint types"""
    GE = {
        'operator': ">=",
        'sort': (lambda lhs, rhs: (lhs, rhs)),
    }
    LE = {
        'operator': "<=",
        'sort': (lambda lhs, rhs: (rhs, lhs)),
    }
    EQ = {
        'operator': "==",
        'sort': (lambda lhs, rhs: (lhs, rhs)),
    }

    def expression(self, lhs: str, rhs: str) -> str:
        return "{} - ({})".format(*self.sort(lhs, rhs))

    def sort(self, lhs: str, rhs: str) -> Tuple[str, str]:
        return self.value['sort'](lhs, rhs)

    def __str__(self) -> str:
        return self.description

    @property
    def operator(self) -> str:
        return self.value['operator']

    @property
    def description(self) -> str:
        return f"Constraint of the kind `lhs {self.operator} rhs`"

    @property
    def is_inequality(self) -> bool:
        return self.operator != "=="


class Constraint(NamedTuple):
    """Named tuple representing a non-negative constraint
    of the kind `lhs <op> rhs`, where `<op>` is either
    `==` (equality) or `>=` (inequality constraint),
    depending on Boolean attribute `is_inequality`.

    Attributes:
    -----------
    - lhs [str]: left-hand side.
    - rhs [str]: right-hand side.
    - is_inequality [bool].

    Properties:
    -----------
    - expression [str]: non-negative constraint `lhs - rhs`.
    """
    lhs: str
    rhs: str
    is_inequality: bool = True

    @property
    def expression(self) -> str:
        return f"{self.lhs} - ({self.rhs})"

    def __str__(self) -> str:
        op = ">=" if self.is_inequality else "=="
        return f"{self.expression} {op} 0"


class ConstraintParser:

    @classmethod
    def parse(cls, expression: Union[str, List[str]]) -> Set[Constraint]:
        """Parse a string expression or a list thereof as a
        set of non-negative constraints to be used in solvers.

        Parameters:
        -----------
        - expression [str or List[str]]:
            Human-readable equality or inequality constraints,
            such as 'x >= y', '0 < alpha < 1', or a list thereof.
        
        Returns:
        --------
        - constraints [Set[Constraint]]:
            Set of `Constraint` named tuple objects.
        """
        check_arg(expression, 'expression', (str, Collection))
        ctypes = cls.types()
        not_in_sides = list(ctypes) + ["="]
        
        def side_ok(side: str) -> bool:
            return not any(nogo in side for nogo in not_in_sides) and side.strip()

        def parse_single(expression: str) -> Set[Constraint]:
            check_arg(expression, 'expression', str)
            constraints = set()

            for operator in "<>":
                expression = expression.replace(f"{operator}=", operator)

            expressions, operators = multi_split(expression, ctypes.keys())

            for lhs, rhs, operator in zip(expressions, expressions[1:], operators):
                ok = side_ok(lhs) and side_ok(rhs)
                ctype = ctypes[operator]
                constraint = Constraint(
                    *ctype.sort(lhs, rhs),
                    ctype.is_inequality,
                )
                if not ok:
                    raise ValueError(f"Invalid constraint {constraint.expression}")
                constraints.add(constraint)

            return constraints
        
        expressions = [expression] if isinstance(expression, str) else expression

        constraints = set()
        for expression in expressions:
            constraints |= parse_single(expression)

        return constraints

    @classmethod
    def types(cls) -> Dict[str, ConstraintType]:
        return {
            ">" : ConstraintType.GE,
            "<" : ConstraintType.LE,
            "<=": ConstraintType.LE,
            "==": ConstraintType.EQ,
            ">=": ConstraintType.GE,
        }
