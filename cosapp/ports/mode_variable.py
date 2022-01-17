"""This module defines the basic class encapsulating mode variable attributes."""
import array
import numpy
import logging
import copy
from collections.abc import MutableSequence
from numbers import Number
from typing import Any, Dict, Iterable, Optional, Tuple, Union

from cosapp.ports import units
from cosapp.ports.enum import Scope
from cosapp.utils.helpers import check_arg, is_numerical, get_typename
from cosapp.utils.naming import NameChecker

logger = logging.getLogger(__name__)

RangeValue = Optional[Tuple[Any, Any]]
ArrayIndices = Optional[Union[int, Iterable[int], Iterable[Iterable[int]], slice]]
Types = Optional[Union[Any, Tuple[Any, ...]]]


class ModeVariable:
    """Container for mode variables.

    Parameters
    ----------
    name : str
        Variable name
    port : ModeVarPort, optional
        Port to which the variable belongs
    value : Any
        Variable value
    unit : str, optional
        Variable unit; default empty string (i.e. dimensionless)
    dtype : type or iterable of type, optional
        Variable type; default None (i.e. type of initial value)
    desc : str, optional
        Variable description; default ''
    init : Any, optional
        Value imposed at the beginning of time simulations, if
        variable is an output (unused otherwise).
        If unspecified (default), the variable remains untouched.
    scope : Scope {PRIVATE, PROTECTED, PUBLIC}, optional
        Variable visibility; default PRIVATE
    """

    __slots__ = [
        "__weakref__",
        "_name",
        "_desc",
        "_unit",
        "_port",
        "_init",
        "_dtype",
        "_scope",
    ]

    __name_check = NameChecker(excluded=["inwards", "outwards"])

    def __init__(
        self,
        name: str,
        port: "cosapp.ports.ModeVarPort",
        value: Optional[Any] = None,
        unit: str = "",
        dtype: Types = None,
        desc: str = "",
        init: Optional[Any] = None,
        scope: Scope = Scope.PRIVATE,
    ):
        from cosapp.ports import ModeVarPort
        from cosapp.core.eval_str import EvalString

        self._name = self.name_check(name)

        check_arg(port, "port", ModeVarPort)
        self._port: ModeVarPort = port

        check_arg(unit, 'unit', str)
        if dtype is not None:
            if not isinstance(dtype, type):
                failed = False
                try:
                    for dtype_ in dtype:
                        failed |= not isinstance(dtype_, type)
                except TypeError:
                    failed = True
                if failed:
                    raise TypeError(f"Types must be defined by a type; got {dtype}.")

        check_arg(desc, 'desc', str)
        check_arg(scope, 'scope', Scope)

        init = EvalString(init, port.owner)
        init_value = init.eval()
        if value is None:
            value = init_value
        elif init_value is not None:
            if not isinstance(init_value, type(value)):
                raise ValueError(
                    f"Initial value {init} appears to be inconsistent with arg value={value}."
                )

        if unit and not is_numerical(value):
            unit = ""
            logger.warning(
                f"A physical unit is defined for non-numerical variable {name!r}; it will be ignored."
            )

        elif not units.is_valid_units(unit) and unit:
            raise units.UnitError(f"Unknown unit {unit}.")

        if dtype is None:
            if value is None:
                dtype = None  # can't figure out type if both value and dtype are None
            else:
                dtype = type(value)
                if is_numerical(value):
                    # Force generic number type only if user has not specified any type
                    if issubclass(dtype, Number):
                        dtype = (Number, numpy.ndarray)
                    elif isinstance(value, (MutableSequence, array.ArrayType)):
                        # We have a collection => transform Mutable to ndarray
                        dtype = (MutableSequence, array.ArrayType, numpy.ndarray)
                        value = numpy.asarray(value)

        elif value is not None:
            # Test value has the right type
            if not isinstance(value, dtype):
                typename = get_typename(dtype)
                varname = f"{port.name}.{name}" if port is not None else name
                raise TypeError(
                    "Cannot set {} of type {} with a {}.".format(
                        varname, typename, type(value).__qualname__
                    )
                )

        # TODO: better handle of this possible misunderstanding for users at numpy array instantiation
        if isinstance(value, numpy.ndarray):
            if issubclass(value.dtype.type, numpy.integer):
                logger.warning(
                    f"Variable {name!r} instantiates a numpy array with integer dtype."
                    " This may lead to unpredictible consequences."
                )

        self._unit = unit  # type: str
        self._desc = desc  # type: str
        self._init = init  # type: EvalString
        self._dtype = dtype  # type: Types
        self._scope = scope  # type: Scope

    @classmethod
    def name_check(cls, name: str):
        return cls.__name_check(name)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self._repr_markdown_()
        
    def _repr_markdown_(self) -> str:
        """Returns the representation of this variable in Markdown format.

        Returns
        -------
        str
            Markdown formatted representation
        """
        msg = {"name":f"**{self.name}**" , "unit": f" {self.unit}" if self.unit else ""}
        value = self.value
        try:
            msg["value"] = f"{value:.5g}"
        except:
            msg["value"] = value

        lock_icon = "&#128274;"
        if self.description:
            msg["description"] = f" | {self.description}"
        else:
            msg["description"] = " |"

        scope_format = {
            Scope.PRIVATE: f" {lock_icon*2} ",
            Scope.PROTECTED: f" {lock_icon} ",
            Scope.PUBLIC: "",
        }
        msg["scope"] = scope_format[self.scope]

        return (
            "{name}{scope}: {value!s}{unit}"
            "{description}".format(**msg)
        )  
        
    def __json__(self) -> Dict[str, Any]:
        """JSONable dictionary representing a variable.
        
        Returns
        -------
        Dict[str, Any]
            The dictionary
        """
        return {
            "value": self.value,
        }

    def filter_value(self, value: Any) -> Any:
        if self.dtype == (
            MutableSequence,
            array.ArrayType,
            numpy.ndarray,
        ):
            value = numpy.asarray(value)
        return value

    @property
    def name(self) -> str:
        """str : Variable name"""
        return self._name

    @property
    def contextual_name(self) -> str:
        """str : Join owner port name and mode variable name.

        If the variable has no owner, only its name is returned.
        """
        port = self._port
        return self._name if port is None else f"{port.name}.{self._name}"

    @property
    def full_name(self) -> str:
        return f"{self._port.contextual_name}.{self.name}"

    @property
    def port(self) -> "cosapp.ports.ModeVarPort":
        return self._port

    @property
    def value(self) -> Any:
        return getattr(self._port, self._name)

    @property
    def unit(self) -> str:
        """str : Variable unit; empty string means dimensionless"""
        return self._unit

    @property
    def dtype(self) -> Types:
        """Type[Any] or Tuple of Type[Any] or None : Type of the variable; default None (i.e. type of default value is set)"""
        return self._dtype

    @property
    def description(self) -> str:
        """str : Variable description"""
        return self._desc

    @property
    def scope(self) -> Scope:
        """Scope : Scope of variable visibility"""
        return self._scope

    @property
    def init_expr(self) -> "cosapp.core.eval_str.EvalString":
        """EvalString : expression of initial value"""
        return self._init

    def init_value(self) -> Any:
        """Evaluate and return initial value"""
        return self._init.eval()

    def initialize(self) -> None:
        """Set mode variable to its prescribed initial value."""
        value = self.init_value()
        if value is not None:
            try:
                setattr(self._port, self._name, value)
            except AttributeError:
                pass

    def copy(self, port: "BasePort", name: Optional[str] = None) -> "ModeVariable":
        if name is None:
            name = self.name
        return ModeVariable(
            name,
            value = copy.copy(self.value),
            port = port,
            unit = self._unit,
            dtype = copy.copy(self._dtype),
            desc = self._desc,
            scope = self._scope,
        )

    def to_dict(self) -> Dict:
        """Convert this variable into a dictionary.
   
        Returns
        -------
        dict
            The dictionary representing this variable.
        """
        ret = {
            "value" : self.value,
            "unit": self._unit or None,
            "desc" : self._desc or None,
        } 

        return { key: value for (key, value) in ret.items() if value is not None }
