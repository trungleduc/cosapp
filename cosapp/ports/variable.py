"""This module define the basic class encapsulating a variable attributes."""
import array
import logging
from collections.abc import MutableSequence
from numbers import Number
import re, inspect
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np

from cosapp.core.numerics.distributions.distribution import Distribution
from cosapp.ports import units
from cosapp.ports.enum import Scope, Validity, RangeType
from cosapp.utils.helpers import check_arg, is_numerical, is_number, get_typename
from cosapp.utils.naming import NameChecker

logger = logging.getLogger(__name__)

RangeValue = Optional[Tuple[Any, Any]]
ArrayIndices = Optional[Union[int, Iterable[int], Iterable[Iterable[int]], slice]]
Types = Optional[Union[Any, Tuple[Any, ...]]]


class Variable:
    """Variable details container.

    The `valid_range` defines the range of value for which a model is known to behave correctly.
    The `limits` are at least as large as the validity range.

    Parameters
    ----------
    name : str
        Variable name.
    port : ExtensiblePort
        Port to which belong the variable
    value : Any
        Variable value
    unit : str, optional
        Variable unit; default empty string (i.e. dimensionless)
    dtype : type or iterable of type, optional
        Variable type; default None (i.e. type of initial value)
    valid_range : Tuple[Any, Any] or Tuple[Tuple], optional
        Validity range of the variable; default None (i.e. all values are valid).
        Tuple[Any, Any] in case of scalar value, tuple of tuples in case of vector value.
    invalid_comment : str, optional
        Comment to show in case the value is not valid; default ''
    limits : Tuple[Any, Any] or Tuple[Tuple], optional
        Limits over which the use of the model is wrong; default valid_range.
        Tuple[Any, Any] in case of scalar value, tuple of tuples in case of vector value.
    out_of_limits_comment : str, optional
        Comment to show in case the value is not valid; default ''
    desc : str, optional
        Variable description; default ''
    distribution : Distribution, optional
        Variable random distribution; default None (no distribution)
    scope : Scope {PRIVATE, PROTECTED, PUBLIC}, optional
        Variable visibility; default PRIVATE
    """

    # Value cannot be integrated in this object
    #   otherwise value getter execution time *200
    #   another advantage is that the metadata can be change/shared without providing
    #     a way to change the value no passing through the Port (which is bad for the
    #     clean dirty logic).
    #   Disadvantage : the logic to store the details and the value is a bit messy in
    #     ExtensiblePort.add_variable

    __slots__ = [
        "__weakref__",
        "_desc",
        "_distribution",
        "_invalid_comment",
        "_limits",
        "_name",
        "_port",
        "_out_of_limits_comment",
        "_scope",
        "_dtype",
        "_unit",
        "_valid_range",
    ]

    __name_check = NameChecker()

    @staticmethod
    def _get_limits_from_type(variable: Any) -> RangeValue:
        """Get default limits for a variable depending of its type.

        Parameters
        ----------
        variable : Any
            Variable of interest

        Returns
        -------
        Tuple[float, float] or None
            Default (lower, upper) limits
        """
        if is_numerical(variable):
            return -np.inf, np.inf
        else:
            return None

    @staticmethod
    def check_range_type( value : Iterable, name : Optional[str] = None) -> RangeType:
        """Get type of valid_range of limits of variable.
        This function check if `value` is a 2-tuple of scalar or
        a tuple of tuples
        
        Parameters
        ----------
        value : Any
            value need to be checked

        Returns
        -------
        int (from enum RangeType)
            Type of `value`
        """

        tuple_check = True
        value_check = True

        if isinstance(value, Number):
            raise TypeError(
                "Validity or limit range must be a tuple with format comparable to value"
            )
        if value is not None:
            for bound in value :
                if isinstance(bound, (tuple,list)):
                    value_check = False
                elif isinstance(bound, Number) or bound is None: 
                    tuple_check = False
                else:
                    value_check = False
                    tuple_check = False        
        else :
            return RangeType.NONE
        if value_check and not tuple_check:
            if len(value) != 2:
                raise TypeError(
                    "Valid range or limits must be a 2-tuple with type comparable to value"
                )
            return RangeType.VALUE
        elif tuple_check and not value_check:
            return RangeType.TUPLE
        else :
            raise ValueError(
                f"Mixed values in valid_range object {value} of {name}."
                " Valid object can contain only numerical values or only tuples"
            ) 

    @staticmethod
    def _check_range(
        limits: RangeValue, valid_range: RangeValue, value: Any, name : str = None
    ) -> Tuple[RangeValue, RangeValue]:
        """Correct coherence of limits and validation range depending on value type.

        Parameters
        ----------
        limits : Tuple[Any, Any] or Tuple[Tuple] or None
            (lower, upper) limits
        valid_range : Tuple[Any, Any] Tuple[Tuple] or None
            (lower, upper) validation range
        value : Any

        Returns
        -------
        Tuple[Tuple[Any, Any] or Tuple[Tuple] or None, Tuple[Any, Any] or Tuple[Tuple] or None]
            Tuple of corrected (limits, validation range)
        """
        
        default = Variable._get_limits_from_type(value)
        range_type = Variable.check_range_type(valid_range, name)
        limits_type = Variable.check_range_type(limits, name)
        
        if default is not None:
            if limits is None:
                limits = (None, None)

            if valid_range is None:
                valid_range = limits

            if range_type == RangeType.VALUE:
                min_range, max_range = valid_range
                if min_range is None:
                    min_range = default[0]
                if max_range is None:
                    max_range = default[1]

                if min_range > max_range:
                    min_range, max_range = max_range, min_range

                valid_range = (min_range, max_range)

                if limits_type == RangeType.VALUE:
                    min_limit, max_limit = limits

                    if min_limit is None:
                        min_limit = default[0]
                    if max_limit is None:
                        max_limit = default[1]

                    if min_limit > max_limit:
                        min_limit, max_limit = max_limit, min_limit

                    limits = (np.minimum(min_range, min_limit), np.maximum(max_range, max_limit))
                elif limits_type == RangeType.NONE:
                    limits = (default[0], default[1])
                else:
                    raise ValueError(
                        f"valid_range {valid_range} and limits {limits} of object {name} do not take the same format")

            elif range_type == RangeType.TUPLE:
                new_range = []
                for range_item in valid_range : 
                    min_range, max_range = range_item
                    if min_range is None:
                        min_range = default[0]
                    if max_range is None:
                        max_range = default[1]
                    if min_range > max_range:
                        min_range, max_range = max_range, min_range

                    new_range.append((min_range, max_range)) 
                valid_range =tuple(new_range)
                if limits_type == RangeType.TUPLE:
                    new_limit = []
                    for limit_item in limits:
                        min_limit, max_limit = limit_item
                        if min_limit is None:
                            min_limit = default[0]
                        if max_limit is None:
                            max_limit = default[1]

                        if min_limit > max_limit:
                            min_limit, max_limit = max_limit, min_limit    

                        new_limit.append((min_limit, max_limit))
                    limits = tuple(new_limit)
                elif limits_type == RangeType.NONE:
                    limits = (default[0], default[1])
                else:
                    raise ValueError(
                        f"valid_range {valid_range} and limits {limits} of object {name} do not take the same format")                     
                                
            elif  range_type == RangeType.NONE :
                if limits_type == RangeType.VALUE:
                    min_limit, max_limit = limits

                    if min_limit is None:
                        min_limit = default[0]
                    if max_limit is None:
                        max_limit = default[1]

                    if min_limit > max_limit:
                        min_limit, max_limit = max_limit, min_limit

                    limits = (min_limit, max_limit)
                    valid_range = limits
                elif  limits_type == RangeType.TUPLE:
                    new_limit = []
                    for limit_item in limits:
                        min_limit, max_limit = limit_item
                        if min_limit is None:
                            min_limit = default[0]
                        if max_limit is None:
                            max_limit = default[1]

                        if min_limit > max_limit:
                            min_limit, max_limit = max_limit, min_limit    

                        new_limit.append((min_limit, max_limit))
                    limits = tuple(new_limit) 
                    valid_range = limits                   
                elif limits_type == RangeType.NONE:
                    valid_range = (default[0], default[1])
                    limits = (default[0], default[1])  

        else:
            valid_range = None
            limits = None
        return limits, valid_range

    def __init__(
        self,
        name: str,
        port: "cosapp.ports.port.ExtensiblePort",
        value: Any,
        unit: str = "",
        dtype: Types = None,
        valid_range: RangeValue = None,
        invalid_comment: str = "",
        limits: RangeValue = None,
        out_of_limits_comment: str = "",
        desc: str = "",
        distribution: Optional[Distribution] = None,
        scope: Scope = Scope.PRIVATE,
    ):
        self._name = self.name_check(name)

        from cosapp.ports.port import ExtensiblePort
        check_arg(port, 'port', ExtensiblePort)
        self._port = port

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

        if not is_numerical(value) and unit:
            unit = ""
            logger.warning(
                f"A physical unit is defined for non-numerical variable {name!r}; it will be ignored."
            )

        elif not units.is_valid_units(unit) and unit:
            raise units.UnitError(f"Unknown unit {unit}.")

        if dtype is None:
            if value is None:
                dtype = (
                    None
                )  # If value and dtype are None, we cannot figure out the type
            else:
                dtype = type(value)
                if is_numerical(value):
                    # Force generic number type only if user has not specified any type
                    if issubclass(dtype, Number):
                        dtype = (Number, np.ndarray)
                    elif isinstance(value, (MutableSequence, array.ArrayType)):
                        # We have a collection => transform Mutable to ndarray
                        dtype = (MutableSequence, array.ArrayType, np.ndarray)
                        value = np.asarray(value)
        elif value is not None:
            # Test value has the right type
            if not isinstance(value, dtype):
                typename = get_typename(dtype)
                raise TypeError(
                    "Cannot set {}.{} of type {} with a {}.".format(
                        port.contextual_name, name, typename, type(value).__qualname__
                    )
                )

        # TODO: better handle of this possible miss of understanding of the user at numpy array instantiation
        if isinstance(value, np.ndarray):
            if value.dtype == np.integer:
                logger.warning(
                    f"Variable {name!r} instantiates a numpy array with integer dtype."
                    " This may lead to unpredictible consequences."
                )

        # Check validation ranges are compatible and meaningful for this type of data
        limits, valid_range = self._check_range(limits, valid_range, value, self._port.contextual_name)

        if valid_range is None and len(invalid_comment):
            logger.warning(
                f"Invalid comment specified for variable {name!r} without validity range."
            )

        if limits is None and len(out_of_limits_comment):
            logger.warning(
                f"Out-of-limits comment specified for variable {name!r} without limits."
            )

        self._unit = unit  # type: str
        self._dtype = dtype  # type: Types
        self._valid_range = valid_range  # type: RangeValue
        self._invalid_comment = ""  # type: str
        self.invalid_comment = invalid_comment
        self._limits = limits  # type: RangeValue
        self._out_of_limits_comment = ""  # type: str
        self.out_of_limits_comment = out_of_limits_comment
        self._desc = desc  # type: str
        self._distribution = None  # type: Optional[Distribution]
        self.distribution = distribution
        self._scope = scope  # type: Scope

    @classmethod
    def name_check(cls, name: str):
        return cls.__name_check(name)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        msg = {"name": self.name, "unit": " " + self.unit if self.unit else ""}
        value = getattr(self._port, self._name)
        if is_number(value):
            msg["value"] = f"{value:.5g}"
        else:
            msg["value"] = value

        min_valid, max_valid = (
            self.valid_range if self.valid_range is not None else (None, None)
        )
        min_limit, max_limit = self.limits if self.limits is not None else (None, None)
        if min_limit is None or np.all(np.isinf(min_limit)):
            msg["min_limit"] = ""
        else:
            msg["min_limit"] = f" &#10647; {min_limit:.5g} &#10205; "
        if max_limit is None or np.all(np.isinf(max_limit)):
            msg["max_limit"] = ""
        else:
            msg["max_limit"] = f" &#10206; {max_limit:.5g} &#10648; "
        if min_valid is None or min_limit == min_valid or np.all(np.isinf(min_valid)):
            msg["min_valid"] = ""
        else:
            msg["min_valid"] = f"{min_valid:.5g} &#10205; "
        if max_valid is None or max_limit == max_valid or np.all(np.isinf(max_valid)):
            msg["max_valid"] = ""
        else:
            msg["max_valid"] = f" &#10206; {max_valid:.5g}"
        if self.description:
            msg["description"] = " # " + self.description
        else:
            msg["description"] = ""

        scope_format = {
            Scope.PRIVATE: " &#128274;&#128274; ",
            Scope.PROTECTED: " &#128274; ",
            Scope.PUBLIC: "",
        }
        msg["scope"] = scope_format[self.scope]

        if len(msg["min_limit"] + msg["min_valid"]) > 0 or len(msg["max_limit"] + msg["max_valid"]) > 0:
            msg["range"] = " value "
            msg["separator"] = "; "
        else:
            msg["range"] = ""
            msg["separator"] = ""

        return (
            "{name}{scope}: {value!s}{unit}"
            "{separator}{min_limit}{min_valid}{range}{max_valid}{max_limit}"
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
            "value": getattr(self._port, self._name),
            "valid_range": self.valid_range,
            "invalid_comment": self.invalid_comment,
            "limits": self.limits,
            "out_of_limits_comment": self.out_of_limits_comment,
            "distribution": self.distribution.__json__() if self.distribution else None,
        }

    @property
    def name(self) -> str:
        """str : Variable name"""
        return self._name

    @property
    def unit(self) -> str:
        """str : Variable unit; empty string means dimensionless"""
        return self._unit

    @property
    def dtype(self) -> Types:
        """Type[Any] or Tuple of Type[Any] or None : Type of the variable; default None (i.e. type of default value is set)"""
        return self._dtype

    @property
    def valid_range(self) -> RangeValue:
        """Tuple[Any, Any] or None : alidity range of the variable and optional comment if unvalid"""
        
        return self._valid_range

    @valid_range.setter
    def valid_range(self, new_range: RangeValue):
        
        range_type = Variable.check_range_type(new_range, self._port.contextual_name)

        value = getattr(self._port, self._name)
        default = self._get_limits_from_type(value)

        if default is not None:
            if range_type == RangeType.VALUE :
                if len(new_range) != 2:
                    raise TypeError(
                        "Validity range must be a 2-tuple with type comparable to value."
                    )
                limits = self.limits
                valid_range = new_range
                

            elif range_type == RangeType.TUPLE: 
                limits = self.limits
                valid_range = new_range
                current_range = self.valid_range
                 
            elif range_type == RangeType.NONE:
                limits = self.limits
                valid_range = new_range
                current_range = self.valid_range
                min_valid = (
                    limits[0] if limits[0] > current_range[0] else current_range[0]
                )
                max_valid = (
                    limits[1] if limits[1] < current_range[1] else current_range[1]
                )
                valid_range = (min_valid, max_valid)      
        

        else:
            valid_range = None
            limits = None

        limits, valid_range = self._check_range(self.limits, new_range, value)
        self._valid_range = valid_range
        self._limits = limits

    @property
    def invalid_comment(self) -> str:
        """str : Comment explaining the reasons of the validity range"""
        return self._invalid_comment

    @invalid_comment.setter
    def invalid_comment(self, new_comment: str):
        if not isinstance(new_comment, str):
            raise TypeError(f"invalid_comment must be a string; got {new_comment!s}")
        self._invalid_comment = new_comment

    @property
    def limits(self) -> RangeValue:
        """Tuple[Any, Any] or None : Variable limits and optional comment if unvalid"""
        return self._limits

    @limits.setter
    def limits(self, new_limits: RangeValue):

        limits_type = Variable.check_range_type(new_limits, self._port.contextual_name)

        value = getattr(self._port, self._name)
        default = self._get_limits_from_type(value)

        if default is not None:
            if limits_type == RangeType.VALUE :
                if len(new_limits) != 2:
                    raise TypeError(
                        "Limits must be a 2-tuple with type comparable to value."
                    )

                limits = new_limits
                if new_limits[0] is None:
                    limits = (default[0], limits[1])
                if new_limits[1] is None:
                    limits = (limits[0], default[1])

                current_range = self.valid_range
                min_valid = limits[0] if limits[0] > current_range[0] else current_range[0]
                max_valid = limits[1] if limits[1] < current_range[1] else current_range[1]
                valid_range = (min_valid, max_valid)
            
            elif limits_type == RangeType.TUPLE:
                limits_list = list(new_limits)
                for index in range(len(limits_list)):

                    if limits_list[index][0] is None:
                       limits_list[index] =  (default[0], limits_list[index][1])
                    if limits_list[index][1] is None:
                       limits_list[index][1] =  (limits_list[index][0],default[1])
                limits = tuple(limits_list)

                current_range = self.valid_range
                valid_range_list = list(current_range)
                for index in range(len(valid_range_list)):
                    limit_value = limits[index]
                    range_value = current_range[index]
                    valid_range_list[index] = (limit_value[0] if limit_value[0] > range_value[0] else range_value[0], valid_range_list[index][1]) 
                    valid_range_list[index] = (valid_range_list[index][0], limit_value[1] if limit_value[1] < range_value[1] else range_value[1])
                valid_range = tuple(valid_range_list)

            elif limits_type == RangeType.NONE:
                limits = self.limits
                current_range = self.valid_range
                min_valid = limits[0] if limits[0] > current_range[0] else current_range[0]
                max_valid = limits[1] if limits[1] < current_range[1] else current_range[1]
                valid_range = (min_valid, max_valid)


        else:
            valid_range = None
            limits = None

        limits, valid_range = self._check_range(limits, valid_range, value)
        self._valid_range = valid_range
        self._limits = limits

    @property
    def out_of_limits_comment(self) -> str:
        """str : Comment explaining the reasons of the limits"""
        return self._out_of_limits_comment

    @out_of_limits_comment.setter
    def out_of_limits_comment(self, new_comment: str):
        if not isinstance(new_comment, str):
            raise TypeError(f"out_of_limits_comment must be a string; got {new_comment!s}")
        self._out_of_limits_comment = new_comment

    @property
    def description(self) -> str:
        """str : Variable description"""
        return self._desc

    @property
    def scope(self) -> Scope:
        """Scope : Scope of variable visibility"""
        return self._scope

    @property
    def distribution(self) -> Optional[Distribution]:
        """Optional[Distribution] : Random distribution of the variable."""
        return self._distribution

    @distribution.setter
    def distribution(self, new_distribution: Optional[Distribution]):
        ok = new_distribution is None or isinstance(new_distribution, Distribution)
        if not ok:
            typename = type(new_distribution).__qualname__
            raise TypeError(f"Random distribution should be of type 'Distribution'; got {typename}.")
        self._distribution = new_distribution

    def is_valid(self) -> Validity:
        """Get the variable value validity.

        Returns
        -------
        Validity
            Variable value validity
        """
        status = Validity.OK
        value = getattr(self._port, self._name)

        if not isinstance(value, (Number, np.ndarray)):
            return status

        if self.valid_range is not None:

            range_type = Variable.check_range_type(self.valid_range, self._port.contextual_name)
            if isinstance(value, np.ndarray):  
                
                if range_type == RangeType.VALUE:
                    min_range, max_range = self.valid_range 
                    if np.any(np.where(value <= max_range, False, True)) or np.any(np.where(value >= min_range, False, True)):
                        status = Validity.WARNING
                    
                    if self.limits is not None:
                        min_limit, max_limit = self.limits
                        if np.any(np.where(value <= max_limit, False, True)) or np.any(np.where(value >= min_limit, False, True)):
                            status = Validity.ERROR

                elif range_type == RangeType.TUPLE:

                    for index,val in enumerate(value):
                        if (self.valid_range[index][0] > val) or (self.valid_range[index][1] < val):
                            status = Validity.WARNING
                            break
                    
                    if self.limits is not None:
                        for index,val in enumerate(value):
                            if (self.limits[index][0] > val) or (self.limits[index][1] < val):
                                status = Validity.ERROR
                                break

                else:
                    raise ValueError(
                        f"Mixed values in valid_range object {self.valid_range} of {self._port.contextual_name}."
                        " Valid object can contain only numerical values or only tuples")                     
                
            else:    
                if range_type == RangeType.VALUE:
                    min_range, max_range = self.valid_range          
                    if not min_range <= value <= max_range:
                        status = Validity.WARNING

                        if self.limits is not None:
                            min_limit, max_limit = self.limits
                            if not min_limit <= value <= max_limit:
                                status = Validity.ERROR

                elif range_type == RangeType.TUPLE:
                    raise ValueError(
                        f"valid_range {self.valid_range} or limits {self.limits} of object {self._port.contextual_name}"
                        f" does not take the same format as its input {value}")

        return status

    def get_validity_comment(self, status: Validity) -> str:
        """Get the ground arguments used to established the variable validity range.

        The status `Validity.OK` has no ground arguments.

        Parameters
        ----------
        status : Validity.{OK, WARNING, ERROR}
            Validity status for which the reasons are looked for.

        Returns
        -------
        str
            Validity comment
        """

        def range2str(range: RangeValue) -> str:

            range_type = Variable.check_range_type(range, self._port.contextual_name)

            if range_type == RangeType.VALUE:
                min_valid, max_valid = range if range is not None else (None, None)

            elif range_type == RangeType.TUPLE:
                return f"{range}"[1:-1].join("[]")

            elif range_type == RangeType.NONE:
                min_valid, max_valid = (None, None)


            def get_range_repr(valid, fmt):
                if not isinstance(valid, (list, tuple, np.ndarray)):
                    fmt.replace("{}", "{:.5g}")
                return fmt.format(valid)

            if min_valid is None:
                range_repr = "] ,"
            else:
                range_repr = get_range_repr(min_valid, "[{}, ")

            if max_valid is None:
                range_repr += " ["
            else:
                range_repr += get_range_repr(max_valid, "{}]")
            return range_repr

        if status == Validity.ERROR:
            return " - ".join((range2str(self.limits), self.out_of_limits_comment))
        elif status == Validity.WARNING:
            return " - ".join((range2str(self.valid_range), self.invalid_comment))
        else:  # Variable is ok
            return ""
