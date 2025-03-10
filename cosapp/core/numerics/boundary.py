from __future__ import annotations
from numbers import Number
from typing import (
    Any, Dict, Optional, Collection,
    Union, Tuple, Type, NamedTuple, T,
    MutableSequence, TYPE_CHECKING,
)
import abc
import copy
import numpy

from cosapp.core.eval_str import EvalString
from cosapp.core.variableref import VariableReference
from cosapp.ports.port import BasePort
from cosapp.ports.exceptions import ScopeError
from cosapp.utils.helpers import check_arg
from cosapp.utils.naming import natural_varname
from cosapp.utils.parsing import find_selector
from cosapp.utils.state_io import object__getstate__

if TYPE_CHECKING:
    from cosapp.systems import System


class AttrRef:
    """Attribute Reference for scalar object.

    In addition to System and its derivatives, manage also complex object 
    which could be included in the evaluation context.

    Parameters
    ----------
    obj: cosapp.systems.System
        System in which the boundary name is defined.
    key: str
        Name of the boundary
    """
    def __init__(self, obj: System, key: str) -> None:
        self._obj: Union[BasePort, Any]

        name = key
        try:
            base, key = key.rsplit(".", maxsplit=1)
        except ValueError:
            base = ""
        else:
            obj = eval(f"__obj__.{base}", {"__obj__": obj})

        from cosapp.systems import System
        if isinstance(obj, System):
            self._obj = obj.name2variable[key].mapping
        else:
            self._obj = obj

        if isinstance(self._obj, dict):
            raise ValueError("Only variables can be used in mathematical algorithms")

        self._key: str = key
        self._base: str = base
        self._name: str = natural_varname(name)

    @property
    def value(self) -> Number:
        return getattr(self._obj, self._key)

    @value.setter
    def value(self, val: Number) -> None:
        setattr(self._obj, self._key, val)

    def __copy__(self) -> AttrRef:
        return AttrRef(self._obj, self._key)

    def __eq__(self, other: AttrRef) -> bool:
        try:
            return self._obj is other._obj and self._key == other._key
        except:
            return False
    
    def __getstate__(self) -> Dict[str, Any]:
        """Creates a state of the object.

        The state type does NOT match type specified in
        https://docs.python.org/3/library/pickle.html#object.__getstate__
        to allow custom serialization.

        Returns
        -------
        Dict[str, Any]:
            state
        """
        return object__getstate__(self)

    def __json__(self) -> Dict[str, Any]:
        """Creates a JSONable dictionary representation of the object.

        Returns
        -------
        Dict[str, Any]
            The dictionary
        """
        state = self.__getstate__().copy()
        # state.pop[""]
        return state

class MaskedAttrRef(AttrRef):
    """Masked Attribute Reference for MutableSequence-like object.

    Include a mask applying to an evaluation context vector.

    Parameters
    ----------
    obj: cosapp.systems.System
        System in which the boundary name is defined.
    key: str
        Name of the boundary
    mask: numpy.ndarray
        Mask of the values in the vector boundary.
    """
    def __init__(self, obj: System, key: str, mask: numpy.ndarray) -> None:
        super().__init__(obj, key)
        self.set_attributes(mask)

    def set_attributes(self, mask: numpy.ndarray) -> None:
        self._mask = numpy.atleast_1d(mask)
        self._mask_idx = self._mask.nonzero()[0]

        array = getattr(self._obj, self._key)
        self._ref_shape = (len(array),)
        self._ref_size = len(array)

    @property
    def value(self) -> MutableSequence:
        obj = getattr(self._obj, self._key)
        return list(map(obj.__getitem__, self._mask_idx))

    @value.setter
    def value(self, val: MutableSequence):
        obj = getattr(self._obj, self._key)
        for i, new in zip(self._mask_idx, val):
            obj.__setitem__(i, new)

    def set_mask(self, mask: numpy.ndarray) -> None:
        self._mask[:] = mask
        self._mask_idx = mask.nonzero()[0]

    def __copy__(self) -> MaskedAttrRef:
        return MaskedAttrRef(self._obj, self._key, self._mask.copy())
    
    def __eq__(self, other: MaskedAttrRef) -> bool:
        try:
            return super().__eq__(other) and numpy.array_equal(self._mask, other._mask)
        except:
            return False

    @classmethod
    def make_from_attr_ref(
        cls: MaskedAttrRef,
        attr_ref: AttrRef,
        obj: Union[BasePort, Any],
        name: str,
        mask: numpy.ndarray
    ) -> MaskedAttrRef:
        mask_ref = attr_ref.__new__(cls, obj, name, mask)
        for key, value in vars(attr_ref).items():
            setattr(mask_ref, key, value)
        mask_ref.set_attributes(mask)
        return mask_ref


class NumpyMaskedAttrRef(AttrRef):
    """ Masked Attribute Reference for numpy arrays.

    Include a mask applying to an evaluation context vector.

    Parameters
    ----------
    obj: cosapp.systems.System
        System in which the boundary name is defined.
    key: str
        Name of the boundary
    mask: numpy.ndarray
        Mask of the values in the vector boundary.
    """
    def __init__(self, obj: System, key: str, mask: numpy.ndarray) -> None:
        super().__init__(obj, key)
        self.set_attributes(mask)

    def set_attributes(self, mask: numpy.ndarray) -> None:
        self._mask = numpy.asarray(mask)
        self._mask_idx = self._mask.nonzero()[0]

        array: numpy.ndarray = getattr(self._obj, self._key)
        self._ref_shape = array.shape
        self._ref_size = array.size

    @property
    def value(self) -> numpy.ndarray:
        return getattr(self._obj, self._key)[self._mask]

    @value.setter
    def value(self, val: Union[numpy.ndarray, MutableSequence]):
        getattr(self._obj, self._key)[self._mask] = numpy.asarray(val)
    
    def set_mask(self, mask: numpy.ndarray) -> None:
        self._mask[:] = mask

    def __copy__(self) -> NumpyMaskedAttrRef:
        return NumpyMaskedAttrRef(self._obj, self._key, self._mask.copy())
    
    def __eq__(self, other: MaskedAttrRef) -> bool:
        try:
            return super().__eq__(other) and numpy.array_equal(self._mask, other._mask)
        except:
            return False

    @classmethod
    def make_from_attr_ref(
        cls: NumpyMaskedAttrRef,
        attr_ref: AttrRef,
        obj: Union[BasePort, Any],
        name: str,
        mask: numpy.ndarray
    ) -> NumpyMaskedAttrRef:
        mask_ref = attr_ref.__new__(cls, obj, name, mask)
        for key, value in vars(attr_ref).items():
            setattr(mask_ref, key, value)
        mask_ref.set_attributes(mask)
        return mask_ref


class MaskedVarInfo(NamedTuple):
    basename: str
    selector: str = ""
    mask: Optional[numpy.ndarray] = None

    @property
    def fullname(self) -> str:
        return f"{self.basename}{self.selector}"

    def __eq__(self, other: MaskedVarInfo) -> bool:
        try:
            return self[:2] == other[:2] and numpy.array_equal(self.mask, other.mask)
        except:
            return False


class Boundary:
    """Numerical solver boundary.

    Parameters
    ----------
    context: cosapp.systems.System
        System in which the boundary is defined.
    name: str
        Name of the boundary
    mask: numpy.ndarray or None
        Mask of the values in the vector boundary.
    default: Number, numpy.ndarray or None
        Default value to set the boundary with.
    inputs_only: bool, optional
        If `True` (default), output variables are regarded as invalid.
    """
    def __init__(self,
        context: System,
        name: str,
        mask: Optional[numpy.ndarray] = None,
        default: Union[Number, numpy.ndarray, None] = None,
        inputs_only: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)  # for collaborative inheritance

        self._context = context
        self._default_value: Union[Number, numpy.ndarray, None] = None

        basename, selector = Boundary.parse_expression(name)
        value, mask = Boundary.create_mask(context, basename, selector, mask)
        self._ref, self._boundary_impl, self._is_scalar = Boundary.create_attr_ref(context, basename, value, mask)
        self.find_port(inputs_only)
        self._name_info = MaskedVarInfo(basename, selector, mask)

        # Set default value if any
        if default is not None:
            self.update_default_value(default)
    
    @property
    def is_scalar(self) -> bool:
        """Returns whether this boundary is scalar or not."""
        return self._is_scalar

    def copy(self) -> Boundary:
        boundary = copy.copy(self)
        boundary._default_value = copy.copy(self._default_value)
        boundary._ref = self._ref.__copy__()
        return boundary

    def __getstate__(self) -> Union[Dict[str, Any], tuple[Optional[Dict[str, Any]], Dict[str, Any]]]:
        """Creates a state of the object.

        The state may take various forms depending on the object, see
        https://docs.python.org/3/library/pickle.html#object.__getstate__
        for further details.
        
        Returns
        -------
        Union[Dict[str, Any], tuple[Optional[Dict[str, Any]], Dict[str, Any]]]:
            state
        """
        return object__getstate__(self)

    def __json__(self) -> Dict[str, Any]:
        """Creates a JSONable dictionary representation of the object.

        Returns
        -------
        Dict[str, Any]
            The dictionary
        """
        qualname = f"{self.__module__}.{self.__class__.__qualname__}"
        state = self.__getstate__().copy()

        state.pop("_context")
        state.pop("_boundary_impl")
        return {
            "__class__": qualname,
            **state,
        }

    @staticmethod
    def parse_expression(expression: str) -> MaskedVarInfo:
        """Decompose a variable specification into its base name and selector.

        Parameters
        ----------
        expression : str
            Variable specification (variable name + optional array mask, if required)

        Returns
        -------
        - str: variable name
        - str: array selector
        """
        check_arg(expression, 'expression', str)
        expression = natural_varname(expression)

        try:
            basename, selector = find_selector(expression)
        except ValueError as error:
            raise SyntaxError(error)

        return basename, selector
    
    @staticmethod
    def create_mask(
        system: System,
        basename: str,
        selector: str,
        mask: Optional[numpy.ndarray] = None,
    ) -> Tuple[Optional[Union[Number, Collection]], Optional[numpy.ndarray]]:
        """Evaluate the basename expression within its context 
        and generate a mask if a selector is specified in the fullname expression.

        Parameters
        ----------
        - system: System
            System to which variable belongs.
        - basename: str
            Variable name without any optional array mask.
        - selector: str
            Expression corresponding to an array mask.
        - mask: Optional[numpy.ndarray]
            Imposed mask to apply on the variable; default is None (i.e. no mask).
        
        Returns
        -------
        Optional[Union[Number, Collection]]
            Value of the context variable.
        Optional[numpy.ndarray]
            Imposed or generated mask to apply on the variable.
        """
        # evaluate expression without mask if any
        try:
            value = eval(f"s.{basename}", {}, {"s": system})
        except AttributeError as error:
            error.args = (f"{basename!r} is not known in {system.name}",)
            raise
        except Exception as error:
            error.args = (f"Can't evaluate {basename!r} in {system.name}",)
            raise

        # get or create mask
        if mask is not None:
            if isinstance(value, Number):
                raise TypeError("A mask cannot be applied on a scalar.")
            check_arg(mask, "mask", (list, tuple, numpy.ndarray))

            return value, numpy.asarray(mask)

        if selector:
            # Check value is an array
            if isinstance(value, numpy.ndarray) or Boundary.is_mutable_sequence(value):
                if isinstance(value, numpy.ndarray) and not (numpy.issubdtype(value.dtype, numpy.number) or value.size > 1):
                    raise ValueError(
                            f"Only non-empty numpy arrays can be partially selected; got {value}."
                        )
                elif Boundary.is_mutable_sequence(value) and not len(value) > 1:
                    raise ValueError(
                            f"Only non-empty MutableSequence-like arrays can be partially selected; got {value}."
                        )
            else:
                raise TypeError(
                        f"Only non-empty arrays can be partially selected; got {type(value)}."
                    )

            # Set mask from selector string
            mask = numpy.zeros_like(value, dtype=bool)
            try:
                exec(f"mask{selector} = True", {}, {"mask": mask})
            except (SyntaxError, IndexError) as error:
                varname = f"{system.name}.{basename}"
                error.args = (
                    f"Invalid selector {selector!r} for variable {varname!r}: {error!s}",
                )
                raise

        elif isinstance(value, numpy.ndarray) or Boundary.is_mutable_sequence(value):
            mask = numpy.ones_like(value, dtype=bool)

        return value, mask

    @staticmethod
    def is_mutable_sequence(value: Any) -> bool:
        """Determine if an object is MutableSequence-like."""
        mandatory_attrs = ["__getitem__", "__setitem__", "__len__"]
        return all([hasattr(value, attr) for attr in mandatory_attrs])

    @staticmethod
    def create_attr_ref(
        context: System,
        basename: str,
        value: Optional[Union[Number, Collection]],
        mask: Optional[numpy.ndarray] = None
    ) -> Tuple[Union[AttrRef, NumpyMaskedAttrRef, MaskedAttrRef], AbstractBoundaryImpl, bool]:
        """
        Returns an `AttrRef`, `MaskedAttrRef`, or ǸumpyMaskedAttrRef` object from a name and its evaluation context.
        The `NumpyMaskedAttrRef` derives from `AttrRef` if the context variable refers to a numpy.array and
        `MaskedAttrRef` for a variable referring to an object similar to a MutableSequence.
        In the two latter cases, a mask may be applied on value.

        Parameters
        ----------
        - context: System
            System in which the boundary is defined.
        - basename: str
            Name of the boundary without its mask if any.
        - value: Optional[Union[Number, Collection]]
            Value of the context variable.
        - mask: Optional[numpy.ndarray]
            Mask to apply on the variable; default is None (i.e. no mask).

        Returns
        -------
        Union[AttrRef, NumpyMaskedAttrRef, MaskedAttrRef]
            (Masked) Attribute Reference object.
        AbstractBoundaryImpl
            Object containing methods specific according to the variable type.
        bool
            Specify if the boundary value is a scalar.
        """
        if mask is None:
            if isinstance(value, Number):
                return AttrRef(context, basename), ScalarBoundaryImpl(), True
            elif value is None:
                return AttrRef(context, basename), UndefinedBoundaryImpl(), True
            else:
                return AttrRef(context, basename), GenericBoundaryImpl(), True
        else:
            if isinstance(value, numpy.ndarray):
                return NumpyMaskedAttrRef(context, basename, mask), NumpyBoundaryImpl(), False
            elif Boundary.is_mutable_sequence(value):
                return MaskedAttrRef(context, basename, mask), MutableSeqBoundaryImpl(), False
        
        raise TypeError("Type of evaluated expression is incompatible as Boundary object handled type.")

    def find_port(self, inputs_only: bool = False) -> None:
        """
        Find port associated to its `AttrRef`.
        In the case of a complex object, the port containing it is retrieved and checks.

        Parameters
        ----------
        - inputs_only [bool, optional]:
            If `True`, output variables are regarded as invalid. Default is `False`.
        
        """
        portname = self._ref._base
        portkey = self._ref._key
        obj = self._ref._obj
        if not isinstance(obj, BasePort):
            for i in range(len(portname.split("."))):
                if portname in self._context.name2variable:
                    obj = self._context.name2variable[portname].mapping
                    break
                else:
                    portname = portname.rsplit('.', maxsplit=1)[0]
        else:
            portname = ".".join(filter(None, portname.split(".") + portkey.split(".")))         

        if not isinstance(obj, BasePort):
            raise TypeError(f"Invalid port; got {type(obj)}")
        self._port = port = obj
        self._portname = portname

        if inputs_only and not port.is_input:
            raise ValueError(
                f"Only variables in input ports can be used as boundaries; got {portname!r} in {port.contextual_name!r}."
            )
        if port.out_of_scope(portkey):
            if self._context is not port.owner:
                # Only owner can set its variables
                raise ScopeError(f"Trying to set variable {self._ref._name!r} out of your scope through a boundary.")

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return f"{self.name} := {self!s}"

    def __eq__(self, other: Boundary) -> bool:
        try:
            return self._context is other._context and self._ref == other._ref
        except:
            return False

    @property
    def context(self) -> System:
        """cosapp.systems.System : `System` in which the boundary is defined."""
        return self._context

    @property
    def port(self) -> BasePort:
        """BasePort: port containing the boundary."""
        return self._port

    @property
    def name(self) -> str:
        """str : Contextual name of the boundary."""
        return self._name_info.fullname

    @property
    def basename(self) -> str:
        """str : Base name of the boundary."""
        return self._name_info.basename

    def contextual_name(self, context: Optional[System] = None) -> str:
        """str : Contextual name of the boundary, relative to `context`.
        If `context` is `None` (default), uses current variable context.
        """
        if context is None:
            path = self.context.name
        else:
            path = context.get_path_to_child(self.context) or self.context.name
        return f"{path}.{self.name}"

    @property
    def ref(self) -> Union[AttrRef, MaskedAttrRef, NumpyMaskedAttrRef]:
        """AttrRef : attribute reference accessed by the boundary."""
        return self._ref

    @property
    def variable_reference(self) -> VariableReference:
        """VariableReference : variable reference accessed by the boundary."""
        return self.context.name2variable[self.portname]

    @property
    def portname(self) -> str:
        """str : name of the port accessed by the boundary."""
        return self._portname

    @property
    def variable(self) -> str:
        """str : name of the variable accessed by the boundary."""
        return self._ref._key

    def touch(self) -> None:
        """Set owner port as 'dirty'."""
        self.port.touch()
        self.port.owner.touch()

    @property
    def mask(self) -> Optional[numpy.ndarray]:
        """numpy.ndarray or None : Mask of the values in the vector boundary."""
        return self._ref._mask

    @mask.setter
    def mask(self, mask: Optional[numpy.ndarray]) -> None:
        if not self._is_scalar and mask is not None:
            mask = numpy.asarray(mask)
            if mask.shape != self._ref._ref_shape:
                raise ValueError(
                    f"Set mask does not fit the context array shape"
                    f"; got {mask.shape!r} to set in {self._ref._ref_shape}."
                )
            if self._default_value is not None:
                default_size = numpy.asarray(self._default_value).size
                if numpy.count_nonzero(mask) != default_size:
                    raise ValueError(
                        f"Set mask does not fit the current boundary value"
                        f"; got {numpy.count_nonzero(mask)!r} mismatching {default_size}."
                    )
            self._ref.set_mask(mask)

    @property
    def value(self) -> Union[Number, numpy.ndarray]:
        return self._ref.value

    def update_value(self, new: Union[Number, MutableSequence, numpy.ndarray], checks: bool = True) -> None:
        if new is not None:
            if self._boundary_impl.update_value(self._ref.value, new, checks):
                self._ref.value = new
                self.touch()

    def set_to_default(self) -> None:
        if self._default_value is not None:
            self.update_value(self._default_value, checks=False)

    @property
    def default_value(self) -> Union[Number, numpy.ndarray]:
        return self._default_value

    def update_default_value(self, new: Union[Number, MutableSequence, numpy.ndarray], mask: Optional[numpy.ndarray] = None, checks: bool = True) -> None:
        if new is not None:
            if mask is not None:
                self.mask = mask
            if checks:
                self._boundary_impl.check_new_value(self._default_value, new)
            self._default_value = new

    @property
    def size(self) -> int:
        return self._boundary_impl.size(self._ref.value)


class AbstractBoundaryImpl(abc.ABC):
    """Abstract Boundary class to manage methods specific to Boundary type."""

    @abc.abstractmethod
    def check_new_value(value: T, new: T) -> None: ...

    @abc.abstractmethod
    def update_value(ref_value: T, new: T, checks: bool = True) -> bool: ...

    @abc.abstractmethod
    def size(value: T) -> int: ...


class UndefinedBoundaryImpl(AbstractBoundaryImpl):
    """Class handling undefined Boundary."""

    @staticmethod
    def check_new_value(value: T, new: T) -> None:
        raise NotImplementedError

    @staticmethod
    def update_value(ref_value: T, new: T, checks: bool = True) -> bool:
        raise NotImplementedError

    @staticmethod
    def size(value: T) -> int:
        raise NotImplementedError


class ScalarBoundaryImpl(AbstractBoundaryImpl):
    """Specific methods for Number Boundary."""

    @staticmethod
    def check_new_value(value: Number, new: Number) -> None:
        if not isinstance(new, Number):
            raise TypeError(
                f"Value to set is incompatible with the boundary value type"
                f"; got {type(new)} mismatching {type(value)}."
            )

    @staticmethod
    def update_value(ref_value: Number, new: Number, checks: bool = True) -> bool:
        if checks:
            ScalarBoundaryImpl.check_new_value(ref_value, new)
        return ref_value != new

    @staticmethod
    def size(value: Number) -> int:
        return 1


class MutableSeqBoundaryImpl(AbstractBoundaryImpl):
    """Specific methods for MutableSequence-like Boundary."""

    @staticmethod
    def check_new_value(value: MutableSequence, new: MutableSequence) -> None:
        if not Boundary.is_mutable_sequence(new):
            raise TypeError(f"Value to set is incompatible with the boundary value type; got {type(new)} \
                            mismatching {type(value)}.")
        if value is not None and len(new) != len(value):
            raise ValueError(f"Value to set does not fit the current boundary value; got {len(new)} \
                            mismatching {len(value)}.")

    @staticmethod
    def update_value(ref_value: MutableSequence, new: MutableSequence, checks: bool = True) -> bool:
        if checks:
            MutableSeqBoundaryImpl.check_new_value(ref_value, new)
        return not numpy.array_equal(ref_value, new)

    @staticmethod
    def size(value: MutableSequence) -> int:
        return len(value)


class NumpyBoundaryImpl(AbstractBoundaryImpl):
    """Specific methods for numpy.ndarray Boundary."""

    @staticmethod
    def check_new_value(value: numpy.ndarray, new: numpy.ndarray) -> None:
        if value is not None and not numpy.isscalar(new):
            if value.shape != numpy.asarray(new).shape:
                raise ValueError(f"Value to set does not fit the current boundary value; got {numpy.asarray(new).shape!r} \
                            mismatching {value.shape}.")

    @staticmethod
    def update_value(ref_value: numpy.ndarray, new: numpy.ndarray, checks: bool = True) -> bool:
        if checks:
            NumpyBoundaryImpl.check_new_value(ref_value, new)
        return not numpy.array_equal(ref_value, new)

    @staticmethod
    def size(value: numpy.ndarray) -> int:
        return value.size


class GenericBoundaryImpl(AbstractBoundaryImpl):
    """Class handling undefined Boundary."""

    @staticmethod
    def check_new_value(value: T, new: T) -> None:
        if None not in (value, new):
            if not isinstance(type(new), type(value)):
                raise TypeError(
                    f"Value to set is incompatible with the boundary value type"
                    f"; got {type(new)} mismatching {type(value)}."
                )

    @staticmethod
    def update_value(ref_value: T, new: T, checks: bool = True) -> bool:
        if checks:
            GenericBoundaryImpl.check_new_value(ref_value, new)
        return ref_value != new

    @staticmethod
    def size(value: T) -> int:
        raise NotImplementedError
    

class Unknown(Boundary):
    """Numerical solver unknown.

    Parameters
    ----------
    context : cosapp.systems.System
        System in which the unknown is defined.
    name : str
        Name of the unknown
    lower_bound : float
        Minimum value authorized; default -numpy.inf
    upper_bound : float
        Maximum value authorized; default numpy.inf
    max_abs_step : float
        Max absolute step authorized in one iteration; default numpy.inf
    max_rel_step : float
        Max relative step authorized in one iteration; default numpy.inf
    mask : numpy.ndarray or None
        Mask of unknown values in the vector variable.

    Attributes
    ----------
    lower_bound : float
        Minimum value authorized; default -numpy.inf
    upper_bound : float
        Maximum value authorized; default numpy.inf
    max_abs_step : float
        Largest absolute step authorized in one iteration; default numpy.inf
    max_rel_step : float
        Largest relative step authorized in one iteration; default numpy.inf

    Notes
    -----
    The dimensionality of the variable should be taken into account in the bounding process.
    """

    def __init__(self,
        context: System,
        name: str,
        # absolute_step: Number = 1.5e-8,  # TODO ?
        # relative_step: Number = 1.5e-8,  # TODO ?
        max_abs_step: Number = numpy.inf,
        max_rel_step: Number = numpy.inf,
        lower_bound: Number = -numpy.inf,
        upper_bound: Number = numpy.inf,
        # reference: Union[Number, numpy.ndarray] = 1.,  # TODO normalize unknown
        mask: Optional[numpy.ndarray] = None,
    ):
        super().__init__(context, name, mask, inputs_only=True)
        self.check_numerical_type()

        check_arg(max_abs_step, 'max_abs_step', Number, lambda x: x > 0)
        check_arg(max_rel_step, 'max_rel_step', Number, lambda x: x > 0)
        check_arg(lower_bound, 'lower_bound', Number)
        check_arg(upper_bound, 'upper_bound', Number)

        # TODO take into account the variable dimension in the constructor ?
        self.lower_bound = lower_bound  # type: Number
        self.upper_bound = upper_bound  # type: Number
        self.max_abs_step = max_abs_step  # type: Number
        self.max_rel_step = max_rel_step  # type: Number

    def __eq__(self, other: Unknown) -> bool:
        try:
            return super().__eq__(other) and all(
                getattr(self, name) == getattr(other, name)
                for name in ("max_abs_step", "max_rel_step", "lower_bound", "upper_bound")
            )
        except:
            return False

    def __str__(self) -> str:
        try:
            return str(self.value)
        except KeyError:  # boundary does not exist in the current context
            return str(self.default_value)
    
    def check_numerical_type(self):
        if not isinstance(self.ref.value, (Number, numpy.ndarray, type(None))):
            if not Boundary.is_mutable_sequence(self.ref.value):
                raise TypeError(
                    f"Only numerical variables can be used in mathematical algorithms; got {self.portname!r} in {self.context.name!r}"
                )

    def copy(self) -> Unknown:
        """Copy the unknown object.

        Returns
        -------
        Unknown
            Duplicated unknown
        """
        return self.transfer(self.context, self.name)

    def transfer(self, context: System, name: str) -> Unknown:
        """Transfer a copy of the unknown in a new context.

        Returns
        -------
        Unknown
            Duplicated unknown, in new context
        """
        new = Unknown(
            context,
            name,
            max_abs_step=self.max_abs_step,
            max_rel_step=self.max_rel_step,
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            mask=self.mask.copy() if not self._is_scalar else None
        )

        return new

    def to_dict(self) -> Dict[str, Any]:
        """Returns a JSONable representation of the unknown.
        
        Returns
        -------
        Dict[str, Any]
            JSONable representation
        """
        return {
            "context": self.context.contextual_name,
            "name": self.name,
            "varname": self.variable,
            "max_abs_step": self.max_abs_step,
            "max_rel_step": self.max_rel_step,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "mask": None if not hasattr(self, "mask") else self.mask.tolist()
        }


class AbstractTimeUnknown(abc.ABC):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # for collaborative inheritance

    @property
    @abc.abstractmethod
    def der(self) -> EvalString:
        """Expression of the time derivative, given as an EvalString"""
        pass

    @property
    @abc.abstractmethod
    def max_time_step_expr(self) -> EvalString:
        """Expression of the maximum admissible time step, given as an EvalString."""
        pass

    @property
    @abc.abstractmethod
    def max_abs_step_expr(self) -> EvalString:
        """Expression of the maximum absolute step in one iteration, given as an EvalString."""
        pass

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset transient unknown to a reference value"""
        pass

    @abc.abstractmethod
    def touch(self) -> None:
        """Set owner port as 'dirty'."""
        pass

    @property
    def d_dt(self) -> Any:
        """Value of time derivative"""
        return self.der.eval()

    @property
    def max_abs_step(self) -> float:
        """float: Maximum absolute step in one iteration"""
        return self.max_abs_step_expr.eval()

    @property
    def max_time_step(self) -> float:
        """float: Maximum admissible time step in one iteration"""
        dt_max = self.max_time_step_expr.eval()
        dx_max = self.max_abs_step
        if numpy.isfinite(dx_max):
            step_based_dt = self.extrapolated_time_step(dx_max)
            dt_max = min(dt_max, step_based_dt)
        return dt_max

    @property
    def constrained(self) -> bool:
        """bool: is unknown constrained by a limiting time step?"""
        constrained = lambda expr: numpy.isfinite(expr.eval()) if expr.constant else True
        return constrained(self.max_time_step_expr) or constrained(self.max_abs_step_expr)

    def extrapolated_time_step(self, step: float) -> float:
        """
        Time step necessary to attain a variation of `step`
        at a rate given by current value of the time derivative.
        """
        rate = numpy.abs(self.d_dt)
        step = numpy.where(rate > 0, abs(step), numpy.inf)
        rate = numpy.where(rate > 0, rate, 1)
        return numpy.min(step / rate)


class TimeUnknown(Boundary, AbstractTimeUnknown):
    """Time-dependent solver unknown.

    Parameters
    ----------
    context : cosapp.systems.System
        System in which the unknown is defined.
    name : str
        Name of the unknown

    Attributes
    ----------
    max_time_step : float
        Max time step authorized in one iteration; default numpy.inf
    """
    def __init__(self,
        context: System,
        name: str,
        der: Any,
        max_time_step: Union[Number, str] = numpy.inf,
        max_abs_step: Union[Number, str] = numpy.inf,
        pulled_from: Optional[VariableReference] = None,
    ):
        super().__init__(context, name)
        self._pulled_from = pulled_from
        self.__type = None
        self.__shape = None
        self.__dt_max = numpy.inf
        self.__dx_max = numpy.inf
        self.d_dt = der
        self.max_time_step = max_time_step
        self.max_abs_step = max_abs_step

    def update_mask(self) -> None:
        ref = self._ref
        self._mask = ref._mask = numpy.ones_like(getattr(ref._obj, ref._key), dtype=bool)

    def __str__(self) -> str:
        try:
            return str(self.value)
        except KeyError:  # does not exist in current context
            return str(self.default_value)

    @property
    def der(self) -> EvalString:
        """Expression of time derivative, given as an EvalString"""
        return self.__der

    @AbstractTimeUnknown.d_dt.setter
    def d_dt(self, expression: Any):
        eval_string, value, dtype = self.der_type(expression, self.context)
        if self.__type is None:
            self.__type = dtype
            if dtype is numpy.ndarray:
                self.__shape = value.shape
            elif dtype is not Number:
                raise TypeError(
                    f"Derivative expressions may only be numbers or array-like collections; got '{value}'")
        elif self.__type is not dtype:
            raise TypeError(
                f"Expression '{expression!s}' is incompatible with declared type {self.__type.__qualname__}")
        if self.__shape and numpy.shape(value) != self.__shape:
            raise ValueError(
                f"Expression '{expression!s}' should be an array of shape {self.__shape}")
        self.__der = eval_string

    @property
    def max_time_step_expr(self) -> EvalString:
        """Maximum admissible time step, given as an EvalString."""
        return self.__dt_max

    @property
    def max_abs_step_expr(self) -> EvalString:
        """Maximum admissible step, given as an EvalString."""
        return self.__dx_max

    @AbstractTimeUnknown.max_time_step.setter
    def max_time_step(self, expression: Any):
        self.__dt_max = self.__positive_expr(expression, "max_time_step")

    @AbstractTimeUnknown.max_abs_step.setter
    def max_abs_step(self, expression: Any):
        self.__dx_max = self.__positive_expr(expression, "max_abs_step")

    def __positive_expr(self, expression: Any, name: str) -> EvalString:
        eval_string, value, dtype = self.der_type(expression, self.context)
        check_arg(value, name, Number)  # checks that expression is scalar
        if value <= 0 and eval_string.constant:
            # Note:
            #   If expression is context-dependent (non-constant), it may turn out to be positive at time driver execution.
            #   Therefore, an exception should only be raised for constant, non-positive expressions.
            raise ValueError(f"{name} must be strictly positive")
        return eval_string

    def copy(self) -> TimeUnknown:
        """Copy time-dependent unknown object.

        Returns
        -------
        TimeUnknown
            Duplicated unknown
        """
        return TimeUnknown(self.context, self.name, self.der, self.max_time_step_expr)

    @staticmethod
    def der_type(expression: Any, context: System) -> Tuple[EvalString, Any, Type]:
        """Static method to evaluate the type and default value of an expression used as time derivative"""
        if isinstance(expression, EvalString):
            eval_string = expression
        else:
            eval_string = EvalString(expression, context)
        value = eval_string.eval()
        if isinstance(value, (list, tuple, numpy.ndarray)):
            value = numpy.array(value)
            dtype = numpy.ndarray
        elif TimeUnknown.is_number(value):
            dtype = Number
        else:
            dtype = type(value)
        return eval_string, value, dtype

    @staticmethod
    def is_number(value) -> bool:
        """Is value suitable for a derivative?"""
        return isinstance(value, Number) and not isinstance(value, bool)

    @property
    def pulled_from(self) -> Optional[VariableReference]:
        """VariableReference or None: Original time unknown before pulling; None otherwise."""
        return self._pulled_from

    @Boundary.value.setter
    def value(self, new: Union[Number, numpy.ndarray]) -> None:
        self.update_value(new)

    def to_dict(self) -> Dict[str, Any]:
        """Returns a JSONable representation of the transient unknown.
        
        Returns
        -------
        Dict[str, Any]
            JSONable representation
        """
        return {
            "context": self.context.contextual_name,
            "name": self.name,
            "der": str(self.__der),
            "max_time_step": str(self.max_time_step_expr),
        }

    def reset(self) -> None:
        """Reset transient unknown to a reference value.
        Inactive for class TimeUnknown."""
        pass


class TimeDerivative(Boundary):
    """Explicit time derivative.

    Parameters
    ----------
    context : cosapp.systems.System
        System in which the unknown is defined.
    name : str
        Name of the variable
    source : str
        Variable such that name = d(source)/dt
    initial_value : Any
        Time derivative initial value
    """
    def __init__(self,
        context: System,
        name: str,
        source: Any,
        initial_value: Any = None,
    ):
        super().__init__(context, name, inputs_only=True)
        self.__shape = None
        self.__previous = None
        eval_string, value, self.__type = self.source_type(source, self.context)

        if self.__type is Number:
            self._boundary_impl = ScalarBoundaryImpl()
            self._is_scalar = True
        elif self.__type is numpy.ndarray:
            self._boundary_impl = NumpyBoundaryImpl()
            self.__shape = value.shape
            self._is_scalar = False
        elif Boundary.is_mutable_sequence(value):
            self._boundary_impl = MutableSeqBoundaryImpl()
            self.__shape = (len(value), )
            self._is_scalar = False
        else:
            raise TypeError("Type of boundary value is not handle.")

        # Set source & initial value
        self.source = source
        self.initial_value = initial_value
        self.reset()

    def copy(self) -> TimeDerivative:
        der = super().copy()
        der.__src = copy.copy(self.__src)
        der.__shape = copy.copy(self.__shape)
        der.__initial = copy.copy(self.__initial)
        der.__previous = copy.copy(self.__previous)
        der.__type = self.__type
        return der

    def __str__(self) -> str:
        try:
            return str(self.value)
        except KeyError:  # does not exist in current context
            return str(self.default_value)

    def reset(self, value: Any = None) -> None:
        self.__previous = self.source
        if value is not None:
            self.initial_value = value  # NB: `value` may be an expression
        value = self.initial_value
        if value is not None:
            self.__set_value(value)

    @property
    def source_expr(self) -> EvalString:
        """Variable whose rate is evaluated, returned as an EvalString"""
        return self.__src

    @property
    def source(self) -> Union[Number, numpy.ndarray]:
        """Value of the variable whose rate is evaluated"""
        return self.__src.eval()

    @source.setter
    def source(self, expression: Any):
        self.__src = self.__parse(expression)
        if self.__previous is None:
            self.__previous = self.source

    @property
    def initial_value_expr(self) -> EvalString:
        """Initial value of time derivative, returned as an EvalString"""
        return self.__initial

    @property
    def initial_value(self) -> Union[Number, numpy.ndarray]:
        """Initial value of time derivative"""
        return self.__initial.eval()

    @initial_value.setter
    def initial_value(self, expression):
        if expression is None:
            self.__initial = EvalString(None, self.context)
        else:
            self.__initial = self.__parse(expression)

    def update(self, dt: Number) -> Number:
        """Evaluate rate-of-change of source over time interval `dt`"""
        current = self.source
        rate = (current - self.__previous) / dt  # backward finite-difference time derivative
        self.__set_value(rate)
        self.__previous = current
        return rate

    def __set_value(self, value: Union[Number, numpy.ndarray]):
        """Private setter for `value`"""
        if self._ref.value is None and not self._is_scalar:
            self.update_value(value)
            mask = numpy.ones_like(value, dtype=bool)
            self._ref = MaskedAttrRef.make_from_attr_ref(self._ref, self._ref._obj, self._ref._name, mask)
        else:
            self.update_value(value)

    @Boundary.value.setter
    def value(self, new: Union[Number, numpy.ndarray]) -> None:
        raise RuntimeError("Time derivatives are computed, and cannot be explicitly set")

    @staticmethod
    def source_type(expression: Any, context: System) -> Tuple:
        """Static method to evaluate the type and default value of an expression used as rate source"""
        eval_string, value, dtype = TimeUnknown.der_type(expression, context)
        if dtype is numpy.ndarray:
            value.fill(0.0)
        else:
            value = 0.0
        return eval_string, value, dtype

    def __parse(self, expression: Any) -> EvalString:
        eval_string, value, dtype = TimeUnknown.der_type(expression, self.context)
        if self.__type is not dtype:
            raise TypeError(
                f"Expression '{expression!s}' is incompatible with declared type {self.__type.__qualname__}")
        if self.__shape and value.shape != self.__shape:
            raise ValueError(f"Expression '{expression!s}' should be an array of shape {self.__shape}")
        return eval_string

    def to_dict(self) -> Dict[str, Any]:
        """Returns a JSONable representation of the time derivative.
        
        Returns
        -------
        Dict[str, Any]
            JSONable representation
        """
        return {
            "context": self.context.contextual_name,
            "name": self.name,
            "source": str(self.source_expr),
            "initial_value": str(self.initial_value_expr),
        }
