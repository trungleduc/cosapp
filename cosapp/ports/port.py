"""
Classes containing all `System` variables.
"""
import array
import logging
from collections import OrderedDict
from collections.abc import MutableSequence
from typing import Any, Dict, Iterator, Optional, Tuple, Union
from types import MappingProxyType
import numpy

from cosapp.patterns import visitor
from cosapp.core.numerics.distributions.distribution import Distribution
from cosapp.ports.enum import PortType, Scope, Validity
from cosapp.ports.exceptions import ScopeError
from cosapp.ports.variable import RangeValue, Types, Variable
from cosapp.utils.helpers import check_arg
from cosapp.utils.naming import NameChecker

logger = logging.getLogger(__name__)


class BasePort(visitor.Component):
    """Base class for ports, containers gathering variables.

    Common users should not use this class directly.

    Parameters
    ----------
    name : str
        Port name
    direction : {PortType.IN, PortType.OUT}
        Port direction
    """

    _name_check = NameChecker()

    def __init__(self, name: str, direction: PortType) -> None:
        """`BasePort` constructor.

        Parameters
        ----------
        name : str
            Port name
        direction : {PortType.IN, PortType.OUT}
            Port direction
        """
        if not isinstance(direction, PortType):
            raise TypeError(f"Direction must be PortType; got {direction}.")

        self._variables = OrderedDict()  # type: Dict[str, Variable]
        self._name = self._name_check(name)  # type: str
        self._direction = direction  # type: PortType
        self._owner = None  # type: Optional[cosapp.systems.System]
        self.__clearance = None
        self.scope_clearance = Scope.PRIVATE

    def accept(self, visitor: visitor.Visitor) -> None:
        """Specifies course of action when visited by `visitor`"""
        visitor.visit_port(self)

    @property
    def owner(self) -> "Optional[cosapp.systems.System]":
        """System : `System` owning the port."""
        return self._owner

    @owner.setter
    def owner(self, new_owner: "cosapp.systems.System"):
        from cosapp.systems import System

        if not isinstance(new_owner, System):
            raise TypeError(f"Port owner must be a `System`; got {new_owner!r}.")
        self._owner = new_owner
        self.__update_filter()

    @property
    def name(self) -> str:
        """str : Port name"""
        return self._name

    @property
    def contextual_name(self) -> str:
        """str : Join port owner name and port name.

        If the port has no owner, the port name is returned.
        """
        owner = self._owner
        return self._name if owner is None else f"{owner.name}.{self._name}"

    def full_name(self, trim_root=False) -> str:
        """Returns full name up to root owner.
        
        Parameters
        ----------
        trim_root : bool (optional, default False)
            Exclude root owner name if True.

        Returns
        -------
        str
            The port full name
        """
        owner = self.owner
        path = []
        if owner is not None:
            path = owner.path_namelist()
            if trim_root:
                path = path[1:]
        path.append(self.name)
        return ".".join(path)

    @property
    def direction(self) -> PortType:
        """:obj:`PortType.IN` or :obj:`PortType.OUT` : Port direction"""
        return self._direction

    @property
    def is_input(self) -> bool:
        """bool: True if port is an input, False otherwise."""
        return self._direction == PortType.IN

    @property
    def is_output(self) -> bool:
        """bool: True if port is an output, False otherwise."""
        return self._direction == PortType.OUT

    def __repr__(self) -> str:
        var_list = [f"'{key}': {getattr(self, key)!r}" for key in self._variables]
        return f"{type(self).__name__}: {{{', '.join(var_list)}}}"

    def _repr_markdown_(self) -> str:
        """Returns the representation of this port variables in Markdown format.

        Returns
        -------
        str
            Markdown formatted representation
        """
        from cosapp.tools.views.markdown import port_to_md

        return port_to_md(self)

    def __json__(self) -> Dict[str, Dict[str, Any]]:
        """JSONable dictionary representing a variable.
        
        Returns
        -------
        Dict[str, Any]
            The dictionary
        """
        rtn = dict()
        for name, var in self._variables.items():
            rtn[name] = var.__json__()

        return rtn

    def serialize_data(self) -> Dict[str, Any]:
        """Serialize the variable values in a dictionary.
        
        Returns
        -------
        Dict[str, Any]
            The dictionary (variable name, value)
        """
        rtn = dict()
        for name in self._variables:
            rtn[name] = getattr(self, name)
        return rtn

    def add_variable(
        self,
        name: str,
        value: Any = 1,
        unit: str = "",
        dtype: Types = None,
        valid_range: RangeValue = None,
        invalid_comment: str = "",
        limits: RangeValue = None,
        out_of_limits_comment: str = "",
        desc: str = "",
        distribution: Optional[Distribution] = None,
        scope: Scope = Scope.PRIVATE,
    ) -> None:
        """Add a variable to the port.

        The `valid_range` defines the range of value for which a model is known to behave correctly.
        The `limits` are at least as large as the validity range.

        Parameters
        ----------
        name : str
            Name of the variable
        value : Any, optional
            Value of the variable; default 1
        unit : str, optional
            Variable unit; default empty string (i.e. dimensionless)
        dtype : type or iterable of type, optional
            Variable type; default None (i.e. type of initial value)
        valid_range : Tuple[Any, Any] or Tuple[Tuple], optional
            Validity range of the variable; default None (i.e. all values are valid)
        invalid_comment : str, optional
            Comment to show in case the value is not valid; default ''
        limits : Tuple[Any, Any] or Tuple[Tuple], optional
            Limits over which the use of the model is wrong; default valid_range
        out_of_limits_comment : str, optional
            Comment to show in case the value is not valid; default ''
        desc : str, optional
            Variable description; default ''
        distribution : Distribution, optional
            Probability distribution of the variable; default None (no distribution)
        scope : Scope {PRIVATE, PROTECTED, PUBLIC}, optional
            Variable visibility; default PRIVATE
        """
        if name in self:
            logger.warning(
                f"Variable {name} already exists in port {self.contextual_name}."
                " It will be overwritten."
            )

        # Value must be set before creating variable as validation in Variable needs it
        self._variables[name] = Variable(
            name,
            self,
            value,
            unit,
            dtype,
            valid_range,
            invalid_comment,
            limits,
            out_of_limits_comment,
            desc,
            distribution,
            scope,
        )

        if self._variables[name].dtype == (
            MutableSequence,
            array.ArrayType,
            numpy.ndarray,
        ):
            value = numpy.asarray(value)

        # For efficiency reasons, variables are stored as port attributes
        setattr(self, name, value)

    # TODO should we override __setattr__ to forward setting to the source port? and/or raise an
    #   error if it is not possible - for example if the source is an output variable
    def __set_notype_checking(self, key, value):
        super().__setattr__(key, value)

        if key in self._variables and self._owner:
            self._owner.set_dirty(self.direction)

    def validate(self, key: str, value: Any) -> None:
        """Check if a variable is in the scope of the user and the type is valid.
        
        Parameters
        ----------
        key : str
            Name of the variable to test
        value : Any
            Value to validate

        Raises
        ------
        ScopeError
            If the variable is not visible for the user
        TypeError
            If the value has an unauthorized type
        """
        # Check scope
        if self.__out_of_scope(key):
            raise ScopeError(f"Cannot set out-of-scope variable {key!r}.")

        if value is not None:
            var_type = self._variables[key].dtype
            ok = var_type is None or isinstance(value, var_type)
            if not ok:
                raise TypeError(
                    "Trying to set {}.{} of type {} with {}.".format(
                        self.contextual_name, key, var_type, type(value)
                    )
                )

    def __set_variable(self, key, value):
        """Set the variable `key` with `value`.
        
        Parameters
        ----------
        key : str
            Name of the variable to be set
        value : Any
            Value to set
        """
        if key.startswith("_"):
            super().__setattr__(key, value)
        elif key in self._variables:
            self.validate(key, value)
            self.__set_notype_checking(key, value)
        elif hasattr(self, key):
            super().__setattr__(key, value)
        else:
            raise AttributeError(
                f"Port variable {self.contextual_name}.{key} can only be created using method 'add_variable'."
            )

    __setattr__ = __set_variable

    @staticmethod
    def set_type_checking(activate: bool) -> None:
        """(Un)set type checking when affecting port variables.

        By default type checking is activated.

        Parameters
        ----------
        activate : bool
            True to activate type checking, False to deactivate
        """
        if activate:
            BasePort.__setattr__ = BasePort.__set_variable
        else:
            BasePort.__setattr__ = BasePort.__set_notype_checking

    def __contains__(self, item: str) -> bool:
        return item in self._variables

    def __getitem__(self, item: str) -> Any:
        try:
            return getattr(self, item)
        except AttributeError:
            raise KeyError(f"Variable or property {item} does not exist in Port {self}.")

    def __setitem__(self, key: str, value: Any) -> None:
        if key in self._variables:
            setattr(self, key, value)
        else:
            raise KeyError(f"Variable {key} does not exist in Port {self}.")

    def __iter__(self) -> Iterator[str]:
        return iter(self._variables)

    def __len__(self) -> int:
        return len(self._variables)

    @property
    def scope_clearance(self) -> Scope:
        """Scope: Current clearance level of the port.
        Determines the set of read-only port variables."""
        return self.__clearance

    @scope_clearance.setter
    def scope_clearance(self, user_scope: Scope) -> None:
        check_arg(user_scope, "scope_clearance", Scope)
        self.__clearance = user_scope
        self.__update_filter()

    def __update_filter(self) -> None:
        """Update the implementation of method `out_of_scope`
        according to port's clearance level and owner."""
        if self._owner is None:
            criterion = lambda name: False
        else:
            read_only = [
                n for n, details in self._variables.items() if details.scope > self.__clearance
            ]
            criterion = lambda name: not self._owner.is_running() and name in read_only
        self.__out_of_scope = criterion

    def out_of_scope(self, name: str) -> bool:
        """
        Asserts if current scope `scope_clearance` is high enough to
        allow the modification of port variable `name`.

        Parameters
        ----------
        name : str
            Variable name to test

        Returns
        -------
        bool
            Is modification forbidden?

        The behavior of this function is set by method `scope_clearance()`.
        By default, clearance level is set to `Scope.PUBLIC`.

        Examples
        --------
        >>> port = Port("myPort", PortType.IN)
        >>> port.add_variable("x", 1.5, scope=Scope.PUBLIC)
        >>> port.add_variable("y", 0.2, scope=Scope.PROTECTED)
        >>> port.add_variable("z", 0.3, scope=Scope.PRIVATE)
        >>>
        >>> port.scope_clearance(Scope.PROTECTED)  # only `port.x` and `port.y` can be modified
        >>> assert not port.out_of_scope("x")
        >>> assert not port.out_of_scope("y")
        >>> assert port.out_of_scope("z")
        >>>
        >>> port.scope_clearance(Scope.PUBLIC)     # only `port.x` can be modified
        >>> assert not port.out_of_scope("x")
        >>> assert port.out_of_scope("y")
        >>> assert port.out_of_scope("z")
        """
        return self.__out_of_scope(name)

    def get_details(self, name: Optional[str] = None) -> Union[Dict[str, Variable], Variable]:
        """Return the variable(s).
        
        Parameters
        ----------
        name : str, optional
            Name of the variable looked for; default None (all variable are returned).

        Returns
        -------
        types.MappingProxyType[str, cosapp.ports.variable.Variable] or cosapp.ports.variable.Variable
            The sought variable, or a read-only view on all port variables.
        """
        if name is None:
            return MappingProxyType(self._variables)
        else:
            return self._variables[name]

    def check(self, name: Optional[str] = None) -> Union[Dict[str, Validity], Validity]:
        """Get the variable value validity.

        If `name` is not provided, returns a dictionary with the validity of all variables. Else
        only, the validity for the given variable will be returned.

        Parameters
        ----------
        name : str, optional
            Variable name; default None = All variables will be tested

        Returns
        -------
        Dict[str, Validity] or Validity
            (Dictionary of) the variable(s) value validity
        """
      
        if name is None:
            return dict(
                (name, variable.is_valid())
                for name, variable in self._variables.items()
            )
        else:           
            return self._variables[name].is_valid()

    def get_validity_ground(self,
        status: Validity,
        name: Optional[str] = None,
    ) -> Union[Dict[str, str], str]:
        # TODO unit tests
        """Get the ground arguments used to established the variable validity range.

        The status `Validity.OK` has no ground arguments.

        Parameters
        ----------
        status : Validity.{OK, WARNING, ERROR}
            Validity status for which the reasons are looked for.
        name : str, optional
            Variable name; default None = All variables will be tested

        Returns
        -------
        Dict[str, str] or str
            (Dictionary of) the variable(s) validity ground
        """
        if name is None:
            return dict(
                (name, variable.get_validity_comment(status))
                for name, variable in self._variables.items()
            )
        else:
            return self._variables[name].get_validity_comment(status)

    def copy(self,
        name: Optional[str] = None,
        direction: Optional[PortType] = None,
    ) -> "BasePort":
        """Duplicates the port.

        Parameters
        ----------
        name : str, optional
            Name of the duplicated port; default original port name
        direction : {PortType.IN, PortType.OUT}, optional
            Direction of the duplicated port; default original direction

        Returns
        -------
        Port
            The copy of the current port
        """
        new_name = self.name if name is None else name
        new_direction = self.direction if direction is None else direction

        new_port = type(self)(new_name, new_direction)
        for name, variable in self._variables.items():
            # Validation criteria are removed to avoid warning duplication when checking
            new_port.add_variable(
                name,
                getattr(self, name),
                unit=variable.unit,
                dtype=variable.dtype,
                desc=variable.description,
                distribution=variable.distribution,
                scope=variable.scope,
            )

        return new_port

    def morph(self, port: "BasePort") -> None:
        """Morph the provided port into this port.

        Morphing a port is useful when converting a `System` in something equivalent. The morphing
        goal is to preserve connections and adapt the port content if needed.

        Parameters
        ----------
        port : BasePort
            The port to morph to
        """

        # TODO unit test for variable details when morphing
        for name, variable in self._variables.items():
            if name not in port:
                port.add_variable(
                    name,
                    getattr(self, name),
                    unit=variable.unit,
                    dtype=variable.dtype,
                    desc=variable.description,
                    valid_range=variable.valid_range,
                    invalid_comment=variable.invalid_comment,
                    limits=variable.limits,
                    out_of_limits_comment=variable.out_of_limits_comment,
                    distribution=variable.distribution,
                    scope=variable.scope,
                )

        for variable in list(port):
            if variable not in self:
                port.remove_variable(variable)

    def to_dict(self, with_def: bool = False) -> Dict[str, Union[str, Tuple[Dict[str, str], str]]]:
        """Convert this port in a dictionary.
   
        Parameters
        ----------
        with_def : bool
            Flag to export also output ports and its class name (default: False).

        Returns
        -------
        dict
            The dictionary representing this port.
        """
        # TODO this is uncomplete as validation ranges and distribution could be changed
        new_dict = dict()

        if with_def:
            tmp =  dict()
            if self.name not in ["inwards", "outwards"]:
                tmp["__class__"] = self.__class__.__qualname__
                for variable in self:
                    tmp[variable] = getattr(self, variable)
            else:
                for v_name, variable in self._variables.items():
                    tmp[v_name]= variable.to_dict() 

            new_dict[self.name] = tmp
        
        else:
            if self.is_input:
                for variable in self:
                    fullname = f"{self.name}.{variable}"
                    new_dict[fullname] = getattr(self, variable)
        
        return new_dict



class ExtensiblePort(BasePort):
    """Class describing ports with a varying number of variables."""

    # TODO unused and should be removed -> to dangerous for consistency
    def remove_variable(self, name: str) -> None:
        """Removes a variable from the port.

        Parameters
        ----------
        name : str
            Name of the variable to be removed

        Raises
        ------
        AttributeError
            If the variable does not exists
        """
        if name not in self._variables:
            raise AttributeError(
                f"Variable {name!r} does not exist in port {self.contextual_name}"
            )
        delattr(self, name)
        self._variables.pop(name)


class Port(BasePort):
    """A `Port` is a container gathering variables tightly linked.

    An optional dictionary may be specified to overwrite variable value and some of their
    metadata. Values of the dictionary may be of two kinds:
    - Only the new value
    - A dictionary with new value and/or new details

    If the value or a metadata is set to `None`, the default value will be kept.

    Parameters
    ----------
    name : str
        `Port` name
    direction : {PortType.IN, PortType.OUT}
        `Port` direction
    variables : Dict[str, Any], optional
        Dictionary of variables with their value and details; default: None = default value

    Attributes
    ----------
    _locked : bool
        if True, `add_variable` is deactivated. This is the default behavior outside the `setup`
        function.

    Examples
    --------

    Use this class by subclassing it:

    >>> class FlowPort(Port):
    >>>     def setup(self):
    >>>         self.add_variable('Pt', 101325., unit='Pa')
    >>>         self.add_variable('Tt', 273.15, unit='K')
    >>>         self.add_variable('W', 1.0, unit='kg/s')
    >>>
    >>> p = FlowPort('myPort', PortType.IN)
    >>> # Overwrite value and some details
    >>> f = FlowPort(
    >>>     'myPort2', 
    >>>     PortType.OUT,
    >>>     {
    >>>         'Pt': 10e6,
    >>>         'W': {
    >>>             'value': 10.,  # New value
    >>>             'unit': 'lbm/s',  # New unit - should be compatible of the original one
    >>>             'valid_range': (0., None),  # Updated validated range
    >>>             'invalid_comment': 'Flow should be positive',  # Comment if out of validated range
    >>>             'limits': None,  # Limits
    >>>             'out_of_limits_comment': '',  # Comment if out of limits
    >>>         }
    >>>     }
    >>> )
    """

    def __init__(
        self,
        name: str,
        direction: PortType,
        variables: Optional[Dict[str, Any]] = None,
    ) -> None:
        """`Port` constructor.

        An optional dictionary may be specified to overwrite variable value and some of their
        metadata. Values of the dictionary may be of two kinds:
        - Only the new value
        - A dictionary with new value and/or new details

        Parameters
        ----------
        name : str
            `Port` name
        direction : {PortType.IN, PortType.OUT}
            `Port` direction
        variables : Dict[str, Any], optional
            Dictionary of variables with their value and details; default: None = default value
        """
        check_arg(variables, "variables", (type(None), dict))

        super().__init__(name, direction)
        self._locked = False  # type: bool

        self.setup()

        if variables is not None:
            for name, value in variables.items():
                variable_value = value
                if isinstance(value, dict):
                    variable_value = value.pop("value", None)

                    details = self.get_details(name)
                    try:
                        for field, param in value.items():
                            setattr(details, field, param)
                    except AttributeError:  # Unexpected keyword for details
                        # If current variable is of type dict
                        if isinstance(self[name], dict):
                            if variable_value is not None:
                                value["value"] = variable_value
                            variable_value = value
                        else:
                            raise

                if variable_value is not None:
                    self[name] = variable_value

        self._locked = True

    def setup(self) -> None:
        """`Port` variables are defined in this function by calling `add_variable`.

        This function allows to populate a customized `Port` class. The `add_variable`
        function is only callable in the frame of this function.

        Examples
        --------

        Here is an example of `Port` subclassing:

        >>> class FlowPort(Port):
        >>>     def setup(self):
        >>>         self.add_variable('Pt', 101325.)
        >>>         self.add_variable('Tt', 273.15)
        >>>         self.add_variable('W', 1.0)
        """
        pass  # pragma: no cover

    def add_variable(
        self,
        name: str,
        value: Any = 1,
        unit: str = "",
        dtype: Types = None,
        valid_range: RangeValue = None,
        invalid_comment: str = "",
        limits: RangeValue = None,
        out_of_limits_comment: str = "",
        desc: str = "",
        distribution: Optional[Distribution] = None,
        scope=Scope.PUBLIC,
    ) -> None:
        """Add a variable to the port.

        The `valid_range` defines the range of value for which a model is known to behave correctly.
        The `limits` are at least as large as the validity range.

        Notes
        -----
        The default visibility of a `Port` variable is `Scope.PUBLIC`.
        This is a difference compared to `BasePort`.

        Parameters
        ----------
        name : str
            Name of the variable
        value : Any, optional
            Value of the variable; default 1
        unit : str, optional
            Variable unit; default empty string (i.e. dimensionless)
        dtype : type or iterable of type, optional
            Variable type; default None (i.e. type of initial value)
        valid_range : Tuple[Any, Any] or Tuple[Tuple], optional
            Validity range of the variable; default None (i.e. all values are valid)
        invalid_comment : str, optional
            Comment to show in case the value is not valid; default ''
        limits : Tuple[Any, Any] or Tuple[Tuple], optional
            Limits over which the use of the model is wrong; default valid_range
        out_of_limits_comment : str, optional
            Comment to show in case the value is not valid; default ''
        desc : str, optional
            Variable description; default ''        
        distribution : Distribution, optional
            Probability distribution of the variable; default None (no distribution)
        scope : Scope {PRIVATE, PROTECTED, PUBLIC}, optional
            Variable visibility; default PUBLIC
        """
        if self._locked:
            raise AttributeError("add_variable cannot be called outside `setup`.")

        super().add_variable(
            name,
            value=value,
            unit=unit,
            dtype=dtype,
            valid_range=valid_range,
            invalid_comment=invalid_comment,
            limits=limits,
            out_of_limits_comment=out_of_limits_comment,
            desc=desc,
            distribution=distribution,
            scope=scope,
        )

    def copy(
        self, name: Optional[str] = None, direction: Optional[PortType] = None
    ) -> "BasePort":
        """Duplicates the port.

        Parameters
        ----------
        name : str, optional
            Name of the duplicated port; default original port name
        direction : {PortType.IN, PortType.OUT}, optional
            Direction of the duplicated port; default original direction

        Returns
        -------
        Port
            The copy of the current port
        """
        new_name = self.name if name is None else name
        new_direction = self.direction if direction is None else direction

        new_port = type(self)(new_name, new_direction)

        for name in self:
            new_port[name] = self[name]

        return new_port
