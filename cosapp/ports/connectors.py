"""
Classes connecting `Port` of foreign `System` to transfer variable values.
"""
from __future__ import annotations
import abc
import copy
import logging
import weakref
from types import MappingProxyType
from typing import (
    Callable, Iterable, Iterator,
    Collection, Mapping, Dict, List, Tuple,
    Optional, Union, Any, Type,
    TYPE_CHECKING,
)

from cosapp.ports import units
from cosapp.ports.port import BasePort, Port
from cosapp.utils.helpers import check_arg, is_numerical
if TYPE_CHECKING:
    from cosapp.systems import System

logger = logging.getLogger(__name__)


class ConnectorError(Exception):
    """Raised if a connector cannot be created between two `Port`."""


class BaseConnector(abc.ABC):
    """This class connect two ports without enforcing that all port variables are connected.

    The link is oriented from the source to the sink.

    Parameters
    ----------
    name : str
        Name of the connector
    source : BasePort
        Port from which originate the variables
    mapping : str or List[str] or Dict[str, str]
        (List of) common name(s) or mapping name dictionary
    sink : BasePort
        Port to which the variables are transferred
    """

    def __init__(
        self,
        name: str,
        sink: BasePort,
        source: BasePort,
        mapping: Union[str, List[str], Dict[str, str], None] = None,
    ):
        """Connector constructor from the two `BasePort` to link and the list of variables to map.

        If no mapping is provided, connection will be made between variables based on their names. 
        If a name mapping is provided as a list, the name should be present in both port. And if 
        the mapping is specified as a dictionary, the keys belong to the sink port and the values
        to the source port.

        Parameters
        ----------
        name : str
            Name of the connector
        sink : BasePort
            Port to which the variables are transferred.
        source : BasePort
            Port from which originate the variables.
        mapping : str or List[str] or Dict[str, str], optional
            (List of) common name(s) or mapping name dictionary; default None (i.e. no mapping).
        
        Raises
        ------
        ConnectorError
            If the connection between the `source` and the `sink` is not possible to establish.
        """
        self.__check_port(sink, 'sink')
        self.__check_port(source, 'source')
        check_arg(name, 'name', str, lambda s: len(s.strip()) > 0)

        if source is sink:
            raise ConnectorError("Source and sink cannot be the same object.")

        # Generate mapping dictionary
        if mapping is None:
            mapping = dict((v, v) for v in sink if v in source)
        else:
            mapping = self.format_mapping(mapping)

        self._name = name  # type: str
        self._mapping = mapping  # type: Dict[str, str]
        self._source = self.__get_port(source, sink=False, check=False)  # type: weakref.ReferenceType[BasePort]
        self._sink = self.__get_port(sink, sink=True, check=False)  # type: weakref.ReferenceType[BasePort]

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
        d = self.__dict__.copy()
        d.update({"_source": self.source, "_sink": self.sink})
        return d

    def __json__(self) -> Dict[str, Any]:
        """Creates a JSONable dictionary representation of the object.
        
        Break circular dependencies by removing some slots from the 
        state.

        Returns
        -------
        Dict[str, Any]
            The dictionary
        """
        d = self.__dict__.copy()
        d.pop("_source")
        d.pop("_sink")
        d.update({"info": self.info()})
        return d

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Sets the object from a provided state.

        Parameters
        ----------
        state : Dict[str, Any]
            State
        """
        self.__dict__.update(state)
        source = weakref.ref(state.pop("_source"))
        sink = weakref.ref(state.pop("_sink"))
        self.__dict__.update({"_source": source, "_sink": sink})

    @abc.abstractmethod
    def transfer(self) -> None:
        """Transfer values from `source` to `sink`."""
        pass

    def __repr__(self) -> str:
        mapping = self._mapping
        if self.preserves_names():
            mapping = list(mapping)
        return "{}({} <- {}, {})".format(
            type(self).__qualname__,
            self.sink.contextual_name,
            self.source.contextual_name,
            mapping,
        )

    @property
    def name(self) -> str:
        """str : name of the connector."""
        return self._name

    @property
    def source(self) -> BasePort:
        """`BasePort`: Port from which transferred values originate."""
        return self._source()

    @source.setter
    def source(self, port: BasePort) -> None:
        self._source = self.__get_port(port, sink=False, check=True)

    @property
    def sink(self) -> BasePort:
        """`BasePort`: Port to which values are transferred."""
        return self._sink()

    @sink.setter
    def sink(self, port: BasePort) -> None:
        self._sink = self.__get_port(port, sink=True, check=True)

    def __get_port(self, port: BasePort, sink: bool, check=True) -> weakref.ref[BasePort]:
        """Returns a weakref to `port`, after compatibility check with internal mapping."""
        port_type = 'sink' if sink else 'source'

        if check:
            self.__check_port(port, port_type)

        varnames: Iterator[str] = getattr(self, f"{port_type}_variables")()
        not_found = list(
            filter(lambda name: name not in port, varnames)
        )
        if not_found:
            if len(not_found) == 1:
                message = f"variable {not_found[0]!r} does not exist"
            else:
                message = f"variables {not_found} do not exist"
            raise ConnectorError(
                f"{port_type.title()} {message} in port {port}."
            )
        
        return weakref.ref(port)

    @staticmethod
    def __check_port(port: BasePort, name: str) -> None:
        check_arg(port, name, BasePort, stack_shift=1)

        if not port.owner:
            raise ConnectorError(f"{name.title()} owner is undefined.")

    def __len__(self) -> int:
        return len(self._mapping)

    @property
    def mapping(self) -> Dict[str, str]:
        """Dict[str, str] : Variable name mapping between the sink (key) and the source (value)."""
        return MappingProxyType(self._mapping)

    def sink_variables(self) -> Iterator[str]:
        return self._mapping.keys()

    def source_variables(self) -> Iterator[str]:
        return self._mapping.values()

    def sink_variable(self, source_variable: str) -> str:
        """Returns the name of the sink variable associated to `source_variable`"""
        for sink, source in self._mapping.items():
            if source == source_variable:
                return sink
        raise KeyError(source_variable)

    def source_variable(self, sink_variable: str) -> str:
        """Returns the name of the source variable associated to `sink_variable`"""
        return self._mapping[sink_variable]

    def contextual_name(self, context: Optional[System]=None, with_mapping=True) -> str:
        """Contextual name of the connector, of the kind system[source -> sink]."""
        name = self.name
        sink = self.sink
        source = self.source
        if source.owner.parent is sink.owner.parent:
            owner = source.owner.parent
        elif source.owner is sink.owner.parent:
            owner = source.owner
        else:
            owner = sink.owner
        if context:
            context_name = context.get_path_to_child(owner)
        else:
            context_name = owner.full_name()
        def contextual_portname(port: Port) -> str:
            return port.name if port.owner is owner else port.contextual_name
        sink_name = contextual_portname(sink)
        source_name = contextual_portname(source)
        to = " \u27F6 "  # long right arrow
        # to = " \u2794 "  # Heavy Wide-Headed Right Arrow
        source_to_sink = f"{source_name}{to}{sink_name}"
        if context_name:
            connector_name = f"{context_name}[{source_to_sink}]"
        else:
            connector_name = source_to_sink
        if with_mapping and not self.is_mirror():
            mapping = self.pretty_mapping()
        else:
            mapping = ""
        return f"{connector_name} ({mapping})" if mapping else connector_name

    def pretty_mapping(self) -> str:
        """Pretty formatting of the variable name mapping applied by the connector."""
        # to = " \u2192 "  # short right arrow with spaces
        to = "\u27F6"  # long right arrow, no spaces
        mapping = ", ".join(
            origin if origin == target else f"{origin}{to}{target}"
            for target, origin in self._mapping.items()
        )
        return mapping

    def preserves_names(self) -> bool:
        """Returns `True` if connector mapping preserves variable names,
        `False` otherwise."""
        return all(target == origin for target, origin in self._mapping.items())

    def is_mirror(self) -> bool:
        """Returns `True` if connector is an identical, one-to-one mapping
        between two ports of the same kind; `False` otherwise."""
        sink, source = self.sink, self.source
        return (
            type(sink) is type(source)
            and isinstance(sink, Port)
            and len(self) == len(sink)
            and self.preserves_names()
        )

    def update_mapping(self, mapping: Dict[str, str]) -> None:
        """Extend current mapping with additional dictionary.

        Parameters
        ----------
        mapping : Dict[str, str]
            Variable name mapping extending current mapping.
        """
        self._mapping.update(mapping)

    def remove_variables(self, names: Iterable[str]) -> None:
        """Remove the provided variables from this connection.

        The provided names should be sink names.

        Parameters
        ----------
        names : Iterable[str]
            Collection of sink variable names to be removed.
        """
        for variable in names:
            del self._mapping[variable]

    def info(self) -> Union[Tuple[str, str], Tuple[str, str, Dict[str, str]]]:
        """Returns connector information in a tuple.

        If the name mapping is complete, with identical names,
        it is omitted, and the output tuple is formatted as:
        - (target_name, source_name)

        Otherwise, output is formatted as:
        - (target_name, source_name, name_mapping)

        Returns
        -------
        tuple
            Tuple representing connector
        """
        # If the mapping is full and with the same nomenclature
        target, origin = self.port_names()
        same_nomenclature = False
        if len(self._mapping) == len(self.source) == len(self.sink):
            same_nomenclature = self.preserves_names()
        if same_nomenclature:
            info = (target, origin)
        else:
            info = (target, origin, self._mapping.copy())
        return info

    def to_dict(self) -> Dict[str, Union[Tuple[str, str], Tuple[str, str, Dict[str, str]]]]:
        """Converts connector into a single-key dictionary.
        The key is the connector name; associated value
        is the tuple returned by method `info()`.

        Returns
        -------
        dict
            Dictionary {name: info_tuple} representing connector
        """
        return {self.name: self.info()}

    def port_names(self) -> Tuple[str, str]:
        """Returns source and sink contextual names as a str tuple.

        Returns
        -------
        tuple
            (source_name, sink_name) tuple
        """
        source, sink = self.source, self.sink

        if source.owner is sink.owner.parent:
            origin = source.name
        else:
            origin = source.contextual_name

        if sink.owner is source.owner.parent:
            target = sink.name
        else:
            target = sink.contextual_name
        
        return target, origin

    @staticmethod
    def format_mapping(mapping: Union[str, Collection[str], Dict[str, str]], /) -> Dict[str, str]:
        """Returns suitable name mapping for connectors,
        from different kinds of argument `mapping`.

        Parameters:
        -----------
        - mapping [str | list[str] | dict[str, str] | None]:
            Name mapping, given as either a string (single variable),
            a collection of strings, or a full name mapping, as a dictionary.
        
        Returns:
        --------
        dict[str, str]: name mapping suitable for connectors.
        """
        name_mapping = {}
        if not isinstance(mapping, Iterable):
            raise TypeError(
                f"name mappings must be either str, collection[str|dict] or dict[str, str]; got {mapping!r}."
            )
        if mapping:
            if isinstance(mapping, str):
                name_mapping = {mapping: mapping}
            elif isinstance(mapping, Mapping):
                name_mapping = dict(mapping)
            elif isinstance(mapping, Collection):
                name_mapping = dict()
                for obj in mapping:
                    if isinstance(obj, str):
                        name_mapping[obj] = obj
                    elif isinstance(obj, Mapping):
                        name_mapping.update(obj)
                    else:
                        raise TypeError(
                            f"when given as a collection, a name mapping may only contain str or dict[str, str] items; got {mapping!r}."
                        )
        return name_mapping


class Connector(BaseConnector):
    """Shallow copy connector.
    See `BaseConnector` for base class details.
    """
    def __init__(
        self,
        name: str,
        sink: BasePort,
        source: BasePort,
        mapping: Union[str, List[str], Dict[str, str], None] = None,
    ):
        super().__init__(name, sink, source, mapping)

        self._unit_conversions = {} # type: Dict[str, Optional[Tuple[float, float]]]
        self._transfer_func = {}  # type: Dict[str, Callable[[Any], Any]]
        self.update_unit_conversion()

    @BaseConnector.source.setter
    def source(self, port: BasePort) -> None:
        cls = self.__class__
        super(cls, cls).source.__set__(self, port)
        self.update_unit_conversion()

    @BaseConnector.sink.setter
    def sink(self, port: BasePort) -> None:
        cls = self.__class__
        super(cls, cls).sink.__set__(self, port)
        self.update_unit_conversion()

    def update_mapping(self, mapping: Dict[str, str]) -> None:
        super().update_mapping(mapping)
        self.update_unit_conversion()

    def remove_variables(self, names: Iterable[str]) -> None:
        super().remove_variables(names)
        for variable in names:
            del self._unit_conversions[variable]
            del self._transfer_func[variable]

    def update_unit_conversion(self, name: Optional[str] = None) -> None:
        """Update the physical unit conversion on the connector.

        If `name` is not `None`, update the conversion only for the connexion towards that variable.

        Parameters
        ----------
        name : str, optional
            Name of the variable for which unit conversion needs an update; default None (i.e. all
            conversions will be updated).

        Raises
        ------
        UnitError
            If unit conversion from source to sink is not possible
        """
        source, sink = self.source, self.sink
        mapping = self.mapping

        def update_one_connection(key: str) -> None:
            """Update the unit converter of the connected key.

            Parameters
            ----------
            key : str
                Name of the connected variable for which the unit converter should be updated.
            """
            target = sink.get_details(key)
            origin = source.get_details(mapping[key])
            has_unit = {
                'target': bool(target.unit),
                'origin': bool(origin.unit),
            }
            if has_unit['origin'] != has_unit['target']:
                # Send a warning if one end of the connector
                # has a physical unit and the other is dimensionless
                message = lambda origin_status, target_status: (
                    f"Connector source {origin.full_name!r} {origin_status}, but target {target.full_name!r} {target_status}."
                )
                if has_unit['origin']:
                    logger.warning(
                        message(f"has physical unit {origin.unit}", "is dimensionless")
                    )
                else:
                    logger.warning(
                        message("is dimensionless", f"has physical unit {target.unit}")
                    )
            # Get conversion constants between units
            constants = units.get_conversion(origin.unit, target.unit)
            if constants is None and is_numerical(sink[key]):
                constants = (1.0, 0.0)  # legacy - tests be must changed if suppressed
            self._unit_conversions[key] = constants

            # Get transfer function
            try:
                factor, offset = self._unit_conversions[key]
            except:
                pass
            else:
                if (factor, offset) != (1, 0):
                    self._transfer_func[key] = lambda value: factor * (value + offset)

        if name is None:
            for name in mapping:
                update_one_connection(name)
        else:
            update_one_connection(name)

    def transfer(self) -> None:
        source, sink = self.source, self.sink

        for target, origin in self._mapping.items():
            # get/setattr faster for Port
            value = getattr(source, origin)
            transfer = self._transfer_func.get(target, copy.copy)
            setattr(sink, target, transfer(value))


def MakeDirectConnector(classname: str, transform: Optional[Callable]=None, **kwargs) -> Type[BaseConnector]:
    """Connector factory using a simple transfer function, with no unit conversion.
    """
    if transform is None:
        transfer_attr = setattr
    else:
        transfer_attr = lambda sink, name, value: setattr(sink, name, transform(value))

    class DirectConnector(BaseConnector):
        """Connector with simple transfer function"""
        def transfer(self) -> None:
            source, sink = self.source, self.sink

            for target, origin in self._mapping.items():
                value = getattr(source, origin)
                transfer_attr(sink, target, value)

    return type(classname, (DirectConnector,), kwargs)


PlainConnector = MakeDirectConnector(
    "PlainConnector",
    __doc__ = (
        "Plain assignment connector, with no unit conversion."
        " Warning: may generate common references between sink and source variables."
    ),
)

CopyConnector = MakeDirectConnector(
    "CopyConnector", copy.copy,
    __doc__ = "Shallow copy connector, with no unit conversion.",
)

DeepCopyConnector = MakeDirectConnector(
    "DeepCopyConnector", copy.deepcopy,
    __doc__ = "Deep copy connector, with no unit conversion.",
)
