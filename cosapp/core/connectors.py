"""
Classes connecting `Port` of foreign `System` to transfer variable values.
"""
import copy
import logging
import weakref
from typing import Callable, Dict, List, Optional, Tuple, Union

from cosapp.ports import units
from cosapp.ports.port import ExtensiblePort, PortType
from cosapp.utils.helpers import check_arg, is_numerical

logger = logging.getLogger(__name__)


class ConnectorError(Exception):
    """Raised if a connector cannot be created between two `Port`.

    Attributes
    ----------
    message : str
        Error message
    """

    def __init__(self, message: str):
        """Instantiate a error object from the error descriptive message.

        Parameters
        ----------
        message : str
            Error message
        """
        self.message = message


class Connector:
    """This class connect two ports without enforcing that all port variables are connected.

    The link is oriented from the source to the sink.

    Parameters
    ----------
    name : str
        Name of the connector
    source : ExtensiblePort
        Port from which originate the variables
    mapping : str or List[str] or Dict[str, str]
        (List of) common name(s) or mapping name dictionary
    sink : ExtensiblePort
        Port to which the variables are transferred
    """

    def __init__(
        self,
        name: str,
        sink: ExtensiblePort,
        source: ExtensiblePort,
        mapping: Union[str, List[str], Dict[str, str], None] = None,
    ):
        """Connector constructor from the two `ExtensiblePort` to link and the list of variables to map.

        If no mapping is provided, connection will be made between variables based on their names. 
        If a name mapping is provided as a list, the name should be present in both port. And if 
        the mapping is specified as a dictionary, the keys belong to the sink port and the values
        to the source port.

        Parameters
        ----------
        name : str
            Name of the connector
        sink : ExtensiblePort
            Port to which the variables are transferred.
        source : ExtensiblePort
            Port from which originate the variables.
        mapping : str or List[str] or Dict[str, str], optional
            (List of) common name(s) or mapping name dictionary; default None (i.e. no mapping).
        
        Raises
        ------
        ConnectorError
            If the connection between the `source` and the `sink` is not possible to establish.
        """
        self._name = name  # type: str

        if not source.owner:
            raise ConnectorError("Source owner is undefined.")

        if not sink.owner:
            raise ConnectorError("Sink owner is undefined.")

        if source is sink:
            raise ConnectorError("Source and sink cannot be the same object.")

        self._source = weakref.ref(source)  # type: ExtensiblePort
        self._sink = weakref.ref(sink)  # type: ExtensiblePort

        # Generate mapping dictionary
        if mapping is None:
            mapping = dict([(v, v) for v in sink if v in source])
        elif isinstance(mapping, str):
            mapping = dict([(mapping, mapping)])
        elif not isinstance(mapping, dict):
            mapping = dict(zip(mapping, mapping))

        self._mapping = mapping  # type: Dict[str, str]

        # Validate the connection based on variable attributes
        for target, origin in self._mapping.items():
            if origin not in self.source:
                raise ConnectorError(
                    f"Source port {self.source} has no variable '{origin}'"
                )
            if target not in self.sink:
                raise ConnectorError(
                    f"Sink port {self.source} has no variable '{target}'"
                )

        self._unit_conversions = dict(
            [(name, None) for name in self._mapping]
        )  # type: Dict[str, Optional[Tuple[float, float]]]

        self.update_unit_conversion()

    def __repr__(self) -> str:
        return "Connector({} <- {}, {})".format(
            self.sink.contextual_name,
            self.source.contextual_name,
            self.variable_mapping,
        )

    @property
    def name(self) -> str:
        """str : name of the connector."""
        return self._name

    @property
    def source(self) -> ExtensiblePort:
        """`ExtensiblePort`: Port from which transferred values originate."""
        return self._source()

    @source.setter
    def source(self, port: ExtensiblePort) -> None:
        check_arg(port, 'source', ExtensiblePort)

        if len(self.source) != len(port):
            raise TypeError(
                f"New port {port} has not the same variables as the source {self.source}."
            )

        for variable in self.source:
            if variable not in port:
                raise AttributeError(
                    f"Source variable {variable} does not exist in port {port}."
                )

        self._source = weakref.ref(port)

    @property
    def sink(self) -> ExtensiblePort:
        """`ExtensiblePort`: Port to which values are transferred."""
        return self._sink()

    @sink.setter
    def sink(self, port: ExtensiblePort) -> None:
        check_arg(port, 'sink', ExtensiblePort)

        if len(self.sink) != len(port):
            raise TypeError(
                f"New port {port} has not the same variables as the sink {self.sink}."
            )

        for variable in self.sink:
            if variable not in port:
                raise AttributeError(
                    f"Sink variable {variable} does not exist in port {port}."
                )

        self._sink = weakref.ref(port)

    @property
    def variable_mapping(self) -> Dict[str, str]:
        """Dict[str, str] : Variable name mapping between the sink (key) and the source (value)."""
        return self._mapping

    def set_perturbation(self, name: str, value: float) -> None:
        """Add a perturbation on a connector.
        
        Parameters
        ----------
        name : str
            Name of the sink variable to perturb
        value : float
            Perturbation value
        """
        if self._unit_conversions[name] is None:
            ValueError(f"'{name}' does not have a numerical type.")

        s, a = self._unit_conversions[name]
        self._unit_conversions[name] = s, a + value
        self.source.owner.set_dirty(PortType.IN)

    def remove_variables(self, names: List[str]) -> None:
        """Remove the provided variables from this connection.

        The provided names should be sink names.

        Parameters
        ----------
        names : List[str]
            list of sink variable names to be removed.
        """
        for variable in names:
            del self._mapping[variable]
            del self._unit_conversions[variable]

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
        def update_one_connection(key: str) -> None:
            """Update the unit converter of the connected key.

            Parameters
            ----------
            key : str
                Name of the connected variable for which the unit converter should be updated.
            """
            source_name = self.variable_mapping[key]
            origin = self.source.get_details(source_name)
            target = self.sink.get_details(key)
            # Get the conversion between unit and check it is valid
            converter = units.get_conversion(origin.unit, target.unit)
            if converter is None:
                # Print a warning if one hand of the connexion is a valid unit but not the other one
                message = lambda origin_status, target_status: (
                    f"Connector source {origin.full_name!r} {origin_status}, but target {target.full_name!r} {target_status}."
                )
                if units.is_valid_units(origin.unit):
                    logger.warning(
                        message(f"has physical unit {origin.unit}", "is dimensionless")
                    )
                elif units.is_valid_units(target.unit):
                    logger.warning(
                        message("is dimensionless", f"has physical unit {target.unit}")
                    )

                if is_numerical(self.sink[key]):
                    converter = (1.0, 0.0)

            self._unit_conversions[key] = converter

        if name is None:
            for name in self._unit_conversions:
                update_one_connection(name)
        else:
            update_one_connection(name)

    def transfer(self) -> None:
        """Transfer values from `source` to `sink`."""
        # TODO improve efficiency
        def conversion_function(key) -> Callable:
            try:
                slope, offset = self._unit_conversions[key]
            except:
                converter = copy.deepcopy
            else:
                converter = lambda var: slope * (var + offset)
            return converter

        source, sink = self.source, self.sink

        if not source.owner.is_clean(source.direction):
            sink.owner.set_dirty(sink.direction)

            for key, item in self._mapping.items():
                # get/setattr faster for Port
                target = getattr(source, item)
                convert = conversion_function(key)
                try:
                    setattr(sink, key, convert(target))
                except TypeError:
                    setattr(sink, key, copy.deepcopy(target))

    def to_dict(self) -> Dict[str, Union[Tuple[str, str], Tuple[str, str, Dict[str, str]]]]:
        """Convert connector to a single-key dictionary.

        The key is the connector name; the associated value
        depends on the variable name mapping.

        If the mapping is complete, with identical names,
        the name mapping is omitted, and the output
        dictionary is formatted as:
            {name: (target_name, source_name)}

        Otherwise, it is formatted as:
            {name: (target_name, source_name, name_mapping)}

        Returns
        -------
        dict
            Dictionary representing this connector
        """
        # If the mapping is full and with the same nomenclature
        same_nomenclature = False
        if len(self._mapping) == len(self.source) == len(self.sink):
            same_nomenclature = True
            for k, v in self._mapping.items():
                if k != v:
                    same_nomenclature = False
                    break

        target, origin = self.port_names()
        if same_nomenclature:
            info = (target, origin)
        else:
            info = (target, origin, self._mapping.copy())

        return {self.name: info}

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
