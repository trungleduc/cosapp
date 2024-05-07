from __future__ import annotations
import logging
from typing import Any, Union, Collection, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from cosapp.systems import System

logger = logging.getLogger(__name__)


def pull_variables(
    child: System, 
    pulling: Union[str, Collection[str], Dict[str, str]],
) -> None:
    """Pull variables from child to the parent.

    Parameters
    ----------
    child: System
        `System` exposing variables to its parent API.
    pulling [str | collection[str] | dict[str, str]]:
        Map of child ports to pulled ports at the parent system level.
    """
    from cosapp.systems import System
    from cosapp.ports.port import BasePort, Port
    from cosapp.ports.connectors import BaseConnector
    from cosapp.utils.naming import CommonPorts

    parent: System = child.parent
    if parent is None:
        raise AttributeError(
            f"Can't pull variables from orphan System {child.name!r}"
        )
    method_map = {
        CommonPorts.INWARDS.value: 'add_inward',
        CommonPorts.OUTWARDS.value: 'add_outward',
        CommonPorts.MODEVARS_IN.value: 'add_inward_modevar',
        CommonPorts.MODEVARS_OUT.value: 'add_outward_modevar',
    }
    def log_debug(parent_attr, child_attr) -> str:
        logger.debug(
            f"{parent_attr} has been duplicated from {child_attr}"
            f" - including validation range and scope."
        )
    def copy_variable(
        child: System,
        port_name: str,
        child_var: str,
        parent_var: str,
        value: Any,
    ) -> None:
        """Copy variable `child_var` from `child` into `parent`, as `parent_var`."""
        child_port: BasePort = child[port_name]
        parent_port: BasePort = parent[port_name]
        lock_status = parent._locked
        parent._locked = False
        # Call `add_inward`, `add_outward`, etc., depending on context
        add_to_parent = getattr(parent, method_map[port_name])
        add_to_parent(parent_var, value)
        parent._locked = lock_status
        # Copy full variable detail in parent port
        parent_port.copy_variable_from(child_port, child_var, parent_var)
        log_debug(
            f"{parent.name}.{parent_var}",
            f"{parent.name}.{child.name}.{child_var}",
        )

    name_mapping = BaseConnector.format_mapping(pulling)

    for child_attr_name, parent_attr_name in name_mapping.items():
        child_attr = getattr(child, child_attr_name)

        if isinstance(child_attr, Port):
            if parent_attr_name not in parent:
                pulled_port = child_attr.copy(parent_attr_name)
                parent._add_port(pulled_port)
                log_debug(
                    f"Port {pulled_port.contextual_name}",
                    f"{parent.name}.{child_attr.contextual_name}",
                )
            else:
                pulled_port = getattr(parent, parent_attr_name)
            
            parent.connect(child_attr, pulled_port)

        else:  # inwards, outwards, or mode variables
            if isinstance(child_attr, BasePort):  # Pulling all variables
                port_name = child_attr.name
                parent_port = getattr(parent, port_name)

                for varname, value in child_attr.items():
                    if varname not in parent_port:
                        copy_variable(child, port_name, varname, varname, value)

                parent.connect(child_attr, parent_port, list(child_attr))

            else:  # Pulling individual variable
                var = child.name2variable[child_attr_name]
                port_name = var.mapping.name

                if parent_attr_name not in parent[port_name]:
                    copy_variable(child, port_name, child_attr_name, parent_attr_name, child_attr)

                parent.connect(child[port_name], parent[port_name], {child_attr_name: parent_attr_name})
