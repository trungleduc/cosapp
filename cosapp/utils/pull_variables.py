import logging
from typing import Any, List, Union, Sequence, Dict

logger = logging.getLogger(__name__)


def pull_variables(
    child: "cosapp.systems.System", 
    pulling: Union[str, List[str], Dict[str, str]],
):
    """Pull variables from child to the parent.

    Parameters
    ----------
    child: System
        `System` asking to pull variables to its parent
    pulling: str or List[str] or Dict[str, str]
        Map of child ports to pulled ports at the parent system level
    selection: str or List[str] or Dict[str, str]

    """
    from cosapp.ports.port import Port, ExtensiblePort
    from cosapp.ports.enum import CommonPorts

    parent = child.parent
    if parent is None:
        raise AttributeError(
            f"Can't pull variables from orphan System {child.name!r}"
        )

    if isinstance(pulling, str):
        pulling = [pulling]
    if isinstance(pulling, Sequence):
        pulling = dict(zip(pulling, pulling))

    for child_port, parent_port in pulling.items():
        sink_port = child[child_port]

        if isinstance(sink_port, Port):
            if parent_port not in parent:
                pulled_port = sink_port.copy(parent_port)
                parent._add_port(pulled_port)
                logger.debug(
                    f"Port {pulled_port.contextual_name} will be duplicated from {sink_port.contextual_name}"
                    " - including validation range and scope."
                )
            else:
                pulled_port = parent[parent_port]
            parent.connect(sink_port, pulled_port)

        else:  # ExtensiblePort (inwards or outwards)
            def copy_variable(
                port: str,
                child: "cosapp.core.Module",
                child_var: str,
                parent_var: str,
                value: Any,
            ) -> None:
                details = child[port].get_details(child_var)
                args = (parent_var, value)
                # Validation criteria are removed to avoid warning duplication when checking
                kwargs = dict(
                    unit=details.unit,
                    dtype=details.dtype,
                    desc=details.description,
                    scope=details.scope,
                )
                old_locked = parent._locked
                parent._locked = False
                # Call add_inward or add_outward depending of the context
                getattr(parent, "add_" + port[:-1])(*args, **kwargs)
                parent._locked = old_locked
                
                logger.debug(
                    "{} {}.{} will be duplicated from {}.{} - including "
                    "validation range and scope.".format(
                        port, parent.name, parent_var, child.name, child_var
                    )
                )

            if isinstance(sink_port, ExtensiblePort):  # Pulling all inwards or outwards
                port_name = (
                    CommonPorts.INWARDS.value
                    if sink_port.name == CommonPorts.INWARDS.value
                    else CommonPorts.OUTWARDS.value
                )

                for variable in sink_port:
                    if variable not in parent[port_name]:
                        copy_variable(
                            port_name, child, variable, variable, sink_port[variable]
                        )

                parent.connect(sink_port, parent[port_name], list(sink_port))

            else:  # Pulling one inwards or one local
                owner = (
                    CommonPorts.INWARDS.value
                    if child_port in child[CommonPorts.INWARDS.value]
                    else CommonPorts.OUTWARDS.value
                )

                if parent_port not in parent[owner]:
                    copy_variable(owner, child, child_port, parent_port, sink_port)

                parent.connect(child[owner], parent[owner], {child_port: parent_port})
