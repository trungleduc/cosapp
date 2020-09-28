from fnmatch import fnmatchcase
from typing import List, Any, Callable

from cosapp.ports.enum import PortType, CommonPorts
from cosapp.ports.port import ExtensiblePort


def search_for_variables(
    watched_object: "cosapp.core.module.Module",
    includes: List[str],
    excludes: List[str],
    advanced_filter: Callable[[Any], bool] = lambda x: True,
    inputs: bool = True,
    outputs: bool = True,
) -> List[str]:
    """Generate the list of requested variables.

    The variable are sorted in alphabetical order.

    Parameters
    ----------
    watched_object : cosapp.core.module.Module
        Object that owns the variables searched
    includes : list of str
        Variables matching these patterns will be included
    excludes : list of str
        Variables matching these patterns will be excluded
    advanced_filter : Callable[[Any], bool]
        Function taking the variable as input and returning an acceptance criteria (True == variable is valid)
    inputs : bool
        Defines if input variables will be accepted or not
    outputs : bool
        Defines if output variables will be accepted or not

    Returns
    -------
    List[str]
        Variable names matching the includes/excludes patterns of the user in the watched object.

    .. note::
        Inward and outward variables will appear without the prefix `inwards.` or `outwards.`.
    """
    from cosapp.systems.system import (
        IterativeConnector
    )  # Local import to avoid forward reference

    tmp = set()
    for name, ref in watched_object.name2variable.items():
        # Save variable for non virtual port and skip IterativeConnector as open_loops append before
        # Driver._precompute
        # Suppress duplicates INWARDS and OUTWARDS
        if (
            isinstance(ref.mapping, ExtensiblePort)
            and not isinstance(ref.mapping.owner, IterativeConnector)
            and CommonPorts.INWARDS.value.join("..") not in name
            and CommonPorts.OUTWARDS.value.join("..") not in name
            and not name.startswith(CommonPorts.INWARDS.value + ".")
            and not name.startswith(CommonPorts.OUTWARDS.value + ".")
            and (
                (ref.mapping.direction == PortType.IN and inputs)
                or (ref.mapping.direction == PortType.OUT and outputs)
            )
        ):
            for include in includes:
                if fnmatchcase(name, include) and advanced_filter(watched_object[name]):
                    include = True
                    for exclude in excludes:
                        if fnmatchcase(name, exclude):
                            include = False
                            break
                    if include:
                        tmp.add(name)
    return sorted(tmp)
