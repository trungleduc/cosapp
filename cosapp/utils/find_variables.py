from fnmatch import fnmatchcase
from typing import List, Any, Callable, Set, Union
from collections.abc import Collection
import inspect

from cosapp.ports.enum import PortType, CommonPorts
from cosapp.ports.port import ExtensiblePort, Port


def natural_varname(name: str) -> str:
    """
    Strip references to 'inwards' and 'outwards' ports from variable name
    """
    inwards = CommonPorts.INWARDS.value
    outwards = CommonPorts.OUTWARDS.value
    name = name.strip()
    name = name.replace(f"{inwards}.", "")
    name = name.replace(f"{outwards}.", "")
    return name


def make_wishlist(wishlist: Union[str, List[str]], name="wishlist") -> List[str]:
    ok = True
    if isinstance(wishlist, str):
        wishlist = [wishlist]
    elif isinstance(wishlist, Collection):
        ok = len(wishlist) == 0 or all(isinstance(item, str) for item in wishlist)
    elif wishlist is None:
        wishlist = []
    else:
        ok = False
    if not ok:
        raise TypeError(
            f"{name!r} must be a string, or a sequence of strings; got {wishlist}."
        )
    # Filter out 'inwards' and 'outwards' in wishlist
    filtered = set(natural_varname(name) for name in wishlist)
    return list(filtered)


def get_attributes(obj) -> Set[str]:
    """Returns non-callable members of an object"""
    return set(
        m[0] for m in inspect.getmembers(obj) if not m[0].startswith("_") and not callable(m[1])
    )


def find_variables(
    watched_object: "cosapp.systems.System",
    includes: List[str],
    excludes: List[str],
    advanced_filter: Callable[[Any], bool] = lambda x: True,
    inputs: bool = True,
    outputs: bool = True,
    include_const: bool = False,
) -> List[str]:
    """Generate the list of requested variables.

    The variables are sorted in alphabetical order.

    Parameters
    ----------
    watched_object : cosapp.systems.System
        Object that owns the variables searched
    includes : list of str
        Variables matching these patterns will be included
    excludes : list of str
        Variables matching these patterns will be excluded
    advanced_filter : Callable[[Any], bool]
        Function taking the variable as input and returning an acceptance criteria (True if variable is valid)
    inputs : bool
        Defines if input variables will be accepted or not
    outputs : bool
        Defines if output variables will be accepted or not
    include_const : bool
        Defines if read-only properties defined with `System.add_property` will be accepted or not

    Returns
    -------
    List[str]
        Variable names matching the includes/excludes patterns of the user in the watched object.

    .. note::
        Inward and outward variables will appear without the prefix `inwards.` or `outwards.`.
    """
    from cosapp.systems.system import System, IterativeConnector  # Local import to avoid recursion

    if not (inputs or outputs):
        # quick return, if possible
        return []

    includes = make_wishlist(includes)
    excludes = make_wishlist(excludes)
    result = set()

    def is_valid(port):
        return isinstance(port, ExtensiblePort) and (
            (port.direction is PortType.IN and inputs) or
            (port.direction is PortType.OUT and outputs)
        )

    def find_matches(name: str, value: Any) -> Set[str]:
        result = set()
        if advanced_filter(value):
            for pattern in includes:
                if fnmatchcase(name, pattern):
                    include = True
                    for pattern in excludes:
                        if fnmatchcase(name, pattern):
                            include = False
                            break
                    if include:
                        result.add(name)
                        break
        return result

    ports = set()
    names = set()

    for name, ref in watched_object.name2variable.items():
        # Save variable for non virtual port and skip IterativeConnector as open_loops append before
        # Driver._precompute
        # Suppress duplicates INWARDS and OUTWARDS
        port = ref.mapping
        if (
            not is_valid(port)
            or isinstance(port.owner, IterativeConnector)
            or name != natural_varname(name)
        ):
            continue  # skip `name`

        ports.add(port)
        names.add(name)
        result |= find_matches(name, watched_object[name])
    
    Port_cls_attr = get_attributes(Port)

    for port in ports:
        if not isinstance(port, Port):
            continue  # exclude `inwards` and `outwards` extensible ports
        port_properties = get_attributes(port) - Port_cls_attr - set(port._variables)
        for attr in port_properties:
            result |= find_matches(f"{port.name}.{attr}", getattr(port, attr))

    obj_attributes = get_attributes(watched_object) - get_attributes(System)
    if not include_const:
        obj_attributes -= set(watched_object.properties)
    for attr in obj_attributes:
        result |= find_matches(attr, getattr(watched_object, attr))

    return sorted(list(result))
