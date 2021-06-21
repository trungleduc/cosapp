from fnmatch import fnmatchcase
from typing import List, Any, Callable, Set, Union
from collections.abc import Collection
import inspect

from cosapp.ports.enum import PortType
from cosapp.ports.port import BasePort, Port
from cosapp.utils.naming import natural_varname
from cosapp.utils.helpers import check_arg

SearchPattern = Union[str, List[str]]


def make_wishlist(wishlist: SearchPattern, name="wishlist") -> List[str]:
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
    # Filter out 'inwards.' and 'outwards.' in wishlist
    filtered = set(natural_varname(name) for name in wishlist)
    return list(filtered)


def get_attributes(obj) -> Set[str]:
    """Returns non-callable members of an object"""
    return set(
        m[0] for m in inspect.getmembers(obj)
        if not m[0].startswith("_") and not callable(m[1])
    )


def find_variables(
    watched_object: "cosapp.systems.System",
    includes: SearchPattern,
    excludes: SearchPattern,
    advanced_filter: Callable[[Any], bool] = lambda any: True,
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
    includes : str or List[str]
        Variables matching these patterns will be included
    excludes : str or List[str]
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
    check_arg(watched_object, 'watched_object', System)

    if not (inputs or outputs):
        # quick return, if possible
        return []

    includes = make_wishlist(includes)
    excludes = make_wishlist(excludes)
    result = set()

    def is_valid(port: BasePort) -> bool:
        return isinstance(port, BasePort) and (
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
        valid = (
            is_valid(port)
            and name == natural_varname(name)
            and not isinstance(port.owner, IterativeConnector)
        )
        if not valid:
            continue  # skip `name`

        ports.add(port)
        names.add(name)
        result |= find_matches(name, watched_object[name])
    
    # Find matches among port properties
    Port_cls_attr = get_attributes(Port)

    for port in ports:
        if not isinstance(port, Port):
            continue  # exclude `inwards` and `outwards` extensible ports
        port_properties = get_attributes(port) - Port_cls_attr - set(port._variables)
        port_name = port.full_name(trim_root=True)
        for name in port_properties:
            result |= find_matches(f"{port_name}.{name}", getattr(port, name))

    # Find matches among system properties
    system_properties = find_system_properties(watched_object, include_const)
    for name in system_properties:
        try:
            child, attr = name.rsplit('.', maxsplit=1)
        except ValueError:
            owner, attr = watched_object, name
        else:
            owner = watched_object[child]
        result |= find_matches(name, getattr(owner, attr))

    return sorted(result)


def find_system_properties(system, include_const=False) -> Set[str]:
    """
    Returns system properties, defined either as class properties
    (with @property decorator), or with `System.add_property`.
    The latter are excluded if optional argument `include_const`
    is False (default).
    """
    from cosapp.systems.system import System  # Local import to avoid recursion
    check_arg(system, 'system', System)

    base_cls_attr = get_attributes(System)

    def local_properties(system) -> set:
        local_attr = get_attributes(system) - base_cls_attr
        if not include_const:
            local_attr -= set(system.properties)
        return local_attr

    def find_properties(system, head=None):
        properties = set()
        if head is None:
            head = system
        prefix = "" if system is head else f"{head.get_path_to_child(system)}."
        properties |= set(f"{prefix}{attr}" for attr in local_properties(system))
        for child in system.children.values():
            properties |= find_properties(child, head)
        return properties

    return find_properties(system)
