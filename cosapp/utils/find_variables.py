from __future__ import annotations
from fnmatch import fnmatchcase
from typing import Any, Callable, Union, TYPE_CHECKING
from collections.abc import Collection
import inspect
import itertools

from cosapp.utils.naming import natural_varname, CommonPorts
from cosapp.utils.helpers import check_arg

if TYPE_CHECKING:
    from cosapp.systems import System


SearchPattern = Union[str, list[str], type[None]]


def make_wishlist(wishlist: SearchPattern, name="wishlist") -> list[str]:
    ok = True
    if not wishlist:
        wishlist = []
    elif isinstance(wishlist, str):
        wishlist = [wishlist]
    elif isinstance(wishlist, Collection):
        ok = len(wishlist) == 0 or all(isinstance(item, str) for item in wishlist)
    else:
        ok = False
    if not ok:
        raise TypeError(
            f"{name!r} must be a string, or a sequence of strings; got {wishlist}."
        )
    # Filter out common port names "inwards.", "outwards.",.. from wishlist
    filtered = set(map(natural_varname, wishlist))
    return list(filtered)


def get_attributes(obj) -> set[str]:
    """Returns non-callable members of an object"""
    return set(
        m[0] for m in inspect.getmembers(obj)
        if not m[0].startswith("_") and not callable(m[1])
    )


def find_variables(
    system: System,
    includes: SearchPattern,
    excludes: SearchPattern,
    advanced_filter: Callable[[Any], bool] = lambda any: True,
    inputs: bool = True,
    outputs: bool = True,
    include_const: bool = False,
) -> dict[str, Any]:
    """Generate the dictionary (name, value) of variables
    whose names match inclusion and exclusion criteria.

    Parameters
    ----------
    system: cosapp.systems.System
        Object that owns the variables searched.
    includes: str or list[str]
        Variables matching these patterns will be included.
    excludes: str or list[str]
        Variables matching these patterns will be excluded.
    advanced_filter: Callable[[Any], bool]
        Function taking the variable as input and returning an acceptance criteria (True if variable is valid).
    inputs: bool
        Defines if input variables will be accepted or not.
    outputs: bool
        Defines if output variables will be accepted or not.
    include_const: bool
        Defines if read-only properties defined with `System.add_property` will be accepted or not.

    Returns
    -------
    dict[str, Any]
        Dictionary (name, value) of matches.

    .. note::
        Inward and outward variables will appear without the prefix `inwards.` or `outwards.`.
    """
    from cosapp.systems import System  # Local import to avoid recursion
    from cosapp.ports.port import BasePort

    check_arg(system, "system", System)

    if not (inputs or outputs):
        # quick return, if possible
        return []

    includes = make_wishlist(includes)
    excludes = make_wishlist(excludes)
    result = dict()

    def is_valid(port: Any) -> bool:
        return isinstance(port, BasePort) and (
            (port.is_input and inputs) or
            (port.is_output and outputs)
        )

    def find_matches(name: str, value: Any) -> dict[str, Any]:
        matches = dict()
        if advanced_filter(value):
            for pattern in includes:
                if fnmatchcase(name, pattern):
                    include = True
                    for pattern in excludes:
                        if fnmatchcase(name, pattern):
                            include = False
                            break
                    if include:
                        matches[name] = value
                        break
        return matches

    ports: set[BasePort] = set()

    for name, ref in system.name2variable.items():
        # Save variable for non virtual port
        # Suppress duplicates INWARDS and OUTWARDS
        port = ref.mapping
        valid = is_valid(port) and name == natural_varname(name)
        if not valid:
            continue  # skip `name`

        ports.add(port)
        result.update(find_matches(name, ref.value))

    # Find matches among system properties
    system_properties = find_system_properties(system, include_const)

    for name in system_properties:
        try:
            child, attr = name.rsplit(".", maxsplit=1)
        except ValueError:
            owner, attr = system, name
        else:
            owner = system[child]
        result.update(
            find_matches(name, getattr(owner, attr))
        )

    return result


def find_variable_names(
    system: System,
    includes: SearchPattern,
    excludes: SearchPattern,
    advanced_filter: Callable[[Any], bool] = lambda any: True,
    inputs: bool = True,
    outputs: bool = True,
    include_const: bool = False,
) -> list[str]:
    """Generate the list of requested variable names,
    given inclusion and exclusion criteria.

    Matching variable names are returned in alphabetical order.

    Parameters
    ----------
    system: cosapp.systems.System
        Object that owns the variables searched.
    includes: str or list[str]
        Variables matching these patterns will be included.
    excludes: str or list[str]
        Variables matching these patterns will be excluded.
    advanced_filter: Callable[[Any], bool]
        Function taking the variable as input and returning an acceptance criteria (True if variable is valid).
    inputs: bool
        Defines if input variables will be accepted or not.
    outputs: bool
        Defines if output variables will be accepted or not.
    include_const: bool
        Defines if read-only properties defined with `System.add_property` will be accepted or not.

    Returns
    -------
    list[str]
        Variable names in `system` matching the requested includes/excludes patterns.

    .. note::
        Inward and outward variables will appear without the prefix `inwards.` or `outwards.`.
        This functions returns the sorted keys of the dictionary returned by `find_variables`.
    """
    return sorted(
        find_variables(
            system,
            includes,
            excludes,
            advanced_filter,
            inputs,
            outputs,
            include_const,
        )
    )


def find_system_properties(system: System, include_const=False) -> set[str]:
    """
    Returns system properties, defined either as class properties
    (with @property decorator), or with `System.add_property`.
    The latter are excluded if optional argument `include_const`
    is False (default).

    Parameters
    ----------
    - system [cosapp.systems.System]:
        System of interest
    - include_const [bool, optional]:
        Defines if read-only properties defined with `System.add_property` will be accepted or not..
        Default: `False`.

    Returns
    -------
    - set[str]:
        set of property names.
    """
    from cosapp.systems import System  # Local import to avoid recursion
    check_arg(system, "system", System)

    base_cls_attr = get_attributes(System)
    get_name = lambda obj: obj.name

    def local_properties(system: System) -> set[str]:
        local_attr = get_attributes(system) - base_cls_attr
        writable = set(
            map(get_name,
                itertools.chain(
                    system.ports(),
                    system.events(),
                    system.children.values(),
                )
            )
        )
        local_attr -= writable
        for port_name in CommonPorts.names():
            local_attr -= set(system[port_name])
        if not include_const:
            local_attr -= set(system.properties)
        return local_attr

    tree = system.tree(downwards=True)
    properties = local_properties(next(tree))
    for child in tree:
        prefix = f"{system.get_path_to_child(child)}."
        properties.update(
            f"{prefix}{attr}"
            for attr in local_properties(child)
        )
    
    return properties
