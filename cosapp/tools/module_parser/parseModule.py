from cosapp.base import System, Port
from cosapp.ports.port import BasePort, BaseVariable
from cosapp.ports.enum import PortType
from cosapp.core.numerics.boundary import Unknown
from cosapp.core.numerics.residues import Residue
from cosapp.tools.views.markdown import PortMarkdownFormatter
from cosapp.utils.find_variables import make_wishlist

import json, os
import jsonschema
from pathlib import Path
from importlib import import_module
from fnmatch import fnmatchcase
from inspect import getmembers, isclass, ismodule
from typing import Set, Type, Tuple, Union, Dict, List, Any, Optional
from types import ModuleType

import logging

logger = logging.getLogger(__name__)


SearchPattern = Union[str, List[str]]
SystemConfig = Union[
    Dict[str, Dict[str, Any]],
    Dict[str, List[Dict[str, Any]]],
]


def format_kwargs(**kwargs):
    args = ", ".join(
        f"{key}={val}"
        for key, val in kwargs.items()
    )
    return f"({args})"


def extract_modules(module: ModuleType):
    """Extract all modules from `module`.
    
    Returns:
    --------
    List[module]
    """
    def recursive_search(module, modules: list):
        for _, obj in getmembers(module, ismodule):
            if obj.__name__.startswith(module.__name__) and obj not in modules:
                modules.append(obj)
                recursive_search(obj, modules)

    modules = [module]
    recursive_search(module, modules)
    return modules


def find_ports_and_systems(
    module: ModuleType,
    includes: SearchPattern = '*',
    excludes: SearchPattern = None,
) -> Tuple[Set[Type[System]], Set[Type[Port]]]:
    """Extract recursively all systems and ports from `module`

    Parameters:
    -----------
    - module [ModuleType]:
        Python module to be parsed.
    - includes [str or List[str]] (optional):
        System and port names matching these patterns will be included.
    - excludes [str or List[str]] (optional):
        System and port names matching these patterns will be excluded.
    
    Returns:
    --------
    systemSet, portSet: set[type[System]], set[type[Port]]
    """
    if not ismodule(module):
        raise TypeError('Argument is expected to be a module')

    includes = make_wishlist(includes)
    excludes = make_wishlist(excludes)
    def is_included(name: str) -> bool:
        include = False
        for pattern in includes:
            if fnmatchcase(name, pattern):
                include = True
                for pattern in excludes:
                    if fnmatchcase(name, pattern):
                        include = False
                        break
        return include
    
    def issubclass_strict(obj: type, base: type) -> bool:
        """Strict version of `issubclass`, returning `False` for base class."""
        return issubclass(obj, base) and obj is not base
    
    modules = extract_modules(module)
    systemSet, portSet = set(), set()
    for mod in modules:
        for _, obj in getmembers(mod):
            if isclass(obj) and is_included(obj.__name__):
                if issubclass_strict(obj, System):
                    systemSet.add(obj)
                if issubclass_strict(obj, Port):
                    portSet.add(obj)
    return systemSet, portSet


def get_data_from_class(
    dtype: Union[Type[System],Type[Port]],
    package_name: Optional[str] = None,
    ports: Optional[Set[Type[Port]]] = None,
    *args,
    **kwargs
):
    """Get informations from a system or a port

    Parameters:
    -----------
    - dtype [type[System | Port]]:
        System or Port class to be analyzed.
    - packageName [str] (optional):
       Custom package name.
    - ports [set[Port]] (optional):
        Used to check that all ports were found
        (in case a port is used in a children system not in the package for example).
    - *args, **kwargs:
        Additional arguments forwarded to class constructor, if required.

    Returns:
    --------
    dict[str, Any]
    """
    if ports is None:
        def register_port(port: BasePort):
            pass
    else:
        def register_port(port: BasePort):
            if isinstance(port, Port):
                ports.add(type(port))

    def get_all_port_data(portDict: Dict[str, BasePort]):
        portList = []
        for port in portDict.values():
            if len(port) > 0:
                portList.append(get_port_data(port))
                register_port(port)
        return portList or None

    sysPackage = package_name or dtype.__module__.split('.', maxsplit=1)[0]

    def get_port_data(port: BasePort, is_port_type=False):
        ptype = type(port)
        typename = ptype.__name__
        modname = ptype.__module__
        if is_port_type:
            portDict = {
                'name': typename,
            }
        else:
            portDict = {
                'name': port.name,
                'type': typename,
            }
        fixed_size_port = isinstance(port, Port)
        if fixed_size_port:
            portDict['pack'] = package_name or modname.split('.', maxsplit=1)[0]
        else:
            portDict['pack'] = sysPackage
        
        if not fixed_size_port or is_port_type:
            portDict['variables'] = get_port_var(port)
            desc = "\n".join(PortMarkdownFormatter(port).var_repr()[3:-1])
            if desc:
                portDict['desc'] = desc
        return portDict

    def get_var_data(variable: BaseVariable):
        data = {'name': variable.name}
        if (desc := variable.description):
            data['desc'] = desc
        if (unit := variable.unit):
            data['unit'] = unit
        return data

    def get_port_var(port: BasePort):
        return list(map(get_var_data, port.variables()))

    def get_shortest_import_path(modulePath: str, dtype: Type[System]) -> str:
        pathParts = modulePath.split('.')
        className = dtype.__name__

        upperModulePath = modulePath
        i = len(pathParts) -1
        while i > 0:
            currentPath = '.'.join(pathParts[:i])
            upperModule = import_module(currentPath)
            attr = getattr(upperModule, className, None)
            if attr is dtype:
                upperModulePath = currentPath
            i -= 1

        return upperModulePath

    def get_system_doc(systemClass: Type[System]) -> str:
        indent = 0
        doc = systemClass.__doc__
        strippedDoc = []
        if doc:
            for line in doc.split("\n"):
                if indent == 0:
                    strippedLine = line.lstrip()
                    if len(strippedLine) > 0:
                        indent = len(line) - len(strippedLine)
                else:
                    strippedLine = line[indent:]
                strippedDoc.append(strippedLine)
            return "\n".join(strippedDoc)
        return None

    def get_math_problem(system: System) -> Dict[str, Any]:
        """Extract mathematical problem metadata from `system`"""
        def extract_data(obj: Union[Unknown, Residue]):
            objDict = obj.to_dict()
            return { 'context': objDict['context'], 'content': objDict['name'] }
        
        def dict_to_list(objList: Union[Dict[str, Unknown], Dict[str, Residue]]):
            return list(map(extract_data, objList.values()))
    
        system.open_loops()
        problem = system.assembled_problem()
        system.close_loops()

        mathProblemDict = {}
        if (unknowns := problem.unknowns):
            mathProblemDict['unknowns'] = dict_to_list(unknowns)
            mathProblemDict['nUnknowns'] = problem.n_unknowns
        if (residues := problem.residues):
            mathProblemDict['equations'] = dict_to_list(residues)
            mathProblemDict['nEquations'] = problem.n_equations

        return mathProblemDict

    if issubclass(dtype, System):
        systemType = dtype.__name__
        systemName = kwargs.pop('__alias__', systemType)
        system = dtype('bogus', *args, **kwargs)
        desc = get_system_doc(dtype)
        inputs = get_all_port_data(system.inputs)
        outputs = get_all_port_data(system.outputs)
        mathProblemDict = get_math_problem(system)
        dtypeDict = {
            'name': systemName,
            'className': systemType,
            'pack': sysPackage,
            'mod': get_shortest_import_path(dtype.__module__, dtype),
        }
        if desc:
            dtypeDict['desc'] = desc
        if inputs:
            dtypeDict['inputs'] = inputs
        if outputs:
            dtypeDict['outputs'] = outputs
        if mathProblemDict:
            dtypeDict['mathProblem'] = mathProblemDict

    else:
        port = dtype('bogus', direction=PortType.IN, *args, **kwargs)
        dtypeDict = get_port_data(port, is_port_type=True)

    if kwargs:
        dtypeDict['kwargs'] = kwargs

    return dtypeDict


def get_data_from_module(
    module: ModuleType,
    ctor_config: SystemConfig = {},
    package_name: Optional[str] = None,
    includes: SearchPattern = '*',
    excludes: SearchPattern = None,
) -> Dict[str, Any]:
    """Extract metadata of all systems and ports found in `module`.

    Parameters:
    -----------
    - module [ModuleType]:
        Python module to be parsed.
    - ctor_config [dict[str, Any] | dict[str, list[dict[str, any]]]] (optional):
        Dictionary or list of dictionaries containing kwargs required for system/port
        construction (if any), referenced by class names (keys).
        If the dictionary contains key '__alias__', the class will be
        renamed into the associated value.
    - packageName [str] (optional):
        Custom package name.
    - includes [str or List[str]] (optional):
        System and port names matching these patterns will be included.
    - excludes [str or List[str]] (optional):
        System and port names matching these patterns will be excluded.

    Returns:
    --------
    dict[str, Any]
    """
    if package_name is None:
        package_name = module.__name__
    
    def get_all_class_data(classSet: Set[Type], ctor_config: SystemConfig, ports=None):
        result = []
        for cls in classSet:
            cls_kwargs_tab = ctor_config.get(cls.__name__, {})
            if not isinstance(cls_kwargs_tab, list):
                cls_kwargs_tab = [cls_kwargs_tab]
            for cls_kwargs in cls_kwargs_tab:
                try:
                    cls_data = get_data_from_class(cls, package_name=package_name, ports=ports, **cls_kwargs)
                except:
                    logger.info(f"Could not instantiate `{cls.__name__}`; skipped")
                    continue
                if len(cls_kwargs_tab) > 1:
                    if not cls_kwargs.pop('__alias__', None):
                        cls_data['name'] += f' {format_kwargs(**cls_kwargs)}'
                result.append(cls_data)
        return result

    systemSet, portSet = find_ports_and_systems(module, includes, excludes)
    metadata = {
        'name': package_name,
        'systems': get_all_class_data(systemSet, ctor_config, ports=portSet),
    }
    # Add port metadata *after* system metadata
    metadata['ports'] = get_all_class_data(portSet, ctor_config)

    try:
        metadata['version'] = module.__version__
    except AttributeError:
        pass

    pwd = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(pwd, "package_metadata.schema.json")) as fp:
        config_schema = json.load(fp)

    jsonschema.validate(metadata, config_schema)

    return metadata


def parse_module(
    module: ModuleType,
    ctor_config: Optional[SystemConfig] = None,
    package_name: Optional[str] = None,
    includes: Optional[SearchPattern] = None,
    excludes: Optional[SearchPattern] = None,
    path: Optional[Union[str, Path]] = None,
) -> None:
    """Creates a json file containing the metadata
    of all systems and ports found in `module`.
    
    Parameters:
    -----------
    - module [ModuleType]:
        Python module to be parsed.
    - ctor_config [dict[str, list[dict[str, any]] | dict[str, Any]]] (optional):
        Dictionary or list of dictionaries containing kwargs required for system/port
        construction (if any), referenced by class names (keys).
        If the dictionary contains key '__alias__', the class will be
        renamed into the associated value.
    - package_name [str] (optional):
        Custom package name.
    - includes [str or List[str]] (optional):
        System and port names matching these patterns will be included.
    - excludes [str or List[str]] (optional):
        System and port names matching these patterns will be excluded
        (ports used by included systems will always be included).
    - path [str | pathlib.Path] (optional):
        Optional path of output file <packageName or module.__name__>.json
        (current directory by default).

    Use pre-defined settings
    ------------------------
    Pre-defined values of optional arguments `ctor_config`, `package_name`, `includes` and `excludes`
    may be specified at module level by implementing hook function `_parse_module_config`,
    returning preset values in a dictionary of the kind {option: value}.
    Typically, `_parse_module_config` may be defined in the `__init__.py` file of the module.
    
    Examples:
    ---------
    >>> parse_module(module1)
    >>>
    >>> parse_module(
    >>>     module2,
    >>>     ctor_config = {
    >>>         'SystemA': dict(n=2, x=0.5),
    >>>         'SystemB': [
    >>>             dict(foo=0),
    >>>             dict(foo=None, __alias__='SystemB [default]'),
    >>>         ],
    >>>     },
    >>> )
    """
    settings = dict(
        ctor_config=ctor_config,
        package_name=package_name,
        includes=includes,
        excludes=excludes,
    )
    default_settings = dict(
        ctor_config={},
        includes='*',
        excludes=None,
        package_name=module.__name__,
    )
    try:
        # Check if module has a pre-defined settings
        get_module_settings = getattr(module, "_parse_module_config")
    except AttributeError:
        pass
    else:
        logger.info("Use pre-defined settings returned by `_parse_module_config`")
        default_settings.update(get_module_settings())

    # Apply user settings, if specified
    for name, value in settings.items():
        if value is None:
            settings[name] = default_settings[name]

    metadata = get_data_from_module(module, **settings)
    package_name = settings['package_name']

    try:
        version = metadata['version']
    except KeyError:
        filename = f"{package_name}.json"
    else:
        filename = f"{package_name} - {version}.json"

    if path and os.path.isdir(path):
        filename = Path(path) / filename

    with open(filename, "w") as fp:
        json.dump(metadata, fp, indent=4)

    logger.info(f"Created file {filename}")
